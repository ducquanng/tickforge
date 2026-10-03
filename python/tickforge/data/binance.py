"""Downloader for Binance's free public data archive (data.binance.vision).

Two datasets are used:

* ``aggTrades``: every aggregated trade with its aggressor side.
* ``bookDepth``: USD-M futures only; cumulative resting size within fixed
  percentage bands around mid (0.2% and 1..5% each side), snapshotted roughly
  every 30 seconds.

Historical top-of-book quotes and full L2 depth are *not* in the free archive.
The archive's files have changed format over time (header row or not,
millisecond or microsecond timestamps), so parsing normalises everything to
one schema.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import io
import time
import zipfile
from dataclasses import dataclass

import numpy as np
import pandas as pd
import requests

BASE_URL = "https://data.binance.vision/data"
MARKETS = {"spot": "spot", "um": "futures/um", "cm": "futures/cm"}
AGG_COLUMNS = ["agg_id", "price", "qty", "first_id", "last_id", "ts", "buyer_maker"]


class ChecksumError(RuntimeError):
    pass


def to_millis(ts: np.ndarray) -> np.ndarray:
    """Normalise epoch timestamps given in ms or us to ms."""
    ts = np.asarray(ts, dtype=np.int64)
    if ts.size and ts.max() > 10**14:  # microseconds
        return ts // 1000
    return ts


def parse_agg_trades(raw: bytes) -> pd.DataFrame:
    """Parse an aggTrades CSV into the normalised trade schema.

    Output columns: ``ts`` (int64 epoch ms), ``price``, ``qty`` (float64),
    ``buyer_maker`` (bool; True means the seller was the aggressor),
    ``agg_id`` (int64). Sorted by (ts, agg_id).
    """
    has_header = raw[:1].isalpha()
    df = pd.read_csv(io.BytesIO(raw), header=0 if has_header else None)
    df = df.iloc[:, :7]
    df.columns = AGG_COLUMNS
    bm = df["buyer_maker"]
    if bm.dtype != bool:
        bm = bm.astype(str).str.lower().map({"true": True, "false": False})
        if bm.isna().any():
            raise ValueError("unparseable buyer_maker flag")
    out = pd.DataFrame(
        {
            "ts": to_millis(df["ts"].to_numpy()),
            "price": df["price"].to_numpy(dtype=np.float64),
            "qty": df["qty"].to_numpy(dtype=np.float64),
            "buyer_maker": bm.to_numpy(dtype=bool),
            "agg_id": df["agg_id"].to_numpy(dtype=np.int64),
        }
    )
    return out.sort_values(["ts", "agg_id"], kind="stable").reset_index(drop=True)


def parse_book_depth(raw: bytes) -> pd.DataFrame:
    """Parse a bookDepth CSV. Output: ``ts`` (epoch ms), ``pct``, ``depth``, ``notional``."""
    df = pd.read_csv(io.BytesIO(raw))
    ts = pd.to_datetime(df["timestamp"], utc=True)
    return pd.DataFrame(
        {
            "ts": ts.dt.as_unit("ms").astype("int64").to_numpy(),
            "pct": df["percentage"].to_numpy(dtype=np.float64),
            "depth": df["depth"].to_numpy(dtype=np.float64),
            "notional": df["notional"].to_numpy(dtype=np.float64),
        }
    )


PARSERS = {"aggTrades": parse_agg_trades, "bookDepth": parse_book_depth}


@dataclass
class BinancePublicData:
    """Client for the daily files of the public archive."""

    market: str = "um"
    timeout: float = 120.0
    retries: int = 3
    verify_checksum: bool = True

    def url(self, kind: str, symbol: str, day: dt.date) -> str:
        if self.market not in MARKETS:
            raise ValueError(f"unknown market {self.market!r}")
        return f"{BASE_URL}/{MARKETS[self.market]}/daily/{kind}/{symbol}/{symbol}-{kind}-{day.isoformat()}.zip"

    def _get(self, url: str) -> bytes | None:
        last: Exception | None = None
        for attempt in range(self.retries):
            try:
                r = requests.get(url, timeout=self.timeout)
                if r.status_code == 404:
                    return None
                r.raise_for_status()
                return r.content
            except requests.RequestException as e:  # transient network error
                last = e
                time.sleep(2**attempt)
        raise RuntimeError(f"download failed: {url}") from last

    def fetch(self, kind: str, symbol: str, day: dt.date) -> pd.DataFrame | None:
        """Download and parse one day. Returns None if the file does not exist."""
        if kind not in PARSERS:
            raise ValueError(f"unsupported dataset {kind!r}")
        url = self.url(kind, symbol, day)
        blob = self._get(url)
        if blob is None:
            return None
        if self.verify_checksum:
            expected = self._get(url + ".CHECKSUM")
            if expected is not None:
                digest = hashlib.sha256(blob).hexdigest()
                if digest != expected.decode().split()[0]:
                    raise ChecksumError(f"sha256 mismatch for {url}")
        with zipfile.ZipFile(io.BytesIO(blob)) as zf:
            raw = zf.read(zf.namelist()[0])
        return PARSERS[kind](raw)


def date_range(start: dt.date, end: dt.date) -> list[dt.date]:
    """Inclusive list of days."""
    return [start + dt.timedelta(days=i) for i in range((end - start).days + 1)]
