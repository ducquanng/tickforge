"""Partitioned Parquet store with DuckDB as the query layer.

Layout: ``<root>/<market>/<kind>/symbol=<SYMBOL>/date=<YYYY-MM-DD>.parquet``.
One file per symbol-day keeps ingestion idempotent and lets DuckDB prune
partitions from the path.
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import duckdb
import pandas as pd

from tickforge.data.binance import BinancePublicData, date_range


class TickStore:
    def __init__(self, root: str | Path):
        self.root = Path(root)

    def path(self, market: str, kind: str, symbol: str, day: dt.date) -> Path:
        return self.root / market / kind / f"symbol={symbol}" / f"date={day.isoformat()}.parquet"

    def has(self, market: str, kind: str, symbol: str, day: dt.date) -> bool:
        return self.path(market, kind, symbol, day).exists()

    def write(self, df: pd.DataFrame, market: str, kind: str, symbol: str, day: dt.date) -> Path:
        p = self.path(market, kind, symbol, day)
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(".tmp")
        df.to_parquet(tmp, compression="zstd", index=False)
        tmp.replace(p)  # atomic: a crashed download never leaves a partial file
        return p

    def ingest(
        self, kind: str, symbol: str, start: dt.date, end: dt.date, market: str = "um", client: BinancePublicData | None = None
    ) -> dict[str, int]:
        """Download missing days. Returns counts of written / skipped / missing."""
        client = client or BinancePublicData(market=market)
        stats = {"written": 0, "skipped": 0, "missing": 0}
        for day in date_range(start, end):
            if self.has(market, kind, symbol, day):
                stats["skipped"] += 1
                continue
            df = client.fetch(kind, symbol, day)
            if df is None:
                stats["missing"] += 1
                continue
            self.write(df, market, kind, symbol, day)
            stats["written"] += 1
        return stats

    def read(self, kind: str, symbol: str, start: dt.date, end: dt.date, market: str = "um") -> pd.DataFrame:
        """Load a date range (inclusive), ordered by time."""
        files = [self.path(market, kind, symbol, d) for d in date_range(start, end)]
        files = [f for f in files if f.exists()]
        if not files:
            raise FileNotFoundError(f"no {kind} data for {symbol} between {start} and {end}")
        return pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)

    def sql(self, query: str) -> pd.DataFrame:
        """Run DuckDB SQL. ``{root}`` in the query expands to the store root, e.g.
        ``SELECT symbol, count(*) FROM read_parquet('{root}/um/aggTrades/*/*.parquet', hive_partitioning=true) GROUP BY 1``.
        """
        return duckdb.sql(query.format(root=self.root.as_posix())).df()
