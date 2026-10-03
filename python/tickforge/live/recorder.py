"""Live L2 order book capture.

Full-depth history is not in Binance's free archive, but the live feed is
free: this module subscribes to the spot diff-depth stream, synchronises it
with a REST snapshot following Binance's documented procedure, maintains the
book in the Rust ``OrderBook`` and records top-of-book features to Parquet.

It uses Binance's market-data-only endpoints, which need no account.

The sequencing logic lives in ``BookSync`` and has no I/O, so it is unit
tested offline; ``record`` is the thin async shell around it.
"""

from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd
import requests

from tickforge._core import OrderBook

REST_URL = "https://data-api.binance.vision/api/v3/depth"
WS_URL = "wss://data-stream.binance.vision/ws"


class SequenceGap(RuntimeError):
    """A depth update was missed; the book must be rebuilt from a new snapshot."""


@dataclass
class BookSync:
    """Applies diff-depth events on top of a snapshot.

    Rules (Binance spot): discard events with ``u <= lastUpdateId``; the first
    applied event must straddle ``lastUpdateId + 1``; after that each event's
    ``U`` must equal the previous event's ``u + 1``.
    """

    tick_size: float
    book: OrderBook = field(init=False)
    last_id: int | None = None
    synced: bool = False

    def __post_init__(self):
        self.book = OrderBook(self.tick_size)

    def load_snapshot(self, snapshot: dict) -> None:
        self.book.clear()
        self.book.update_many("bid", [(float(p), float(q)) for p, q in snapshot["bids"]])
        self.book.update_many("ask", [(float(p), float(q)) for p, q in snapshot["asks"]])
        self.last_id = int(snapshot["lastUpdateId"])
        self.synced = False

    def apply(self, event: dict) -> bool:
        """Apply one event. Returns False if it was stale and skipped."""
        if self.last_id is None:
            raise RuntimeError("load a snapshot first")
        first, final = int(event["U"]), int(event["u"])
        if final <= self.last_id:
            return False
        if first > self.last_id + 1:
            raise SequenceGap(f"expected update {self.last_id + 1}, got {first}")
        self.book.update_many("bid", [(float(p), float(q)) for p, q in event["b"]])
        self.book.update_many("ask", [(float(p), float(q)) for p, q in event["a"]])
        self.last_id = final
        self.synced = True
        return True

    def row(self, event_ms: int, recv_ms: int) -> dict:
        bid, ask = self.book.best_bid(), self.book.best_ask()
        return {
            "ts": event_ms,
            "recv_ts": recv_ms,
            "bid": bid[0],
            "bid_qty": bid[1],
            "ask": ask[0],
            "ask_qty": ask[1],
            "microprice": self.book.microprice(),
            "imbalance_1": self.book.imbalance(1),
            "imbalance_5": self.book.imbalance(5),
            "imbalance_20": self.book.imbalance(20),
        }


async def record(symbol: str, seconds: float, out: str | Path, tick_size: float = 0.01, depth_limit: int = 1000) -> pd.DataFrame:
    """Record ``seconds`` of top-of-book features for ``symbol`` to a Parquet file."""
    import websockets  # optional dependency: pip install tickforge[live]

    sync = BookSync(tick_size)
    rows: list[dict] = []
    url = f"{WS_URL}/{symbol.lower()}@depth@100ms"
    deadline = time.monotonic() + seconds
    async with websockets.connect(url, max_queue=4096) as ws:
        # Start buffering the stream before taking the snapshot, as Binance requires.
        first = json.loads(await ws.recv())
        buffered = [first]
        while True:
            snap = await asyncio.to_thread(lambda: requests.get(REST_URL, params={"symbol": symbol.upper(), "limit": depth_limit}, timeout=10).json())
            if int(snap["lastUpdateId"]) >= int(first["U"]):
                break
            buffered.append(json.loads(await ws.recv()))
        sync.load_snapshot(snap)
        while time.monotonic() < deadline:
            event = buffered.pop(0) if buffered else json.loads(await ws.recv())
            if sync.apply(event):
                rows.append(sync.row(int(event["E"]), time.time_ns() // 1_000_000))
    df = pd.DataFrame(rows)
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, compression="zstd", index=False)
    return df
