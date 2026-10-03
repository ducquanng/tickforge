"""Bar construction: thin pandas wrapper over the Rust aggregator."""

from __future__ import annotations

import numpy as np
import pandas as pd

from tickforge import _core


def make_bars(trades: pd.DataFrame, kind: str = "time", threshold: float = 60_000) -> pd.DataFrame:
    """Aggregate normalised trades (see ``data.binance.parse_agg_trades``) into bars.

    ``kind`` is "time" (threshold in ms), "tick", "volume" or "dollar".
    ``close_ts`` is the first instant at which a bar is fully known; every
    downstream timestamp join keys on it.
    """
    out = _core.build_bars(
        np.ascontiguousarray(trades["ts"].to_numpy(dtype=np.int64)),
        np.ascontiguousarray(trades["price"].to_numpy(dtype=np.float64)),
        np.ascontiguousarray(trades["qty"].to_numpy(dtype=np.float64)),
        np.ascontiguousarray(trades["buyer_maker"].to_numpy(dtype=bool)),
        kind,
        float(threshold),
    )
    bars = pd.DataFrame(out)
    bars["sell_volume"] = bars["volume"] - bars["buy_volume"]
    bars["vwap"] = bars["dollar"] / bars["volume"]
    return bars


def regular_grid(bars: pd.DataFrame, interval: int) -> pd.DataFrame:
    """Reindex time bars onto a gap-free grid.

    Intervals without trades get zero volume and carry the last close forward,
    which is exactly what an observer would have known at that time.
    """
    if bars.empty:
        return bars
    grid = np.arange(bars["close_ts"].iloc[0], bars["close_ts"].iloc[-1] + interval, interval)
    out = bars.set_index("close_ts").reindex(grid)
    empty = out["close"].isna()
    out["close"] = out["close"].ffill()
    for col in ("open", "high", "low", "vwap"):
        out[col] = out[col].fillna(out["close"])
    for col in ("volume", "dollar", "buy_volume", "sell_volume", "n_trades"):
        out[col] = out[col].fillna(0.0)
    out["open_ts"] = out["open_ts"].fillna(pd.Series(grid - interval, index=grid)).astype(np.int64)
    out["n_trades"] = out["n_trades"].astype(np.int64)
    out["empty"] = empty
    return out.rename_axis("close_ts").reset_index()
