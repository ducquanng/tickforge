"""Synthetic trade tape for tests and offline demos.

Prices follow a random walk with GARCH-like volatility clustering, and a
small part of each return is driven by lagged order flow, so there is a real
but weak signal for models to find.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def synthetic_trades(n: int = 200_000, seed: int = 0, start_ts: int = 1_700_000_000_000, flow_impact: float = 0.3) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    ts = start_ts + np.cumsum(rng.exponential(250.0, n)).astype(np.int64)
    var = np.empty(n)
    shock = rng.standard_normal(n)
    v = 1.0
    for i in range(n):
        var[i] = v
        v = 0.02 + 0.93 * v + 0.05 * v * shock[i] ** 2
    # Persistent aggressor side (order-splitting), which then moves the price.
    flow = np.sign(pd.Series(rng.standard_normal(n)).ewm(span=50).mean().to_numpy() + 0.3 * rng.standard_normal(n))
    step = 1e-5 * (np.sqrt(var) * shock + flow_impact * np.roll(flow, 1))
    price = np.round(100.0 * np.exp(np.cumsum(step)), 2)
    return pd.DataFrame(
        {
            "ts": ts,
            "price": price,
            "qty": np.round(rng.lognormal(-2.0, 1.0, n), 4) + 1e-4,
            "buyer_maker": flow < 0,
            "agg_id": np.arange(n, dtype=np.int64),
        }
    )
