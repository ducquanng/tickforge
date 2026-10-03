import datetime as dt

import numpy as np
import pandas as pd
import pytest

from tickforge.data.store import TickStore
from tickforge.data.synthetic import synthetic_trades

DAY_MS = 86_400_000


@pytest.fixture(scope="session")
def trades() -> pd.DataFrame:
    return synthetic_trades(n=300_000, seed=1)


@pytest.fixture(scope="session")
def synthetic_store(tmp_path_factory) -> tuple[TickStore, dt.date, dt.date]:
    """Five days of synthetic ticks for two symbols, written like real ingested data."""
    root = tmp_path_factory.mktemp("store")
    store = TickStore(root)
    start = dt.date(2024, 1, 1)
    t0 = int(dt.datetime(2024, 1, 1, tzinfo=dt.timezone.utc).timestamp() * 1000)
    for seed, symbol in enumerate(("AAAUSDT", "BBBUSDT")):
        df = synthetic_trades(n=1_900_000, seed=seed, start_ts=t0)
        day = (df["ts"].to_numpy() - t0) // DAY_MS
        for d in range(5):
            store.write(df[day == d], "um", "aggTrades", symbol, start + dt.timedelta(days=d))
        rng = np.random.default_rng(seed)
        ts = np.arange(t0, t0 + 5 * DAY_MS, 30_000)
        rows = []
        for pct in (-5.0, -1.0, -0.2, 0.2, 1.0, 5.0):
            rows.append(pd.DataFrame({"ts": ts, "pct": pct, "depth": 1.0, "notional": rng.lognormal(10, 0.3, ts.size)}))
        depth = pd.concat(rows).sort_values("ts")
        dday = (depth["ts"].to_numpy() - t0) // DAY_MS
        for d in range(5):
            store.write(depth[dday == d], "um", "bookDepth", symbol, start + dt.timedelta(days=d))
    return store, start, start + dt.timedelta(days=4)
