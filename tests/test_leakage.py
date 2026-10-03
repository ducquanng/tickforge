"""Lookahead tests. A feature at time t must not move when data after t changes."""

import numpy as np
import pandas as pd
import pytest

from tickforge.bars import make_bars, regular_grid
from tickforge.features import build_features, depth_features
from tickforge.labels import forward_log_rv, forward_return


@pytest.fixture(scope="module")
def bars(trades):
    return regular_grid(make_bars(trades, "time", 5_000), 5_000)


def _depth(bars, seed=0):
    rng = np.random.default_rng(seed)
    ts = np.arange(bars["close_ts"].iloc[0] - 60_000, bars["close_ts"].iloc[-1], 7_000)
    return pd.concat([pd.DataFrame({"ts": ts, "pct": p, "depth": 1.0, "notional": rng.lognormal(10, 0.5, ts.size)}) for p in (-1.0, -0.2, 0.2, 1.0)])


@pytest.mark.parametrize("cut", [0.5, 0.8, 0.97])
def test_features_are_invariant_to_truncating_the_future(bars, cut):
    depth = _depth(bars)
    other = bars.assign(close=bars["close"] * 1.01)
    full = build_features(bars, depth, {"x": other})
    k = int(len(bars) * cut)
    t_cut = bars["close_ts"].iloc[k - 1]
    part = build_features(bars.iloc[:k], depth[depth["ts"] < t_cut], {"x": other.iloc[:k]})
    pd.testing.assert_frame_equal(part, full.iloc[:k], check_exact=False, rtol=1e-9, atol=1e-12)


def test_features_are_invariant_to_corrupting_the_future(bars):
    k = len(bars) // 2
    corrupted = bars.copy()
    cols = ["open", "high", "low", "close", "volume", "dollar", "buy_volume", "sell_volume"]
    corrupted.loc[k:, cols] = corrupted.loc[k:, cols] * 3.0
    a, b = build_features(bars), build_features(corrupted)
    pd.testing.assert_frame_equal(a.iloc[:k], b.iloc[:k])
    assert not a.iloc[k:].equals(b.iloc[k:])  # the test has teeth


def test_depth_join_is_strictly_backward():
    depth = pd.concat([pd.DataFrame({"ts": [1000, 2000], "pct": p, "depth": 1.0, "notional": n}) for p, n in ((-1.0, [3.0, 1.0]), (1.0, [1.0, 3.0]))])
    f = depth_features(depth, pd.Index([1000, 1001, 2000, 2001], name="close_ts"))
    # At t=1000 the 1000 snapshot is not yet usable; at t=2000 only the 1000 one is.
    assert np.isnan(f["depth_imb_1"].iloc[0])
    np.testing.assert_allclose(f["depth_imb_1"].iloc[1:].to_numpy(), [0.5, 0.5, -0.5])


def test_stale_depth_is_dropped():
    depth = pd.concat([pd.DataFrame({"ts": [1000], "pct": p, "depth": 1.0, "notional": 1.0}) for p in (-1.0, 1.0)])
    f = depth_features(depth, pd.Index([2000, 400_000], name="close_ts"))
    assert f["depth_imb_1"].notna().tolist() == [True, False]


def test_labels_look_forward_by_exactly_the_horizon():
    close = pd.Series(np.exp(np.arange(20) * 0.01))
    y = forward_return(close, 3)
    np.testing.assert_allclose(y.iloc[:17], 0.03)
    assert y.iloc[17:].isna().all()
    bumped = close.copy()
    bumped.iloc[10] *= 2
    changed = (forward_log_rv(close, 3) - forward_log_rv(bumped, 3)).abs() > 1e-9
    # A shock in bar 10 touches returns 10 and 11, hence labels 7..10 only.
    assert changed[changed].index.tolist() == [7, 8, 9, 10]
