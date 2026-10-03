import numpy as np
import pandas as pd
import pytest

from tickforge.bars import make_bars, regular_grid


def test_time_bars_match_pandas_reference(trades):
    bars = make_bars(trades, "time", 60_000)
    bucket = trades["ts"] // 60_000
    ref = trades.assign(dollar=trades["price"] * trades["qty"], buy=np.where(trades["buyer_maker"], 0.0, trades["qty"])).groupby(bucket)
    ref = ref.agg(
        open=("price", "first"),
        high=("price", "max"),
        low=("price", "min"),
        close=("price", "last"),
        volume=("qty", "sum"),
        dollar=("dollar", "sum"),
        buy=("buy", "sum"),
        n=("qty", "size"),
    ).iloc[:-1]
    assert len(bars) == len(ref)
    np.testing.assert_array_equal(bars["close_ts"], (ref.index.to_numpy() + 1) * 60_000)
    for a, b in [("open", "open"), ("high", "high"), ("low", "low"), ("close", "close")]:
        np.testing.assert_array_equal(bars[a], ref[b])
    np.testing.assert_allclose(bars["volume"], ref["volume"])
    np.testing.assert_allclose(bars["dollar"], ref["dollar"])
    np.testing.assert_allclose(bars["buy_volume"], ref["buy"])
    np.testing.assert_array_equal(bars["n_trades"], ref["n"])


@pytest.mark.parametrize("kind,threshold", [("tick", 500), ("volume", 100.0), ("dollar", 10_000.0)])
def test_information_bars_conserve_volume(trades, kind, threshold):
    bars = make_bars(trades, kind, threshold)
    assert len(bars) > 10
    used = int(bars["n_trades"].sum())
    np.testing.assert_allclose(bars["volume"].sum(), trades["qty"].iloc[:used].sum())
    assert (bars["high"] >= bars[["open", "close"]].max(axis=1)).all()
    assert (bars["low"] <= bars[["open", "close"]].min(axis=1)).all()
    assert bars["close_ts"].is_monotonic_increasing


def test_unknown_kind_raises(trades):
    with pytest.raises(ValueError):
        make_bars(trades, "renko", 1)


def test_regular_grid_fills_gaps_without_inventing_volume():
    t = pd.DataFrame(
        {
            "ts": [0, 10, 180_000, 180_500, 240_001],
            "price": [1.0, 2.0, 3.0, 4.0, 5.0],
            "qty": [1.0] * 5,
            "buyer_maker": [True, False, True, False, True],
        }
    )
    g = regular_grid(make_bars(t, "time", 60_000), 60_000)
    assert g["close_ts"].tolist() == [60_000, 120_000, 180_000, 240_000]
    assert g["empty"].tolist() == [False, True, True, False]
    assert g["close"].tolist() == [2.0, 2.0, 2.0, 4.0]
    assert g["volume"].tolist() == [2.0, 0.0, 0.0, 2.0]
