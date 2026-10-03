import numpy as np
import pandas as pd
import pytest

from tickforge import backtest
from tickforge._core import OrderBook
from tickforge.bars import make_bars


def test_pnl_identity_and_fee_accounting(trades):
    bars = make_bars(trades, "time", 60_000)
    rng = np.random.default_rng(0)
    target = rng.choice([-1.0, 0.0, 1.0], len(bars))
    target[-1] = 0.0
    free = backtest.run(trades, bars["close_ts"].to_numpy(), target, backtest.Costs(taker_fee_bps=0.0, tick_size=0.01))
    paid = backtest.run(trades, bars["close_ts"].to_numpy(), target, backtest.Costs(taker_fee_bps=5.0, tick_size=0.01))
    assert paid.total_fees == pytest.approx(paid.turnover * 5e-4)
    assert paid.final_equity == pytest.approx(free.final_equity - paid.total_fees)
    # Ends flat, so equity is exactly the cash from fills.
    cash = -(paid.fills["price"] * paid.fills["qty"]).sum() - paid.fills["fee"].sum()
    assert paid.final_equity == pytest.approx(cash)
    assert paid.pnl.sum() == pytest.approx(paid.final_equity - paid.equity.iloc[0])


def test_random_trading_loses_the_spread(trades):
    bars = make_bars(trades, "time", 10_000)
    rng = np.random.default_rng(3)
    target = rng.choice([-1.0, 1.0], len(bars))
    r = backtest.run(trades, bars["close_ts"].to_numpy(), target, backtest.Costs(taker_fee_bps=0.0, slippage_bps=1.0, tick_size=0.01))
    assert r.final_equity < 0


def test_more_latency_cannot_fill_earlier(trades):
    ts = make_bars(trades, "time", 60_000)["close_ts"].to_numpy()
    tgt = np.resize([1.0, 0.0], len(ts))
    fast = backtest.run(trades, ts, tgt, backtest.Costs(latency=0))
    slow = backtest.run(trades, ts, tgt, backtest.Costs(latency=5_000))
    np.testing.assert_array_equal(slow.fills["ts"].to_numpy()[: len(fast.fills)] - fast.fills["ts"].to_numpy()[: len(slow.fills)], 5_000)


def test_threshold_targets():
    t = backtest.threshold_targets(np.array([3e-4, -3e-4, 1e-4]), np.array([100.0, 200.0, 100.0]), 2e-4, 1000.0)
    np.testing.assert_allclose(t, [10.0, -5.0, 0.0])


def test_engine_rejects_unsorted_orders(trades):
    with pytest.raises(ValueError):
        backtest.run(trades, np.array([trades["ts"].iloc[10], trades["ts"].iloc[5]]), np.array([1.0, 0.0]))


def test_order_book_bindings():
    b = OrderBook(0.5)
    b.update_many("bid", [(100.0, 2.0), (99.5, 1.0)])
    b.update("ask", 100.5, 6.0)
    assert b.best_bid() == (100.0, 2.0)
    assert b.spread() == 0.5
    assert b.imbalance(1) == pytest.approx(-0.5)
    assert b.sweep_vwap("bid", 3.0) == pytest.approx((200.0 + 99.5) / 3)
    assert b.sweep_vwap("ask", 7.0) is None
    with pytest.raises(ValueError):
        b.update("middle", 1.0, 1.0)
    assert isinstance(pd.DataFrame(b.depth("bid", 5)), pd.DataFrame)
