"""End-to-end pipeline on synthetic ticks, including a planted-leak check."""

import numpy as np

from tickforge.bars import make_bars
from tickforge.experiment import ExperimentConfig, bars_from_store, build_dataset, run_experiment
from tickforge.report import write_report


def _cfg(start, end):
    return ExperimentConfig(symbol="AAAUSDT", cross_symbol="BBBUSDT", start=start, end=end, train_days=2, tick_size=0.01, thresholds_bps=(0.0, 1.0))


def test_daily_bar_building_equals_single_pass(synthetic_store):
    store, start, end = synthetic_store
    cfg = _cfg(start, end)
    one_pass = make_bars(store.read("aggTrades", "AAAUSDT", start, end), "time", 60_000)
    daily = bars_from_store(store, "AAAUSDT", cfg)
    daily = daily[~daily["empty"]]
    np.testing.assert_array_equal(daily["close_ts"], one_pass["close_ts"])
    np.testing.assert_allclose(daily["volume"], one_pass["volume"])
    np.testing.assert_array_equal(daily["close"], one_pass["close"])


def test_pipeline_runs_end_to_end(synthetic_store, tmp_path):
    store, start, end = synthetic_store
    res = run_experiment(store, _cfg(start, end))
    assert res.n_folds >= 1
    assert len(res.preds) == res.n_folds * 1440
    assert not res.preds.isna().any().any()
    assert {"lightgbm", "HAR"} <= set(res.vol_table.index)
    assert len(res.backtests) == 5
    # Fees can only hurt: the zero-fee run bounds the fee-paying one from above.
    bt = res.backtests["net_pnl_usd"]
    assert bt["taker, 0 bps threshold, zero fees"] >= bt["taker, 0 bps threshold"]
    path = write_report(res, tmp_path, figures=False)
    assert "walk-forward results" in path.read_text()


def test_a_leaky_feature_would_be_obvious(synthetic_store):
    """Sanity check on the evaluation itself: handing the model the label makes
    the IC jump to ~1, so a near-zero IC on honest features is not a bug."""
    from tickforge.experiment import score, walk_forward_predict

    store, start, end = synthetic_store
    cfg = _cfg(start, end)
    ds = build_dataset(store, cfg)
    honest, _, _ = walk_forward_predict(ds, cfg)
    leaky, _, _ = walk_forward_predict(ds.assign(leak=ds["y_ret"]), cfg)
    ic_honest = score(honest)[0].loc["lightgbm", "IC"]
    ic_leaky = score(leaky)[0].loc["lightgbm", "IC"]
    assert ic_leaky > 0.9 > abs(ic_honest) + 0.5
