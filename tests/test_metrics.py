import numpy as np

from tickforge import metrics


def test_r2_oos_and_qlike_are_zero_at_the_benchmark():
    y = np.random.default_rng(0).normal(size=500)
    assert metrics.r2_oos(y, np.zeros(500)) == 0.0
    assert metrics.r2_oos(y, y) == 1.0
    assert metrics.qlike(y, y) == 0.0
    assert metrics.qlike(y, y + 0.3) > 0


def test_max_drawdown():
    assert metrics.max_drawdown(np.array([0, 5, 2, 8, 1, 3])) == 7


def test_probabilistic_sharpe_is_calibrated_under_the_null():
    rng = np.random.default_rng(0)
    p = [metrics.probabilistic_sharpe(rng.normal(0, 1, 250)) for _ in range(400)]
    assert abs(np.mean(p) - 0.5) < 0.05
    assert metrics.probabilistic_sharpe(rng.normal(0.3, 1, 250)) > 0.99


def test_deflated_sharpe_penalises_more_trials():
    rng = np.random.default_rng(1)
    r = rng.normal(0.1, 1, 500)
    few = metrics.deflated_sharpe(r, rng.normal(0, 0.05, 5))
    many = metrics.deflated_sharpe(r, rng.normal(0, 0.05, 500))
    assert many < few < metrics.probabilistic_sharpe(r)


def test_best_of_many_random_strategies_is_not_significant():
    rng = np.random.default_rng(2)
    trials = rng.normal(0, 1, (200, 250))
    sr = np.array([metrics.sharpe(t) for t in trials])
    best = trials[sr.argmax()]
    assert metrics.probabilistic_sharpe(best) > 0.95  # naive test is fooled
    assert metrics.deflated_sharpe(best, sr) < 0.95  # deflated one is not
