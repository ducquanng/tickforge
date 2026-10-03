"""Forecast and strategy evaluation, including multiple-testing corrections."""

from __future__ import annotations

import numpy as np
from scipy import stats

EULER_GAMMA = 0.5772156649015329


def r2_oos(y: np.ndarray, pred: np.ndarray, benchmark: np.ndarray | float = 0.0) -> float:
    """Out-of-sample R^2 against a benchmark forecast (Campbell & Thompson, 2008).
    Positive means the model beats the benchmark in squared error."""
    y, pred = np.asarray(y, float), np.asarray(pred, float)
    return 1.0 - np.sum((y - pred) ** 2) / np.sum((y - benchmark) ** 2)


def information_coefficient(y: np.ndarray, pred: np.ndarray) -> float:
    """Spearman rank correlation between forecast and outcome."""
    if np.std(pred) == 0:
        return 0.0
    return float(stats.spearmanr(y, pred).statistic)


def hit_rate(y: np.ndarray, pred: np.ndarray) -> float:
    """Share of non-zero forecasts with the correct sign."""
    m = (pred != 0) & (y != 0)
    return float(np.mean(np.sign(pred[m]) == np.sign(y[m]))) if m.any() else float("nan")


def qlike(log_rv: np.ndarray, log_rv_pred: np.ndarray) -> float:
    """QLIKE loss on variances; robust to noise in the volatility proxy (Patton, 2011)."""
    ratio = np.exp(2 * (np.asarray(log_rv) - np.asarray(log_rv_pred)))
    return float(np.mean(ratio - np.log(ratio) - 1.0))


def sharpe(returns: np.ndarray, periods_per_year: float = 1.0) -> float:
    r = np.asarray(returns, float)
    if r.size < 2:
        return 0.0
    sd = r.std(ddof=1)
    return float(r.mean() / sd * np.sqrt(periods_per_year)) if sd > 0 else 0.0


def max_drawdown(equity: np.ndarray) -> float:
    """Largest peak-to-trough loss, in the units of ``equity``."""
    e = np.asarray(equity, float)
    return float(np.max(np.maximum.accumulate(e) - e)) if e.size else 0.0


def probabilistic_sharpe(returns: np.ndarray, benchmark_sr: float = 0.0) -> float:
    """P(true per-period Sharpe > benchmark), adjusting for sample length,
    skewness and fat tails (Bailey & Lopez de Prado, 2012)."""
    r = np.asarray(returns, float)
    n = r.size
    sr = sharpe(r)
    skew, kurt = stats.skew(r), stats.kurtosis(r, fisher=False)
    denom = np.sqrt(max(1.0 - skew * sr + (kurt - 1.0) / 4.0 * sr**2, 1e-12))
    return float(stats.norm.cdf((sr - benchmark_sr) * np.sqrt(n - 1) / denom))


def expected_max_sharpe(trial_sharpes: np.ndarray) -> float:
    """Expected best per-period Sharpe among N unskilled trials."""
    s = np.asarray(trial_sharpes, float)
    n = s.size
    if n < 2:
        return 0.0
    sd = s.std(ddof=1)
    return float(sd * ((1 - EULER_GAMMA) * stats.norm.ppf(1 - 1 / n) + EULER_GAMMA * stats.norm.ppf(1 - 1 / (n * np.e))))


def deflated_sharpe(returns: np.ndarray, trial_sharpes: np.ndarray) -> float:
    """Deflated Sharpe ratio (Bailey & Lopez de Prado, 2014): the probability
    that the selected strategy has positive skill once you account for having
    picked the best of ``len(trial_sharpes)`` configurations.
    ``trial_sharpes`` are per-period Sharpe ratios of every configuration tried."""
    return probabilistic_sharpe(returns, expected_max_sharpe(trial_sharpes))
