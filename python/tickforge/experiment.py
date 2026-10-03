"""End-to-end walk-forward experiment: ticks -> bars -> features -> models ->
out-of-sample forecasts -> tick-level backtest.

Every number reported comes from test days the models never saw, with
training samples purged by the label horizon.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import stats

from tickforge import backtest, metrics
from tickforge.bars import make_bars, regular_grid
from tickforge.data.binance import date_range
from tickforge.data.store import TickStore
from tickforge.features import build_features
from tickforge.labels import forward_log_rv, forward_return
from tickforge.models import GBM, HAR, Column, Model, RidgeModel, Zero
from tickforge.validation import WalkForward, assert_no_overlap

DAY_MS = 86_400_000


@dataclass
class ExperimentConfig:
    symbol: str = "BTCUSDT"
    cross_symbol: str | None = "ETHUSDT"
    start: dt.date = dt.date(2026, 8, 1)
    end: dt.date = dt.date(2026, 9, 30)
    market: str = "um"
    bar_ms: int = 60_000
    ret_horizon: int = 5
    vol_horizon: int = 30
    train_days: int = 21
    notional: float = 10_000.0
    thresholds_bps: tuple[float, ...] = (0.0, 2.0, 4.0, 8.0)
    tick_size: float = 0.1
    seed: int = 0

    @property
    def bars_per_day(self) -> int:
        return DAY_MS // self.bar_ms


def bars_from_store(store: TickStore, symbol: str, cfg: ExperimentConfig) -> pd.DataFrame:
    """Build bars one day at a time, carrying the still-forming bar's trades
    into the next day so the result equals a single pass over all ticks."""
    out, carry = [], None
    for day in date_range(cfg.start, cfg.end):
        trades = store.read("aggTrades", symbol, day, day, cfg.market)
        if carry is not None and len(carry):
            trades = pd.concat([carry, trades], ignore_index=True)
        b = make_bars(trades, "time", cfg.bar_ms)
        cut = b["close_ts"].iloc[-1] if len(b) else trades["ts"].iloc[0]
        carry = trades[trades["ts"] >= cut]
        out.append(b)
    return regular_grid(pd.concat(out, ignore_index=True), cfg.bar_ms)


def build_dataset(store: TickStore, cfg: ExperimentConfig) -> pd.DataFrame:
    """Feature matrix plus targets, indexed by ``close_ts``."""
    bars = bars_from_store(store, cfg.symbol, cfg)
    depth = None
    try:
        depth = store.read("bookDepth", cfg.symbol, cfg.start, cfg.end, cfg.market)
    except FileNotFoundError:
        pass
    others = None
    if cfg.cross_symbol:
        ob = bars_from_store(store, cfg.cross_symbol, cfg)
        ob = ob.set_index("close_ts").reindex(bars["close_ts"]).ffill().reset_index()
        others = {"x": ob}
    X = build_features(bars, depth, others)
    close = pd.Series(bars["close"].to_numpy(), index=X.index)
    X["y_ret"] = forward_return(close, cfg.ret_horizon)
    X["y_vol"] = forward_log_rv(close, cfg.vol_horizon)
    X["close"] = close
    # Drop the warm-up period of the longest rolling window and rows with no label yet.
    return X.iloc[1440:].dropna(subset=["y_ret", "y_vol", "rv_1440"])


def return_models(seed: int) -> dict[str, Model]:
    return {
        "zero (random walk)": Zero(),
        "momentum (ret_5)": Column("ret_5", fit_scale=True),
        "order flow (ofi_5)": Column("ofi_5", fit_scale=True),
        "ridge": RidgeModel(),
        "lightgbm": GBM(seed=seed),
    }


def vol_models(seed: int) -> dict[str, Model]:
    return {
        "persistence (rv_30)": Column("rv_30", log=True),
        "ewma": Column("ewma_vol", log=True),
        "HAR": HAR(),
        "lightgbm": GBM(seed=seed),
    }


@dataclass
class ExperimentResult:
    config: ExperimentConfig
    preds: pd.DataFrame  # OOS forecasts, columns "<target>|<model>", plus y_ret, y_vol, close
    return_table: pd.DataFrame
    vol_table: pd.DataFrame
    backtests: pd.DataFrame
    daily_pnl: dict[str, pd.Series] = field(default_factory=dict)
    importance: pd.Series | None = None
    n_folds: int = 0


def walk_forward_predict(ds: pd.DataFrame, cfg: ExperimentConfig) -> tuple[pd.DataFrame, pd.Series, int]:
    feats = [c for c in ds.columns if c not in ("y_ret", "y_vol", "close")]
    X = ds[feats]
    purge = max(cfg.ret_horizon, cfg.vol_horizon)
    # Align folds to UTC midnight so each test block is one calendar day.
    offset = int(np.argmax((ds.index.to_numpy() - cfg.bar_ms) % DAY_MS == 0))
    wf = WalkForward(train_size=cfg.train_days * cfg.bars_per_day - purge, test_size=cfg.bars_per_day, horizon=purge)
    chunks, imp, n_folds = [], [], 0
    for tr, te in wf.split(len(ds) - offset):
        tr, te = tr + offset, te + offset
        assert_no_overlap(tr, te, purge)
        fold = ds.iloc[te][["y_ret", "y_vol", "close"]].copy()
        for target, models in (("y_ret", return_models(cfg.seed)), ("y_vol", vol_models(cfg.seed))):
            for name, model in models.items():
                model.fit(X.iloc[tr], ds[target].iloc[tr])
                fold[f"{target}|{name}"] = model.predict(X.iloc[te])
                if target == "y_ret" and isinstance(model, GBM):
                    imp.append(model.importance())
        chunks.append(fold)
        n_folds += 1
    importance = pd.concat(imp, axis=1).mean(axis=1).sort_values(ascending=False)
    return pd.concat(chunks), importance / importance.sum(), n_folds


def _day(index: pd.Index) -> np.ndarray:
    # A bar closing exactly at midnight belongs to the day that just ended.
    return (np.asarray(index, dtype=np.int64) - 1) // DAY_MS


def _paired_t(loss_a: pd.Series, loss_b: pd.Series, day: np.ndarray) -> tuple[float, float]:
    """t-statistic of the daily mean loss difference (a - b) and the share of days a < b."""
    d = (loss_a - loss_b).groupby(day).mean()
    if len(d) < 2 or d.std(ddof=1) == 0:
        return float("nan"), float("nan")
    return float(d.mean() / (d.std(ddof=1) / np.sqrt(len(d)))), float((d < 0).mean())


def score(preds: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    day = _day(preds.index)
    y = preds["y_ret"].to_numpy()
    rows = []
    for col in [c for c in preds.columns if c.startswith("y_ret|")]:
        p = preds[col].to_numpy()
        t, win = _paired_t((preds["y_ret"] - preds[col]) ** 2, preds["y_ret"] ** 2, day)
        rows.append(
            {
                "model": col.split("|")[1],
                "IC": metrics.information_coefficient(y, p),
                "hit_rate": metrics.hit_rate(y, p),
                "R2_oos_vs_zero_bps": 1e4 * metrics.r2_oos(y, p, 0.0),
                "t_vs_zero": -t,
                "days_beating_zero": win,
            }
        )
    ret_table = pd.DataFrame(rows).set_index("model")
    yv = preds["y_vol"].to_numpy()
    har = preds["y_vol|HAR"]
    rows = []
    for col in [c for c in preds.columns if c.startswith("y_vol|")]:
        p = preds[col].to_numpy()
        t, win = _paired_t((preds["y_vol"] - preds[col]) ** 2, (preds["y_vol"] - har) ** 2, day)
        rows.append(
            {
                "model": col.split("|")[1],
                "MSE_log_rv": float(np.mean((yv - p) ** 2)),
                "QLIKE": metrics.qlike(yv, p),
                "R2_oos_vs_HAR": metrics.r2_oos(yv, p, har.to_numpy()),
                "t_vs_HAR": -t if col != "y_vol|HAR" else float("nan"),
                "days_beating_HAR": win if col != "y_vol|HAR" else float("nan"),
            }
        )
    return ret_table, pd.DataFrame(rows).set_index("model")


def run_backtests(store: TickStore, preds: pd.DataFrame, cfg: ExperimentConfig, model: str = "lightgbm"):
    """Trade the return forecast on the tick tape, one UTC day at a time,
    flat at each day's end. Returns a summary per configuration and daily PnL."""
    pred = preds[f"y_ret|{model}"]
    day = _day(preds.index)
    configs = []
    for th in cfg.thresholds_bps:
        configs.append((f"taker, {th:g} bps threshold", th, backtest.Costs(tick_size=cfg.tick_size)))
        configs.append((f"maker, {th:g} bps threshold", th, backtest.Costs(tick_size=cfg.tick_size, maker=True)))
    configs.append(("taker, 0 bps threshold, zero fees", 0.0, backtest.Costs(tick_size=cfg.tick_size, taker_fee_bps=0.0)))
    daily: dict[str, dict[int, float]] = {name: {} for name, _, _ in configs}
    agg = {name: {"fees": 0.0, "turnover": 0.0, "fills": 0, "cancelled": 0} for name, _, _ in configs}
    for d in np.unique(day):
        m = day == d
        date = dt.datetime.fromtimestamp(int(d) * 86_400, dt.timezone.utc).date()
        trades = store.read("aggTrades", cfg.symbol, date, date, cfg.market)
        ts = preds.index[m].to_numpy(dtype=np.int64)
        px = preds["close"].to_numpy()[m]
        for name, th, costs in configs:
            tgt = backtest.threshold_targets(pred.to_numpy()[m], px, th * 1e-4, cfg.notional)
            tgt[-1] = 0.0  # flatten into the end of the day
            r = backtest.run(trades, ts, tgt, costs)
            daily[name][int(d)] = r.final_equity
            a = agg[name]
            a["fees"] += r.total_fees
            a["turnover"] += r.turnover
            a["fills"] += len(r.fills)
            a["cancelled"] += r.n_cancelled
    pnl = {k: pd.Series(v).sort_index() for k, v in daily.items()}
    trials = [n for n, _, _ in configs if "zero fees" not in n]
    trial_sr = np.array([metrics.sharpe(pnl[n].to_numpy()) for n in trials])
    rows = []
    for name, _, _ in configs:
        p = pnl[name].to_numpy()
        a = agg[name]
        rows.append(
            {
                "strategy": name,
                "net_pnl_usd": p.sum(),
                "pnl_bps_of_turnover": 1e4 * p.sum() / a["turnover"] if a["turnover"] else float("nan"),
                "fees_usd": a["fees"],
                "turnover_usd": a["turnover"],
                "fills": a["fills"],
                "cancelled": a["cancelled"],
                "sharpe_annual": metrics.sharpe(p, 365),
                "max_drawdown_usd": metrics.max_drawdown(np.cumsum(p)),
                "deflated_sharpe": metrics.deflated_sharpe(p, trial_sr) if name in trials and len(p) > 2 and p.std() > 0 else float("nan"),
                "t_stat": float(stats.ttest_1samp(p, 0.0).statistic) if len(p) > 2 and p.std() > 0 else float("nan"),
            }
        )
    return pd.DataFrame(rows).set_index("strategy"), pnl


def run_experiment(store: TickStore, cfg: ExperimentConfig) -> ExperimentResult:
    ds = build_dataset(store, cfg)
    preds, importance, n_folds = walk_forward_predict(ds, cfg)
    ret_table, vol_table = score(preds)
    bt, pnl = run_backtests(store, preds, cfg)
    return ExperimentResult(cfg, preds, ret_table, vol_table, bt, pnl, importance, n_folds)
