"""Write an experiment's results to disk: JSON, Markdown tables and figures."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

from tickforge.experiment import DAY_MS, ExperimentResult

SURFACE, INK, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]


def _md(df: pd.DataFrame, digits: int = 4) -> str:
    def fmt(v):
        if isinstance(v, (int, np.integer)):
            return f"{v:,}"
        if isinstance(v, (float, np.floating)):
            if np.isnan(v):
                return "–"
            return f"{v:,.0f}" if abs(v) >= 1000 else f"{v:.{digits}g}"
        return str(v)

    head = "| " + " | ".join([df.index.name or ""] + list(df.columns)) + " |"
    sep = "|" + "|".join(["---"] + ["---:"] * len(df.columns)) + "|"
    rows = ["| " + " | ".join([str(i)] + [fmt(v) for v in r]) + " |" for i, r in zip(df.index, df.to_numpy(dtype=object), strict=True)]
    return "\n".join([head, sep, *rows])


def _axes(ax, title: str, ylabel: str):
    ax.set_facecolor(SURFACE)
    ax.set_title(title, loc="left", fontsize=11, color=INK, pad=10)
    ax.set_ylabel(ylabel, fontsize=9, color=MUTED)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.tick_params(colors=MUTED, labelsize=8, length=0)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)


def _lines(ax, series: dict[str, pd.Series]):
    for (name, s), color in zip(series.items(), SERIES, strict=False):
        x = pd.to_datetime(np.asarray(s.index, dtype=np.int64) * DAY_MS, unit="ms")
        ax.plot(x, s.to_numpy(), color=color, linewidth=2, label=name)
    ax.axhline(0, color=MUTED, linewidth=0.8)
    ax.legend(frameon=False, fontsize=8, labelcolor=INK, loc="upper left", bbox_to_anchor=(1.0, 1.0))
    ax.tick_params(axis="x", rotation=30)


def write_figures(res: ExperimentResult, out: Path) -> list[Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sym = res.config.symbol
    p = res.preds
    day = (np.asarray(p.index, dtype=np.int64) - 1) // DAY_MS
    har = (p["y_vol"] - p["y_vol|HAR"]) ** 2
    gain = {}
    for name in ("lightgbm", "ewma", "persistence (rv_30)"):
        loss = (p["y_vol"] - p[f"y_vol|{name}"]) ** 2
        gain[name] = (har - loss).groupby(day).mean().cumsum()
    fig, ax = plt.subplots(figsize=(9.5, 3.6), facecolor=SURFACE)
    _axes(ax, f"{sym}: volatility forecast accuracy vs HAR, out of sample", "cumulative daily MSE gain over HAR\n(above 0 = better than HAR)")
    _lines(ax, gain)
    fig.tight_layout()
    f1 = out / f"{sym}_vol_vs_har.png"
    fig.savefig(f1, dpi=160)
    plt.close(fig)

    keep = ["taker, 0 bps threshold, zero fees", "maker, 8 bps threshold", "taker, 8 bps threshold", "taker, 0 bps threshold"]
    fig, ax = plt.subplots(figsize=(9.5, 3.6), facecolor=SURFACE)
    _axes(ax, f"{sym}: return-forecast strategy, tick-level backtest, out of sample", "cumulative net PnL (USD, $10k position)")
    _lines(ax, {k: res.daily_pnl[k].cumsum() for k in keep})
    # Symmetric log scale: the fee-paying curves are orders of magnitude below the zero-fee one.
    ax.set_yscale("symlog", linthresh=1000)
    ticks = [-100_000, -10_000, -1000, 0, 1000]
    ax.set_yticks(ticks, [f"{t:,}" for t in ticks])
    ax.minorticks_off()
    fig.tight_layout()
    f2 = out / f"{sym}_backtest.png"
    fig.savefig(f2, dpi=160)
    plt.close(fig)
    return [f1, f2]


def write_report(res: ExperimentResult, out: str | Path, figures: bool = True) -> Path:
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    sym = res.config.symbol
    cfg = {k: (v.isoformat() if hasattr(v, "isoformat") else v) for k, v in asdict(res.config).items()}
    days = np.unique((np.asarray(res.preds.index, dtype=np.int64) - 1) // DAY_MS)
    first, last = (pd.to_datetime(int(d) * DAY_MS, unit="ms").date().isoformat() for d in (days[0], days[-1]))
    payload = {
        "config": cfg,
        "n_folds": res.n_folds,
        "test_period": [first, last],
        "n_test_samples": len(res.preds),
        "returns": res.return_table.reset_index().to_dict("records"),
        "volatility": res.vol_table.reset_index().to_dict("records"),
        "backtests": res.backtests.reset_index().to_dict("records"),
        "top_features_return_model": res.importance.head(15).round(4).to_dict(),
    }
    (out / f"{sym}.json").write_text(json.dumps(payload, indent=2, default=float))
    md = [
        f"# {sym} walk-forward results",
        "",
        f"Test period {first} to {last}: {res.n_folds} one-day folds, {len(res.preds):,} out-of-sample one-minute bars. "
        f"Each fold trains on the preceding {res.config.train_days} days, purged by the label horizon.",
        "",
        f"## {res.config.ret_horizon}-minute return forecasts",
        "",
        _md(res.return_table),
        "",
        f"## {res.config.vol_horizon}-minute realised volatility forecasts",
        "",
        _md(res.vol_table),
        "",
        "## Tick-level backtest of the LightGBM return forecast",
        "",
        _md(res.backtests),
        "",
    ]
    (out / f"{sym}.md").write_text("\n".join(md))
    if figures:
        write_figures(res, out)
    return out / f"{sym}.md"
