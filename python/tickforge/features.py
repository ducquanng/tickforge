"""Point-in-time features.

Every feature in row ``t`` is a function of information available at
``close_ts[t]`` only: backward-looking rolling windows over bars, and
strictly-earlier as-of joins for anything arriving on another clock.
``tests/test_leakage.py`` enforces this by checking that truncating the
future never changes a feature value in the past.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

RET_WINDOWS = (1, 5, 15, 60)
VOL_WINDOWS = (5, 30, 240, 1440)
FLOW_WINDOWS = (1, 5, 15, 60)
EPS = 1e-12


def bar_features(bars: pd.DataFrame, prefix: str = "") -> pd.DataFrame:
    """Price, volatility, order-flow and liquidity features from one symbol's bars."""
    close = bars["close"]
    logp = np.log(close)
    r1 = logp.diff()
    f: dict[str, pd.Series] = {}
    for w in RET_WINDOWS:
        f[f"ret_{w}"] = logp.diff(w)
    r2 = r1**2
    for w in VOL_WINDOWS:
        f[f"rv_{w}"] = np.sqrt(r2.rolling(w).mean())
    f["ewma_vol"] = np.sqrt(r2.ewm(halflife=30, min_periods=60).mean())
    # Parkinson range estimator: uses intrabar high/low, less noisy than close-to-close.
    hl2 = np.log(bars["high"] / bars["low"]) ** 2 / (4 * np.log(2))
    f["parkinson_30"] = np.sqrt(hl2.rolling(30).mean())
    signed = bars["buy_volume"] - bars["sell_volume"]
    for w in FLOW_WINDOWS:
        f[f"ofi_{w}"] = signed.rolling(w).sum() / (bars["volume"].rolling(w).sum() + EPS)
    logv = np.log1p(bars["dollar"])
    f["volume_z"] = (logv - logv.rolling(1440).mean()) / (logv.rolling(1440).std() + EPS)
    logn = np.log1p(bars["n_trades"])
    f["trades_z"] = (logn - logn.rolling(1440).mean()) / (logn.rolling(1440).std() + EPS)
    f["trade_size"] = np.log((bars["dollar"].rolling(15).sum() + 1.0) / (bars["n_trades"].rolling(15).sum() + 1.0))
    for w in (15, 60):
        vwap = bars["dollar"].rolling(w).sum() / (bars["volume"].rolling(w).sum() + EPS)
        f[f"vwap_dev_{w}"] = close / vwap - 1.0
    # Amihud illiquidity: absolute return per dollar traded.
    f["amihud_60"] = np.log((r1.abs() / (bars["dollar"] + 1.0)).rolling(60).mean() + EPS)
    out = pd.DataFrame(f)
    out.index = pd.Index(bars["close_ts"].to_numpy(), name="close_ts")
    return out.add_prefix(prefix)


def clock_features(close_ts: pd.Index) -> pd.DataFrame:
    """Time-of-day encoding; intraday volatility is strongly seasonal."""
    minute = (np.asarray(close_ts, dtype=np.int64) // 60_000) % 1440
    ang = 2 * np.pi * minute / 1440
    return pd.DataFrame({"tod_sin": np.sin(ang), "tod_cos": np.cos(ang)}, index=close_ts)


def depth_features(depth: pd.DataFrame, close_ts: pd.Index, prefix: str = "") -> pd.DataFrame:
    """Order-book imbalance and liquidity from bookDepth snapshots.

    Snapshots are joined to bars with a strictly-backward as-of join, so a bar
    closing at ``t`` only ever sees a snapshot stamped before ``t``.
    """
    wide = depth.pivot_table(index="ts", columns="pct", values="notional", aggfunc="last").sort_index()
    snap: dict[str, pd.Series] = {}
    for k, name in ((0.2, "02"), (1.0, "1"), (5.0, "5")):
        if -k in wide.columns and k in wide.columns:
            bid, ask = wide[-k], wide[k]
            snap[f"depth_imb_{name}"] = (bid - ask) / (bid + ask + EPS)
    if -1.0 in wide.columns and 1.0 in wide.columns:
        snap["depth_log_1"] = np.log(wide[-1.0] + wide[1.0] + 1.0)
    s = pd.DataFrame(snap).reset_index()
    s["ts"] = s["ts"].astype(np.int64)
    left = pd.DataFrame({"close_ts": np.asarray(close_ts, dtype=np.int64)})
    j = pd.merge_asof(left, s, left_on="close_ts", right_on="ts", direction="backward", allow_exact_matches=False)
    # A snapshot older than five minutes is treated as missing rather than stale-but-trusted.
    stale = (j["close_ts"] - j["ts"]) > 300_000
    j.loc[stale, list(snap)] = np.nan
    out = j.drop(columns=["ts"]).set_index("close_ts")
    return out.add_prefix(prefix)


def build_features(
    bars: pd.DataFrame,
    depth: pd.DataFrame | None = None,
    others: dict[str, pd.DataFrame] | None = None,
) -> pd.DataFrame:
    """Full feature matrix for one symbol, indexed by ``close_ts``.

    ``others`` maps a label to another symbol's bars; a compact set of that
    symbol's features is aligned on the same clock (cross-asset lead-lag).
    """
    parts = [bar_features(bars)]
    parts.append(clock_features(parts[0].index))
    if depth is not None:
        parts.append(depth_features(depth, parts[0].index))
    for label, ob in (others or {}).items():
        cross = bar_features(ob, prefix=f"{label}_")
        keep = [f"{label}_{c}" for c in ("ret_1", "ret_5", "ret_15", "ofi_1", "ofi_5", "rv_30")]
        parts.append(cross[keep].reindex(parts[0].index))
    return pd.concat(parts, axis=1)
