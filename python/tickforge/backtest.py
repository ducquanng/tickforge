"""Strategy simulation on the tick tape via the Rust engine."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from tickforge import _core


@dataclass(frozen=True)
class Costs:
    """Execution assumptions. Defaults are Binance USD-M futures base-tier fees
    (0.05% taker, 0.02% maker) and 50 ms of latency."""

    latency: int = 50
    taker_fee_bps: float = 5.0
    maker_fee_bps: float = 2.0
    slippage_bps: float = 0.0
    tick_size: float = 0.1
    maker: bool = False


@dataclass
class BacktestResult:
    equity: pd.Series  # indexed by decision timestamp (ms), in quote currency
    position: pd.Series
    fills: pd.DataFrame
    final_equity: float
    total_fees: float
    n_cancelled: int

    @property
    def pnl(self) -> np.ndarray:
        """Per-decision-interval PnL in quote currency."""
        return np.diff(np.append(self.equity.to_numpy(), self.final_equity))

    @property
    def turnover(self) -> float:
        """Traded notional in quote currency."""
        return float((self.fills["price"] * self.fills["qty"].abs()).sum())


def run(trades: pd.DataFrame, order_ts: np.ndarray, target: np.ndarray, costs: Costs | None = None) -> BacktestResult:
    """Execute ``target`` (signed position in base units at each ``order_ts``)
    against the trade tape."""
    costs = costs or Costs()
    order_ts = np.ascontiguousarray(order_ts, dtype=np.int64)
    out = _core.run_backtest(
        np.ascontiguousarray(trades["ts"].to_numpy(dtype=np.int64)),
        np.ascontiguousarray(trades["price"].to_numpy(dtype=np.float64)),
        np.ascontiguousarray(trades["buyer_maker"].to_numpy(dtype=bool)),
        order_ts,
        np.ascontiguousarray(target, dtype=np.float64),
        **asdict(costs),
    )
    fills = pd.DataFrame({"ts": out["fill_ts"], "price": out["fill_price"], "qty": out["fill_qty"], "fee": out["fill_fee"]})
    return BacktestResult(
        equity=pd.Series(out["equity"], index=order_ts),
        position=pd.Series(out["position"], index=order_ts),
        fills=fills,
        final_equity=out["final_equity"],
        total_fees=out["total_fees"],
        n_cancelled=out["n_cancelled"],
    )


def threshold_targets(pred: np.ndarray, price: np.ndarray, threshold: float, notional: float) -> np.ndarray:
    """Long/short/flat rule: hold ``notional`` in the direction of the forecast
    when its magnitude exceeds ``threshold`` (log-return units), else go flat."""
    side = np.where(pred > threshold, 1.0, np.where(pred < -threshold, -1.0, 0.0))
    return side * notional / price
