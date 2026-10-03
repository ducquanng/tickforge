"""Offline tour of the building blocks on synthetic ticks (no download needed).

python examples/quickstart.py
"""

from tickforge import backtest
from tickforge.bars import make_bars
from tickforge.data.synthetic import synthetic_trades
from tickforge.features import build_features
from tickforge.labels import forward_return
from tickforge.validation import PurgedKFold

trades = synthetic_trades(n=400_000, seed=0)
bars = make_bars(trades, "dollar", 2_500)
X = build_features(bars)
y = forward_return(bars["close"], 5)
print(f"{len(trades):,} trades -> {len(bars):,} dollar bars, {X.shape[1]} features")

for k, (train, test) in enumerate(PurgedKFold(n_splits=4, horizon=5, embargo=10).split(len(X))):
    print(f"fold {k}: train {len(train):,}  test {len(test):,}  (purged {len(X) - len(train) - len(test)})")

# Follow the last bar's order-flow imbalance, crossing the spread every bar.
signal = X["ofi_5"].fillna(0.0).to_numpy()
target = backtest.threshold_targets(signal, bars["close"].to_numpy(), 0.2, 1_000.0)
target[-1] = 0.0
for label, costs in [("no fees", backtest.Costs(taker_fee_bps=0, tick_size=0.01)), ("5 bps taker fee", backtest.Costs(tick_size=0.01))]:
    r = backtest.run(trades, bars["close_ts"].to_numpy(), target, costs)
    print(f"{label:>15}: PnL {r.final_equity:9.2f}  fills {len(r.fills):,}  turnover {r.turnover:,.0f}")
