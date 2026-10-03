# BTCUSDT walk-forward results

Test period 2026-08-23 to 2026-09-29: 38 one-day folds, 54,720 out-of-sample one-minute bars. Each fold trains on the preceding 21 days, purged by the label horizon.

## 5-minute return forecasts

| model | IC | hit_rate | R2_oos_vs_zero_bps | t_vs_zero | days_beating_zero |
|---|---:|---:|---:|---:|---:|
| zero (random walk) | 0 | – | 0 | – | – |
| momentum (ret_5) | 0.02526 | 0.5045 | -9.576 | -0.7078 | 0.5526 |
| order flow (ofi_5) | 0.004789 | 0.5014 | -7.175 | -1.367 | 0.4474 |
| ridge | 0.01709 | 0.5068 | -94.24 | -2.122 | 0.2632 |
| lightgbm | 0.01238 | 0.5023 | -448.8 | -4.804 | 0.1053 |

## 30-minute realised volatility forecasts

| model | MSE_log_rv | QLIKE | R2_oos_vs_HAR | t_vs_HAR | days_beating_HAR |
|---|---:|---:|---:|---:|---:|
| persistence (rv_30) | 0.1478 | 0.4046 | -0.2251 | -13.33 | 0.05263 |
| ewma | 0.1255 | 0.3164 | -0.04003 | -2.329 | 0.3158 |
| HAR | 0.1206 | 0.343 | 0 | – | – |
| lightgbm | 0.1107 | 0.2874 | 0.08266 | 4.299 | 0.7895 |

## Tick-level backtest of the LightGBM return forecast

| strategy | net_pnl_usd | pnl_bps_of_turnover | fees_usd | turnover_usd | fills | cancelled | sharpe_annual | max_drawdown_usd | deflated_sharpe | t_stat |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| taker, 0 bps threshold | -83,098 | -4.997 | 83,156 | 166,312,348 | 53,398 | 0 | -75.32 | 80,912 | 1.589e-19 | -24.3 |
| maker, 0 bps threshold | -37,957 | -3.175 | 23,910 | 119,550,823 | 39,275 | 15,201 | -59.88 | 36,988 | 5.375e-37 | -19.32 |
| taker, 2 bps threshold | -25,800 | -4.938 | 26,123 | 52,245,222 | 13,175 | 0 | -32.26 | 25,125 | 1.965e-71 | -10.41 |
| maker, 2 bps threshold | -13,169 | -3.066 | 8,592 | 42,959,985 | 10,963 | 2,278 | -27.67 | 12,896 | 8.891e-99 | -8.927 |
| taker, 4 bps threshold | -12,592 | -5.028 | 12,521 | 25,042,118 | 5,740 | 0 | -23.58 | 12,306 | 3.374e-94 | -7.608 |
| maker, 4 bps threshold | -6,685 | -3.14 | 4,258 | 21,289,613 | 4,948 | 797 | -22.56 | 6,581 | 2.526e-94 | -7.28 |
| taker, 8 bps threshold | -3,499 | -5.066 | 3,454 | 6,907,447 | 1,459 | 0 | -18.68 | 3,389 | 1.096e-91 | -6.026 |
| maker, 8 bps threshold | -1,775 | -2.818 | 1,259 | 6,296,968 | 1,337 | 116 | -13.72 | 1,769 | 2.205e-97 | -4.426 |
| taker, 0 bps threshold, zero fees | 58.06 | 0.003491 | 0 | 166,312,348 | 53,398 | 0 | 0.1656 | 524 | – | 0.05344 |
