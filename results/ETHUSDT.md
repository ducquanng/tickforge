# ETHUSDT walk-forward results

Test period 2026-08-23 to 2026-09-29: 38 one-day folds, 54,720 out-of-sample one-minute bars. Each fold trains on the preceding 21 days, purged by the label horizon.

## 5-minute return forecasts

| model | IC | hit_rate | R2_oos_vs_zero_bps | t_vs_zero | days_beating_zero |
|---|---:|---:|---:|---:|---:|
| zero (random walk) | 0 | – | 0 | – | – |
| momentum (ret_5) | 0.01003 | 0.5027 | -14.88 | -0.8195 | 0.3947 |
| order flow (ofi_5) | 0.00851 | 0.4963 | -5.192 | -1.494 | 0.4737 |
| ridge | 0.00426 | 0.5014 | -137.9 | -4.265 | 0.2368 |
| lightgbm | 0.009951 | 0.5058 | -445 | -4.288 | 0.1053 |

## 30-minute realised volatility forecasts

| model | MSE_log_rv | QLIKE | R2_oos_vs_HAR | t_vs_HAR | days_beating_HAR |
|---|---:|---:|---:|---:|---:|
| persistence (rv_30) | 0.1668 | 0.5071 | -0.2594 | -13.06 | 0 |
| ewma | 0.1431 | 0.3976 | -0.08093 | -3.737 | 0.1316 |
| HAR | 0.1324 | 0.425 | 0 | – | – |
| lightgbm | 0.1234 | 0.353 | 0.06839 | 2.933 | 0.8684 |

## Tick-level backtest of the LightGBM return forecast

| strategy | net_pnl_usd | pnl_bps_of_turnover | fees_usd | turnover_usd | fills | cancelled | sharpe_annual | max_drawdown_usd | deflated_sharpe | t_stat |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| taker, 0 bps threshold | -80,845 | -5.006 | 80,751 | 161,502,448 | 54,048 | 0 | -82.76 | 78,704 | 4.936e-21 | -26.7 |
| maker, 0 bps threshold | -40,919 | -3.086 | 26,523 | 132,615,922 | 43,005 | 11,444 | -68.59 | 39,833 | 2.215e-24 | -22.13 |
| taker, 2 bps threshold | -37,561 | -4.969 | 37,796 | 75,592,510 | 18,547 | 0 | -44.13 | 35,882 | 3.727e-43 | -14.24 |
| maker, 2 bps threshold | -19,572 | -3.024 | 12,943 | 64,716,088 | 15,851 | 2,764 | -39.3 | 18,727 | 8.223e-55 | -12.68 |
| taker, 4 bps threshold | -19,401 | -4.865 | 19,939 | 39,877,450 | 8,634 | 0 | -29.79 | 18,461 | 5.328e-106 | -9.611 |
| maker, 4 bps threshold | -10,347 | -2.96 | 6,991 | 34,954,429 | 7,581 | 1,107 | -25.86 | 9,918 | 5.87e-103 | -8.344 |
| taker, 8 bps threshold | -6,458 | -5.219 | 6,187 | 12,374,057 | 2,426 | 0 | -17.03 | 6,194 | 1.202e-162 | -5.495 |
| maker, 8 bps threshold | -3,478 | -3.088 | 2,253 | 11,263,120 | 2,230 | 192 | -11.9 | 3,503 | 2.079e-161 | -3.839 |
| taker, 0 bps threshold, zero fees | -94.16 | -0.00583 | 0 | 161,502,448 | 54,048 | 0 | -0.2019 | 918.3 | – | -0.06516 |
