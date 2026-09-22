| Portfolio                    |   Cost (bps) | Total return   | Ann. return   | Ann. vol   |   Sharpe | Max drawdown   | Avg. turnover / rebalance   |
|:-----------------------------|-------------:|:---------------|:--------------|:-----------|---------:|:---------------|:----------------------------|
| LSTM top 30                  |            0 | 22.5%          | 14.5%         | 18.1%      |     0.69 | -13.5%         | 131.4%                      |
| LSTM top 30                  |            5 | 21.0%          | 13.6%         | 18.1%      |     0.64 | -13.7%         | 131.4%                      |
| LSTM top 30                  |           20 | 16.8%          | 10.9%         | 18.1%      |     0.49 | -14.1%         | 131.4%                      |
| Momentum 12-1 top 30         |            0 | 34.2%          | 21.7%         | 20.1%      |     0.98 | -17.5%         | 66.5%                       |
| Momentum 12-1 top 30         |           20 | 31.0%          | 19.7%         | 20.1%      |     0.88 | -17.8%         | 66.5%                       |
| Highest 60-day vol top 30    |            0 | 5.4%           | 3.6%          | 22.7%      |     0.07 | -20.6%         | 60.3%                       |
| Biggest 60-day losers top 30 |            0 | 10.0%          | 6.6%          | 17.4%      |     0.26 | -18.7%         | 109.2%                      |
| Equal-weight universe        |            0 | 23.5%          | 15.1%         | 11.7%      |     1.12 | -12.2%         | 10.4%                       |
| Equal-weight universe        |            5 | 23.4%          | 15.1%         | 11.7%      |     1.12 | -12.2%         | 10.4%                       |
| Thematic 30                  |            0 | 43.1%          | 27.0%         | 14.5%      |     1.72 | -9.9%          | 10.3%                       |
| SPY                          |            0 | 34.9%          | 22.1%         | 12.2%      |     1.64 | -10.0%         | 0.0%                        |

Signal: {"mean_ic": -0.0075392498481391184, "std_ic": 0.10763806740352486, "t_stat": -0.29716557466754967, "share_positive": 0.5, "top_basket_beat_median_share": 0.6111111111111112, "n_rebalances": 18}

Diagnostics: {"ic_pred_vs_trailing60_return": -0.4721506388511157, "ic_pred_vs_mean60_over_last": 0.6030133070078926, "ic_pred_vs_vol60": 0.18729472206927883, "ic_trailing60_vs_realised": 0.01187431203607556, "ic_vol60_vs_realised": -0.006491803748218694, "basket_trailing60_minus_universe_median": -0.09534723628348019, "basket_vol60_over_universe_median": 1.5435056975511854, "overlap_with_momentum_basket": 0.15000000000000002, "overlap_with_highvol_basket": 0.28148148148148144, "overlap_with_losers_basket": 0.312962962962963, "months_lstm_beat_ew_universe": 10}

Universe check: {"universe": "pit", "excluded_tickers": ["APO", "ARES", "BE", "BX", "CASY", "CIEN", "COHR", "CRH", "CVNA", "DECK", "DELL", "ECHO", "EME", "ERIE", "FERG", "FIX", "FLEX", "GDDY", "HUBB", "IBKR", "ILMN", "JBL", "KKR", "LII", "LITE", "LULU", "MRVL", "P", "SMCI", "SW", "TKO", "TPL", "VEEV", "VRT", "VST", "WDAY", "WSM", "XYZ"], "n_universe_min": 441, "n_universe_max": 454, "current_list_lstm_total_return": 0.5275622562248989, "current_list_lstm_sharpe": 1.4618977697597955, "current_list_contribution_of_excluded_pp": 27.446507149596773}

Luck test: {"draws": 500, "random_median": 0.23629582180207498, "random_p5": 0.1306631535447291, "random_p95": 0.3349421901721619, "lstm_percentile": 42.4, "random_sharpe_median": 1.057287615045952, "random_sharpe_p95": 1.5604750652442294, "lstm_sharpe_percentile": 12.6, "random_vol_median": 0.12484916707676524, "random_turnover_ex_initial": 1.8710819688748161}

Top contributors (pp):
        contribution_pp  months_held date_added_to_index
ticker                                                  
AVGO           3.039912           10          2014-05-08
ANET           2.772904           10          2018-08-28
TSLA           2.167200            9          2020-12-21
HWM            1.386780            6          2016-10-21
VST            1.302115            3          2024-05-08
ON             1.243973            4          2022-06-21
KEY            1.107503            1          1994-03-01
WDC            1.084349            2          2009-07-01
GNRC           0.966049            7          2021-03-22
FICO           0.908020            6          2023-03-20
