Window Generalization Tests

Purpose
- Compare which training window generalizes best to future candles.
- Default candidates: 1 day, 2 days, 3 days, 7 days.
- Evaluation is walk-forward and strictly chronological (future-only test blocks).

Script
- compare_training_windows.py
- fetch_backtest_data.py

Generate backtest dataset from Binance API
- c:/Users/fiona/Desktop/polymarket_bot/.venv/Scripts/python.exe research/window_generalization_tests/fetch_backtest_data.py --symbol BTCUSDT --days 8 --output research/window_generalization_tests/btcusdt_1s_backtest_data.csv
- For more stable walk-forward comparisons, consider 10-14 days of data.

Required input
- Enriched 1-second CSV containing the same features used by training.
- At minimum: open_time, open, close, volume, number_of_trades,
  inst_vol_60s, inst_vol_60s_bps, log_return_1s.

Default output files
- output/per_split_metrics.csv
- output/window_summary.csv
- output/anchor_wins.csv

Example run
- c:/Users/fiona/Desktop/polymarket_bot/.venv/Scripts/python.exe research/window_generalization_tests/compare_training_windows.py --input research/window_generalization_tests/btcusdt_1s_backtest_data.csv --windows-days 1,2,3,7 --test-hours 6 --step-hours 6 --calibration sigmoid --output-dir research/window_generalization_tests/output

Important
- To evaluate a 7-day training window with a 6-hour future test block,
  you need more than ~7.25 days of chronological data.
- If your file only has 24 hours of data, 2/3/7-day windows will be skipped
  or the run will stop with an insufficient-span message.
