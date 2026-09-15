# Prompt For A Fresh AI: Recreate The Robust Walk-Forward 5-Minute Candle Direction Model

Use this prompt with a new AI session that has no prior context.

---

You are a senior quantitative ML engineer. Recreate an end-to-end BTCUSDT predictive modeling pipeline in Python that estimates the probability that the current 5-minute candle closes up.

I need a robust implementation with strict walk-forward testing (future-only evaluation), not only cross-validation. Build all scripts, run the workflow, and produce output CSV reports.

## Goal

At every 1-second timestamp inside a 5-minute candle, predict:

- P(5-minute candle closes up)

Define target:

- target_up = 1 if candle_close > candle_open for that candle
- target_up = 0 otherwise
- Drop doji candles where candle_close == candle_open

## Environment and assumptions

- Python 3.10+
- Use Binance Spot API endpoint: /api/v3/klines
- Symbol: BTCUSDT
- Base interval: 1-second klines
- Use pandas, numpy, requests, scikit-learn
- Use logistic regression as primary model with probability calibration

## What to build

Create a folder:

- research/window_generalization_tests/

Create these scripts:

1. fetch_backtest_data.py
2. compare_training_windows.py

Create output folder:

- research/window_generalization_tests/output/

## Script 1: fetch_backtest_data.py (data generation)

Purpose:

- Pull multi-day 1-second BTCUSDT data from Binance
- Engineer the same features needed by the model
- Save enriched CSV for backtesting

CLI args:

- --symbol (default BTCUSDT)
- --days (default 8)
- --output (default research/window_generalization_tests/btcusdt_1s_backtest_data.csv)
- --sleep-seconds (default around 0.03)

Implementation requirements:

1. Fetch klines with pagination:
   - interval=1s
   - limit=1000 per request
   - startTime/endTime control
   - loop until end is reached
2. Convert raw columns to typed DataFrame
3. Convert open_time and close_time to UTC datetimes
4. Engineer these features exactly:
   - log_return_1s = log(close).diff()
   - inst_vol_60s = rolling std of log_return_1s over window=60, min_periods=10
   - inst_vol_60s_bps = inst_vol_60s * 10000
   - trend_mean_60s = rolling mean log return over 60s
   - trend_zscore = trend_mean_60s / (inst_vol_60s + 1e-12)
   - vwap = quote_asset_volume / volume (safe divide)
   - buy_volume_ratio = taker_buy_base_asset_volume / volume (safe divide)
5. Build market_regime labels using volatility quantiles:
   - vol_low = 30th percentile of inst_vol_60s
   - vol_high = 70th percentile of inst_vol_60s
   - labels:
     - calm: inst_vol_60s <= vol_low
     - volatile_trend_up: inst_vol_60s >= vol_high and trend_zscore >= 1.5
     - volatile_trend_down: inst_vol_60s >= vol_high and trend_zscore <= -1.5
     - high_vol_chop: inst_vol_60s >= vol_high and abs(trend_zscore) < 0.5
     - otherwise normal
6. Deduplicate by open_time and sort chronologically
7. Save CSV
8. Print progress every ~100 API calls

## Script 2: compare_training_windows.py (model + robust evaluation)

Purpose:

- Compare training windows (1, 2, 3, 7 days)
- Use strict walk-forward future-only test blocks
- Output per-split and summary metrics
- Output time-to-close bucket reports per window

CLI args:

- --input (default research/window_generalization_tests/btcusdt_1s_backtest_data.csv)
- --windows-days (default 1,2,3,7)
- --test-hours (default 6)
- --step-hours (default 6)
- --calibration choices: none/sigmoid/isotonic (default sigmoid)
- --output-dir (default research/window_generalization_tests/output)

### Data prep requirements

1. Load CSV and enforce required columns:
   - open_time, open, close, volume, number_of_trades, inst_vol_60s, inst_vol_60s_bps, log_return_1s
2. Parse open_time to UTC and sort
3. Create candle_5m_start = floor(open_time, 5min)
4. Build candle-level target_up from candle open/close and merge back to second-level rows
5. Drop doji candles
6. Create timing features:
   - seconds_to_5m_close
   - fraction_of_candle_elapsed = 1 - seconds_to_5m_close / 300
7. Create return_from_candle_open = close / first_open_of_5m - 1
8. Clip buy_volume_ratio to [0, 1] if present

### Feature set

Use this exact candidate list and keep available ones:

- close
- volume
- number_of_trades
- log_return_1s
- inst_vol_60s
- inst_vol_60s_bps
- trend_mean_60s
- trend_zscore
- buy_volume_ratio
- vwap
- return_from_candle_open
- seconds_to_5m_close
- fraction_of_candle_elapsed

### Model requirements

Base model:

- LogisticRegression(max_iter=2000, solver='lbfgs')

Preprocessing:

- SimpleImputer(strategy='median')
- StandardScaler
- Put in sklearn Pipeline/ColumnTransformer

Calibration:

- If calibration != none, wrap with CalibratedClassifierCV(cv=3, method=sigmoid or isotonic)

### Walk-forward protocol (critical)

Use anchors to enforce chronology:

1. max_window = max(windows_days)
2. first_anchor = min(open_time) + max_window days
3. last_anchor = max(open_time) - test_hours
4. anchors every step_hours

For each anchor and each window_days:

- Train range: [anchor - window_days, anchor)
- Test range: [anchor, anchor + test_hours)
- Train and test must be strictly disjoint and chronological
- Skip split if train/test empty or train has single class

This is the key robust evaluation design.

### Per-split metrics

For each successful split, compute:

- accuracy
- roc_auc (if both classes in test)
- brier
- log_loss
- actual_up_rate
- avg_pred_up_prob
- calibration_gap = abs(avg_pred_up_prob - actual_up_rate)

Save all split rows to:

- per_split_metrics.csv

### Summary metrics by window

Aggregate mean/std by window_days:

- accuracy_mean/std
- roc_auc_mean/std
- brier_mean/std
- log_loss_mean/std
- calibration_gap_mean/std

Sort primarily by log_loss_mean (ascending), then brier_mean (ascending), then accuracy_mean (descending).

Save:

- window_summary.csv

### Anchor win counts

For each anchor:

- best window by log_loss
- best window by brier
- best window by accuracy

Count wins and save:

- anchor_wins.csv

### Time-to-close bucket reports per window

Using all out-of-sample predictions from successful walk-forward splits, generate bucket metrics with bins:

- 0-5s
- 5-15s
- 15-30s
- 30-60s
- 60-120s
- 120-180s
- 180-240s
- 240-300s

For each bucket compute:

- samples
- actual_up_rate
- avg_pred_up_prob
- accuracy
- brier
- log_loss
- roc_auc (if both classes)

Save one file per window:

- output/time_to_close_bucket_reports/time_to_close_bucket_report_1d.csv
- output/time_to_close_bucket_reports/time_to_close_bucket_report_2d.csv
- output/time_to_close_bucket_reports/time_to_close_bucket_report_3d.csv
- output/time_to_close_bucket_reports/time_to_close_bucket_report_7d.csv

Also save combined table:

- output/time_to_close_bucket_report_by_window.csv

## Commands to run (after script creation)

1) Fetch backtest dataset (minimum 8 days):

```powershell
c:/Users/fiona/Desktop/polymarket_bot/.venv/Scripts/python.exe research/window_generalization_tests/fetch_backtest_data.py --symbol BTCUSDT --days 8 --output research/window_generalization_tests/btcusdt_1s_backtest_data.csv
```

2) Run walk-forward comparison:

```powershell
c:/Users/fiona/Desktop/polymarket_bot/.venv/Scripts/python.exe research/window_generalization_tests/compare_training_windows.py --input research/window_generalization_tests/btcusdt_1s_backtest_data.csv --windows-days 1,2,3,7 --test-hours 6 --step-hours 6 --calibration sigmoid --output-dir research/window_generalization_tests/output
```

## Expected behavior and interpretation

- Performance should generally improve as time-to-close decreases
- 0-5s bucket should be best, 240-300s typically weakest
- Some anchor-to-anchor variability is normal due to regime shifts
- Select best window using probability quality metrics first (log loss, brier), not only accuracy

## Acceptance checklist

Before finishing, verify:

1. All scripts run without errors
2. CSV outputs are created in expected locations
3. per_split_metrics.csv has one row per (anchor, window)
4. window_summary.csv ranks windows by log loss as specified
5. Per-window bucket files exist for 1d/2d/3d/7d
6. Combined bucket report includes window_days column
7. No leakage: test period always starts at anchor and is strictly after training range

## Optional enhancement (if time permits)

Add a small helper printout after run completion:

- Best window by summary ranking and its mean log_loss/brier/accuracy
- Number of anchors evaluated
- Number of successful vs skipped split evaluations

Return final response with:

- Key metrics table (window_summary)
- Best window
- Paths to all generated outputs

---

End of prompt.
