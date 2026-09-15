# Prompt For A Fresh AI: Recreate The Strict 7-Day Window vs Polymarket Test

You are a senior Python and quantitative modeling engineer. Recreate the strict walk-forward comparison script for BTC 5-minute direction prediction versus Polymarket implied probabilities.

I need the full procedure rebuilt in Python, including the exact data inputs, preprocessing, model logic, calibration logic, walk-forward evaluation, and output files.

## Goal

Build a script that compares three models on the same future-only test rows:

- `7day_logistic_baseline`
- `polymarket`

The task is to predict whether the current BTC 5-minute candle closes up.

## Required input CSV files

You must use these files from `notes/resources/`:

- `market_data_2sec_weekly5_with_resolutions.csv`
- `binance_1s_for_polymarket_window.csv`

The Polymarket file provides the 2-second order-book data that gets upsampled to 1-second resolution.
The Binance file provides the BTC 1-second market data and engineered features.

## Output folder

Write all results to:

- `researchv2/polymarket_model/strict_7day_vs_pm_results/`

The script should save:

- `overall_comparison.csv`
- `per_split_metrics.csv`
- `row_level_predictions.csv`

## What the script must do

### 1. Load Polymarket data

Load `market_data_2sec_weekly5_with_resolutions.csv` and create these columns:

- `timestamp` from `timestamp_log`
- `start_time` as numeric
- `elapsed` as numeric
- `label` = 1 if `winner` is `up`, else 0
- `implied_prob` = midpoint of `bid_YES` and `ask_YES`, clipped to [0, 1]

Then upsample each market slug from 2-second rows to 1-second rows by carry-forward within each slug.

Keep only rows where:

- `elapsed >= 100`
- `elapsed <= 290`

### 2. Load or fetch Binance BTC data

Use `binance_1s_for_polymarket_window.csv` as the preferred cache.

If the cache exists:

- load it
- parse `open_time` as UTC datetime
- filter it to the Polymarket timestamp range

If it does not exist:

- fetch BTCUSDT 1-second klines from Binance
- engineer the BTC features
- save the cache file for reuse

The Binance fetch should use the spot API and 1-second interval.

### 3. Engineer BTC features

The BTC DataFrame must contain these features:

- `close`
- `volume`
- `number_of_trades`
- `log_return_1s`
- `inst_vol_60s`
- `inst_vol_60s_bps`
- `trend_mean_60s`
- `trend_zscore`
- `buy_volume_ratio`
- `vwap`
- `return_from_candle_open`
- `seconds_to_5m_close`
- `fraction_of_candle_elapsed`

Feature engineering details:

- `log_return_1s = log(close).diff()`
- `inst_vol_60s` is rolling std over 60 seconds with a small minimum period
- `inst_vol_60s_bps = inst_vol_60s * 10000`
- `trend_mean_60s` is the rolling mean of log returns
- `trend_zscore = trend_mean_60s / (inst_vol_60s + 1e-12)`
- `vwap = quote_asset_volume / volume`
- `buy_volume_ratio = taker_buy_base_asset_volume / volume`
- `return_from_candle_open = close / first_open_of_current_5m - 1`
- `seconds_to_5m_close` is the remaining seconds until the 5-minute candle ends
- `fraction_of_candle_elapsed = 1 - seconds_to_5m_close / 300`

### 4. Build the target from actual BTC price movement

This is critical.

The target must be created from actual BTC 5-minute candles, not from Polymarket labels.

For each 5-minute candle:

- `candle_open` = first open in the 5-minute window
- `candle_close` = last close in the 5-minute window
- `target_up = 1` if `candle_close > candle_open`, else `0`

Merge `target_up` back onto the per-second BTC rows.

Do not use the Polymarket `label` as the target.

### 5. Align BTC and Polymarket rows by timestamp

Merge the BTC rows and Polymarket rows on exact second timestamps:

- left key: `open_time`
- right key: `timestamp`

Keep the merged rows only where the features and `target_up` are present.

### 6. Use a strict 7-day walk-forward evaluation

The evaluation logic must be future-only.

For each split:

- Train on the prior 7 days
- Test on the next 6 hours
- Step forward by 6 hours
- Never leak future rows into training
- **CRITICAL**: Discard any evaluating split where the `train_data` has fewer than 100 rows or the `test_data` has fewer than 50 rows to prevent computing metrics on statistically insignificant sample distributions.

For each split, evaluate all three models on exactly the same test rows.

### 7. Baseline model

Create a standalone function `make_base_pipeline(feature_cols)` that constructs the base pipeline safely. The pipeline must:
- Use a `ColumnTransformer` that applies a `Pipeline` of `SimpleImputer(strategy="median")` and `StandardScaler()` specifically to the provided `feature_cols`, using `remainder="drop"`.
- Use a `LogisticRegression(max_iter=2000, solver="lbfgs")` as the final estimator.

During the walk-forward evaluation:
- Wrap this base pipeline with `CalibratedClassifierCV(method="sigmoid", cv=3)`.
- Fit this calibrated model on the full training split.
- **CRITICAL:** When predicting test probabilities, you must mathematically clip the output array immediately upon returning from `predict_proba` strictly between `[1e-6, 1 - 1e-6]` using `np.clip` to prevent infinite log-loss penalties and extreme edge probability artifacts. Store bounded variables natively.

### 9. Polymarket model

The Polymarket model is simply:

- the `implied_prob` midpoint from the order book

No training is required for this model.

### 10. Metrics to compute for every split

For each model on each split compute:

- accuracy
- roc_auc
- log_loss
- brier
- calibration_gap_pp

Where:

- `calibration_gap_pp = mean(abs(pred_prob - y_true)) * 100`

### 11. Overall summary metrics

Aggregate the split metrics into a summary CSV with mean and std for each model.

Include at least:

- `accuracy_mean`
- `accuracy_std`
- `roc_auc_mean`
- `roc_auc_std`
- `log_loss_mean`
- `log_loss_std`
- `brier_mean`
- `brier_std`
- `calibration_gap_pp_mean`
- `calibration_gap_pp_std`

### 12. Row-level predictions

Also save a row-level CSV containing, for each test row:

- split index
- timestamp
- slug
- elapsed
- `y_true`
- `prob_7d_baseline`
- `prob_polymarket`

## Suggested script structure

Create a single script, for example:

- `researchv2/polymarket_model/strict_7day_vs_polymarket.py`

Recommended functions:

1. `make_base_pipeline(feature_cols)`
2. `bucket_series(elapsed)`
3. `fit_sigmoid_calibrator(x_prob, y)`
4. `predict_sigmoid(calibrator, x_prob)`
5. `fit_bucket_calibrators(calib_df, ...)`
6. `apply_bucket_calibrators(raw_prob, elapsed, calibrators)`
7. `tune_shrinkage_lambda(p_calib, y_calib, base_rate, ...)`
8. `fetch_klines_1s(symbol, start_ms, end_ms)`
9. `fetch_klines_1s_chunked(symbol, start_ms, end_ms, chunk_hours=6)`
10. `build_binance_df(raw_rows)`
11. `load_polymarket_1s(polymarket_csv)`
12. `compute_metrics(y_true, y_prob)`
13. `run_walk_forward_test(binance_df, polymarket_df, training_days=7, test_hours=6, step_hours=6)`

## Reproduction procedure

1. Load `market_data_2sec_weekly5_with_resolutions.csv`.
2. Upsample the Polymarket order-book rows to 1-second resolution.
3. Load `binance_1s_for_polymarket_window.csv`.
4. Ensure Binance features and `target_up` are present.
5. Merge BTC and Polymarket by exact timestamp.
6. Run the 7-day walk-forward evaluation.
7. Save CSV outputs to `strict_7day_vs_pm_results/`.
8. Print the final comparison table and winner by metric count.

## Validation checklist

Before trusting a run, verify that:

- the BTC target is built from actual candle open/close values
- Polymarket labels are not used as the BTC target
- train and test windows are strictly disjoint
- the cache CSV contains the expected BTC feature columns
- the output CSVs were written successfully
- the summary metrics match the expected 7-day baseline vs Polymarket comparison

## Exact run command
Activate the virtual environment `.venv` to load all dependencies.

```powershell
c:/Users/fiona/Desktop/polymarket_bot/.venv/Scripts/python.exe researchv2/polymarket_model/strict_7day_vs_polymarket.py
```

## Expected result

The output should reproduce a strict future-only comparison between:

- the calibrated 7-day logistic BTC model
- Polymarket implied probabilities

The final comparison should write the summary CSVs and print the overall winner.
