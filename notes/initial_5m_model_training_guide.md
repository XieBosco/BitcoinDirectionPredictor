# Initial 5-Minute Candle Direction Model Training Guide

This guide explains exactly how the initial model was built to estimate whether the current 5-minute candle will close up or down.

## 1. Objective

At every 1-second timestamp inside a 5-minute candle, predict:

- Probability that the containing 5-minute candle closes up
- Binary target is:
  - 1 when candle_close > candle_open
  - 0 otherwise

## 2. Files Used

- Data fetch and feature engineering: research/fetch_data.py
- Model training and evaluation: research/train_5m_close_direction_model.py

## 3. Step-by-Step Process

### Step 1: Fetch 1-second BTCUSDT data from Binance

Run:

```powershell
c:/Users/fiona/Desktop/polymarket_bot/.venv/Scripts/python.exe research/fetch_data.py --output research/btcusdt_1s_last24h.csv
```

What this does:

1. Pulls Binance klines at 1-second resolution with pagination.
2. Converts raw columns to typed numeric/time values.
3. Produces enriched per-second features.

Output dataset:

- research/btcusdt_1s_last24h.csv

### Step 2: Build the label at the candle level

Inside training:

1. Compute candle_5m_start by flooring each open_time to 5-minute boundaries.
2. Group rows by candle_5m_start.
3. For each 5-minute candle:
   - candle_open = first open
   - candle_close = last close
   - target_up = 1 if candle_close > candle_open, else 0
4. Drop exact doji candles where close equals open.
5. Merge target_up back onto all 1-second rows in that candle.

This gives each second inside a candle the final outcome label for that candle.

### Step 3: Add time-position features inside each 5-minute candle

Added features:

1. seconds_to_5m_close
2. fraction_of_candle_elapsed = 1 - seconds_to_5m_close / 300
3. return_from_candle_open = close / candle_open_of_current_5m - 1

These are crucial because predictability changes as the candle gets closer to close.

### Step 4: Use engineered market features

Core features used by the trainer:

1. close
2. volume
3. number_of_trades
4. log_return_1s
5. inst_vol_60s
6. inst_vol_60s_bps
7. trend_mean_60s
8. trend_zscore
9. buy_volume_ratio
10. vwap
11. return_from_candle_open
12. seconds_to_5m_close
13. fraction_of_candle_elapsed

### Step 5: Build preprocessing and model pipeline

For the baseline logistic model:

1. Median imputation for missing numeric values
2. Standard scaling
3. LogisticRegression with lbfgs solver
4. Probability calibration using sigmoid

Why:

- Scaling helps stability for linear models
- Calibration improves probability quality (important for log loss and decision confidence)

### Step 6: Use leakage-safe validation with GroupKFold

Validation split key:

- group = candle_5m_start

Effect:

- All rows from one 5-minute candle stay entirely in train or test
- Prevents leakage from nearly identical seconds within the same candle

### Step 7: Train on each fold and collect out-of-fold predictions

For each fold:

1. Fit model on train groups
2. Predict probabilities on test groups
3. Store out-of-fold probability per row
4. Compute metrics:
   - accuracy
   - roc_auc
   - log_loss
   - brier

Out-of-fold probabilities are then used to build realistic diagnostics.

### Step 8: Build time-to-close bucket diagnostics

Rows are bucketed by seconds_to_5m_close:

- 0-5s
- 5-15s
- 15-30s
- 30-60s
- 60-120s
- 120-180s
- 180-240s
- 240-300s

For each bucket, compute:

1. samples
2. actual_up_rate
3. avg_pred_up_prob
4. accuracy
5. brier
6. roc_auc

This shows where the model performs best and worst by time remaining.

### Step 9: Save outputs

Training writes:

1. research/btcusdt_1s_with_up_prob.csv
2. research/model_cv_metrics.csv
3. research/time_to_close_bucket_report.csv

## 4. Exact Reproduction Commands

### Fetch data

```powershell
c:/Users/fiona/Desktop/polymarket_bot/.venv/Scripts/python.exe research/fetch_data.py --output research/btcusdt_1s_last24h.csv
```

### Train initial baseline model

```powershell
c:/Users/fiona/Desktop/polymarket_bot/.venv/Scripts/python.exe research/train_5m_close_direction_model.py --input research/btcusdt_1s_last24h.csv --models logreg --calibration sigmoid --n-splits 5 --pred-output research/btcusdt_1s_with_up_prob.csv --metrics-output research/model_cv_metrics.csv --bucket-output research/time_to_close_bucket_report.csv
```

## 5. Validation Checklist

Before trusting a run, verify:

1. Input CSV contains required columns
2. Both classes exist in target_up
3. GroupKFold is grouped by candle_5m_start
4. Probabilities are between 0 and 1
5. Time bucket results show logical behavior (generally stronger near close)

## 6. Why this initial model worked

1. Leakage control via candle-grouped validation
2. Strong timing features for intra-candle context
3. Calibrated logistic probabilities
4. Large number of 1-second observations in 24h
