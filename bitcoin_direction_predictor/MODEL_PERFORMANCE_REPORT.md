# Bitcoin 5-Minute Direction Predictor Performance Report

## Executive Summary
This report evaluates a **calibrated Logistic Regression model** that predicts whether Bitcoin will close **UP or DOWN** in 5-minute intervals. The model combines Polymarket implied log-odds and intra-candle microstructure variables, trained on Polymarket 2-second order book data and tested against **actual Binance 1-second spot ground truth data** across the exact corresponding time period (`Feb 23, 2026` to `Mar 5, 2026`).

By modeling in logit space and evaluating out-of-sample on strictly future 5-minute candle blocks, the model produces calibrated direction probabilities $P(\text{Up})$ and binary directional calls evaluated against Binance spot settlement.

---

## Dataset & Splitting Methodology
- **Polymarket Raw Records**: 113,245 observations across 1,191 unique 5-minute markets.
- **Intra-Candle Filtering**: 1,053 records outside elapsed bounds [100s, 290s] were dropped, leaving **112,192** aligned observations.
- **Binance Ground Truth**: 834,001 1-second candles with ground truth label `target_up = 1` if `candle_close > candle_open` else `0`.
- **Chronological Split**:
  - **Train Period**: `2026-02-23 11:51:41+00:00` to `2026-03-02 15:04:49+00:00` (84,055 rows, 893 candles, 75% temporal split).
  - **Test Period**: `2026-03-02 15:06:41+00:00` to `2026-03-05 03:31:41+00:00` (28,137 rows, 298 candles, 25% temporal split).
  - **Zero Lookahead Leakage**: Train and test partitions are strictly partitioned on 5-minute candle boundaries.
- **Oracle Settlement Basis Risk**: In 21 candles (1,998 rows, or 1.78% of data), Polymarket's external oracle resolution differed from Binance spot 5-minute return direction due to price feed / strike boundary nuances.

---

## Model Coefficients & Odds Ratios
Features are standardized to ensure $L_2$ regularization treats microstructure features proportionately alongside implied probability:

| Feature | Standardized Coef | Exp(Coef) / Odds Ratio | Description |
|:---|:---:|:---:|:---|
| `Intercept` | `0.0420` | `N/A` | Base log-odds |
| `logit_implied` | `+2.7393` | `15.4765` | Polymarket implied log-odds |
| `spread` | `-0.0479` | `0.9532` | Order book spread |
| `fraction_elapsed` | `+0.0032` | `1.0032` | Intra-candle elapsed fraction |
| `btc_gap_bps` | `+0.2009` | `1.2225` | BTC strike price gap (bps) |


> [!NOTE]
> The primary driver is Polymarket's implied log-odds (`coef = 2.7393`). Transforming implied probability to log-odds space eliminates probability squashing, enabling calibrated predictions even during high-certainty regimes ($p > 0.99$ or $p < 0.01$).

---

## Overall Predictive Performance vs Baselines

| Metric | Logistic Regression (Model) | 95% Cluster Bootstrap CI (LR) | Polymarket Raw Implied | Majority Class Baseline | Superior Model |
|:---|:---:|:---:|:---:|:---:|:---:|
| **Accuracy** | **0.7900** (79.00%) | [0.7559, 0.8212] | 0.7920 (79.20%) | 0.5068 (50.68%) | **Polymarket (-0.21 pp)** |
| **ROC AUC** | **0.8843** | [0.8538, 0.9113] | 0.8840 | 0.5000 | **Logistic Regression** |
| **Log Loss** | **0.4195** | [0.3751, 0.4673] | 0.4207 | 0.6931 | **Logistic Regression** |
| **Brier Score** | **0.1392** | [0.1223, 0.1571] | 0.1392 | 0.2500 | **Logistic Regression** |
| **Calibration Gap (pp)** | **1.45 pp** | [0.07, 5.23] | 0.83 pp | 0.17 pp | **Polymarket** |
| **ECE (10-bin)** | **2.37 pp** | N/A | 2.47 pp | 0.17 pp | **Logistic Regression** |
| **Precision** | **0.7814** | N/A | 0.7884 | 0.5068 | **Polymarket** |
| **Recall** | **0.8130** | N/A | 0.8059 | 1.0000 | **Logistic Regression** |
| **F1-Score** | **0.7969** | N/A | 0.7971 | 0.6727 | **Polymarket** |

### Confusion Matrix (Logistic Regression on Test Set)
| Actual \ Predicted | Predicted Down (0) | Predicted Up (1) |
|:---|:---:|:---:|
| **Actual Down (0)** | **TN = 10,633** | FP = 3,243 |
| **Actual Up (1)**   | FN = 2,667 | **TP = 11,594** |


---

## Intra-Candle Performance by Elapsed Time
Predictability changes substantially as the 5-minute candle progresses toward resolution:

| Elapsed Bucket | Samples | Actual Up Rate | LR Accuracy | Raw PM Accuracy | LR AUC | LR Log Loss | LR Brier |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 100-129s | 4,439 | 0.507 | 0.6979 | 0.6963 | 0.7642 | 0.5789 | 0.1983 |
| 130-159s | 4,439 | 0.507 | 0.7150 | 0.7173 | 0.7969 | 0.5436 | 0.1843 |
| 160-189s | 4,437 | 0.507 | 0.7595 | 0.7703 | 0.8556 | 0.4635 | 0.1552 |
| 190-219s | 4,438 | 0.507 | 0.8105 | 0.8123 | 0.8998 | 0.3953 | 0.1289 |
| 220-249s | 4,436 | 0.507 | 0.8411 | 0.8406 | 0.9249 | 0.3480 | 0.1100 |
| 250-290s | 5,948 | 0.507 | 0.8838 | 0.8840 | 0.9621 | 0.2466 | 0.0788 |


### Key Timing Insights:
1. **Early in Candle (100-129s)**: Higher uncertainty; ROC-AUC is 0.7642 and accuracy is 69.79%.
2. **Late in Candle (250-290s)**: High certainty; ROC-AUC rises to **0.9621** and accuracy reaches **88.38%**.

---

## Conclusion & Verification Summary
- **Calibrated Log-Odds Modeling**: Transforming implied probability to logit space eliminates probability squashing, achieving competitive scoring rules and strong discrimination (0.8843 ROC-AUC).
- **Intra-Candle Progression**: Predictability scales monotonically as the candle closes, reaching ~88.4% accuracy and 0.9621 ROC-AUC in the final 40 seconds.
- **Execution Consistency**: Dynamic winner calculations and cluster-level block bootstrapping provide an honest, reproducible benchmark against Polymarket raw implied odds.
