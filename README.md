<p align="center">
  <h1 align="center">📈 Bitcoin 5-Minute Direction Predictor</h1>
</p>
<p align="center">
    <em>Machine learning pipeline predicting 5-minute Bitcoin candle direction with logistic regression and Polymarket CLOB integration.</em>
</p>


---

**Performance Report**: [MODEL_PERFORMANCE_REPORT.md](bitcoin_direction_predictor/MODEL_PERFORMANCE_REPORT.md)

**Source Code**: [https://github.com/XieBosco/BitcoinDirectionPredictor](https://github.com/XieBosco/BitcoinDirectionPredictor)

---

**Bitcoin Direction Predictor** predicts whether Bitcoin closes **UP or DOWN** in 5-minute intervals at 1-second frequency by combining Polymarket CLOB implied log-odds with intra-candle microstructure features.

Key features:

* **Calibrated Log-Odds Modeling**: Transforms Polymarket CLOB implied probabilities to logit space to eliminate probability squashing.
* **Microstructure Signals**: Uses order book spread, elapsed candle fraction, strike gap (bps), and implied log-odds.
* **Walk-Forward Validation**: Evaluated on 112,192 aligned observations across 1,191 markets with a 75/25 chronological split on candle boundaries to prevent lookahead leakage.
* **Scoring Rule Benchmarking**: Evaluated against Polymarket raw implied odds across Brier Score, Log Loss, ROC-AUC, and 10-bin ECE.

---

## Performance Benchmarks

Evaluated out-of-sample on **28,137 test seconds** (298 candles) against Binance 1-second spot:

### Overall Performance vs Baselines

| Metric | Logistic Regression (Model) | 95% Cluster Bootstrap CI | Polymarket Raw Implied | Majority Class Baseline | Superior Model |
|:---|:---:|:---:|:---:|:---:|:---:|
| **Accuracy** | **0.7900** (79.00%) | [0.7559, 0.8212] | 0.7920 (79.20%) | 0.5068 (50.68%) | Polymarket (-0.20 pp) |
| **ROC AUC** | **0.8843** | [0.8538, 0.9113] | 0.8840 | 0.5000 | **Logistic Regression** |
| **Log Loss** | **0.4195** | [0.3751, 0.4673] | 0.4207 | 0.6931 | **Logistic Regression** |
| **Brier Score** | **0.1392** | [0.1223, 0.1571] | 0.1392 | 0.2500 | **Logistic Regression** |
| **Calibration Gap** | **1.45 pp** | [0.07, 5.23] | 0.83 pp | 0.17 pp | Polymarket |
| **ECE (10-bin)** | **2.37 pp** | N/A | 2.47 pp | 0.17 pp | **Logistic Regression** |

### Intra-Candle Progression

Predictability increases as the candle approaches close:

| Elapsed Bucket | Samples | Actual Up Rate | LR Accuracy | Raw PM Accuracy | LR ROC-AUC | LR Log Loss |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **100–129s** | 4,439 | 0.507 | 0.6979 | 0.6963 | 0.7642 | 0.5789 |
| **130–159s** | 4,439 | 0.507 | 0.7150 | 0.7173 | 0.7969 | 0.5436 |
| **160–189s** | 4,437 | 0.507 | 0.7595 | 0.7703 | 0.8556 | 0.4635 |
| **190–219s** | 4,438 | 0.507 | 0.8105 | 0.8123 | 0.8998 | 0.3953 |
| **220–249s** | 4,436 | 0.507 | 0.8411 | 0.8406 | 0.9249 | 0.3480 |
| **250–290s** | 5,948 | 0.507 | **0.8838** | **0.8840** | **0.9621** | **0.2466** |

---

## Testing & Verification

Run the test suite:

```console
$ pytest bitcoin_direction_predictor/test_predictor.py -v

============================= test session starts ==============================
platform win32 -- Python 3.10.11, pytest-9.0.2
collected 6 items

bitcoin_direction_predictor/test_predictor.py::test_logistic_regression_predictor PASSED        [ 16%]
bitcoin_direction_predictor/test_predictor.py::test_single_sample_and_series_inference PASSED  [ 33%]
bitcoin_direction_predictor/test_predictor.py::test_model_coefficients_and_scaling PASSED      [ 50%]
bitcoin_direction_predictor/test_predictor.py::test_compute_metrics PASSED                     [ 66%]
bitcoin_direction_predictor/test_predictor.py::test_cluster_bootstrap PASSED                   [ 83%]
bitcoin_direction_predictor/test_predictor.py::test_pipeline_outputs_exist PASSED             [100%]

============================== 6 passed in 4.44s ==============================
```
