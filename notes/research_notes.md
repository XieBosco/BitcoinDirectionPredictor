# Strong Ways to Improve Predictive Power
*(Ordered by likely impact)*

## 1. Regime-Specialized Models
Train separate models for different volatility/trend regimes, then route each prediction through a regime classifier. One global model is often averaging across very different market behaviors.

## 2. Multi-Horizon Feature Stack
Add features at multiple windows (e.g., 5s, 15s, 30s, 60s, 180s):
* Return, realized volatility, range, and skew
* Volume/trade-intensity acceleration
* Micro-trend slope changes

*Note: This usually improves early-bucket performance (120-300s to close), where your model is likely weakest.*

## 3. Order-Book and Trade-Flow Features
If available, include:
* Top-of-book imbalance
* Spread and spread changes
* Depth imbalance (L1-L10)
* Aggressive buy/sell flow imbalance

*Note: These often provide the biggest lift for short-horizon direction models.*

## 4. Time-Aware Ensembling
Use a mixture model segmented by `seconds_to_close`:
* **Model A:** 240-300s
* **Model B:** 120-240s
* **Model C:** 0-120s

Your current results already show different difficulty by time-to-close, making a single decision boundary suboptimal.

## 5. Better Calibration by Bucket
Calibrate probabilities separately by time-to-close bucket (or regime × bucket), rather than using one global calibrator. This improves probability quality and can improve log loss even if overall accuracy barely moves.

## 6. Rolling Retraining + Recency Weighting
Use rolling updates (e.g., every 1-6 hours) and weight recent data higher than older data. Crypto microstructure drifts quickly; this approach often improves forward stability.

## 7. Purged/Embargoed Validation for Development
For model selection and hyperparameter tuning, use purged folds with an embargo window around test spans. This provides more leakage-robust tuning than naive k-folds.

## 8. Hyperparameter Search Focused on Log Loss
Optimize directly for out-of-sample log loss (not accuracy), and include:
* Regularization strength grids for logistic models
* Monotonic constraints / leaf tuning for boosted trees
* Early stopping on strictly future validation slices

## 9. Feature Interaction Models
Try models that capture nonlinear interactions better than plain logistic regression:
* **LightGBM / XGBoost** with careful regularization
* Alternatively, keep the logistic model but add explicit interaction terms (e.g., `return × volatility`, `imbalance × seconds_to_close`).

## 10. Label Quality Improvements
Use cleaner target definitions:
* Keep the current close-up/down label, but drop ultra-tiny body candles near zero movement (these act as ambiguous labels).
* Optionally add **confidence-weighted training** (where a larger candle body equals higher label confidence).

## 11. Cost-Sensitive Thresholding for Deployment
Even if training stays probabilistic, set trading thresholds by expected value (aware of fees and slippage), not a fixed `0.5` probability threshold. This won’t change log loss, but it improves your practical trading edge.

## 12. More Anchors and Longer Walk-Forward History
Your current 3-anchor test is too small for stable conclusions. Expanding to 14-30 days with many anchors improves model selection reliability and helps avoid overfitting to a few specific market periods.