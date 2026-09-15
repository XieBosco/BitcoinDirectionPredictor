# Ranked Feature Roadmap for Next-Gen Polymarket Models

This roadmap ranks feature families by a mix of expected lift, orthogonality to the current Polymarket implied-probability signal, and implementation cost.

## Tier 1: Highest Priority

### 1. Cross-Market BTC Signals
Best first target for true orthogonal information.

Add features from markets that are not the same signal source as the Polymarket book:
- Binance spot return and momentum across several horizons
- Perp funding rate and open interest changes
- Spot-futures basis
- BTC dominance / broad crypto beta when available
- Volatility regime from Binance rather than Polymarket

Why this ranks first:
- It gives the model information from a different venue and microstructure.
- It is less likely to be redundant with the Polymarket order book itself.
- It can explain when Polymarket is lagging or temporarily mispriced.

### 2. Order-Flow and Liquidity Microstructure
If you can get more than top-of-book, this is usually the most valuable local signal.

Add:
- L1 to L10 depth imbalance
- Bid/ask slope or book convexity
- Spread changes and spread acceleration
- Trade aggressor imbalance
- Quote refresh rate and order-book churn

Why this ranks high:
- It is directly related to imminent price pressure.
- It can outperform generic trend features in short-horizon settings.
- It is more orthogonal than simply reusing the mid-price.

### 3. Regime Detection With Separate Experts
Split the problem before modeling it.

Add regime features and/or separate expert models for:
- Calm vs volatile
- Trend up vs trend down
- Chop vs directional continuation
- High-liquidity vs thin-liquidity windows

Why this ranks high:
- Your current results already suggest the market behaves differently by regime.
- A single global classifier likely averages away real structure.
- This is often easier to profit from than raw feature expansion.

## Tier 2: High Value, Moderate Effort

### 4. Multi-Horizon Momentum and Volatility Stack
Build a compact feature ladder over several windows.

Add:
- Returns at 5s, 15s, 30s, 60s, 180s
- Realized volatility over the same windows
- Range expansion and compression
- Trend slope changes
- Volume and trade-intensity acceleration

Why this matters:
- It captures local momentum and exhaustion.
- It improves model visibility across different time-to-close buckets.
- It often helps the model distinguish strong moves from noise.

### 5. Time-to-Close Mixture Model
Use different logic depending on how far the candle is from closing.

Add:
- Separate models for early, middle, and late candle phases
- Time-bucket-specific calibration
- Time x feature interactions

Why this matters:
- The same price action means different things at 280s versus 20s to close.
- A time-aware blend is usually better than one static boundary.

### 6. Probability Calibration by Regime and Bucket
Improve output quality even when class accuracy barely changes.

Add:
- Separate calibrators by time bucket
- Separate calibrators by regime
- Regime x bucket calibration where sample size allows

Why this matters:
- If your edge is small, calibration quality matters a lot.
- Better calibration can improve log loss and deployment decisions.

## Tier 3: Useful, But Usually Secondary

### 7. Label Refinement and Confidence Weighting
Reduce noise in the target.

Add:
- Drop candles with tiny absolute body size
- Weight larger-body candles more heavily
- Separate analysis for strong vs weak closes

Why this matters:
- The model may be fighting ambiguous labels rather than weak features.
- Cleaner targets can improve apparent signal without changing feature logic.

### 8. Rolling Retraining and Recency Weighting
Make the model adapt faster to drift.

Add:
- Sliding training windows
- Higher weights for more recent samples
- More frequent retrains in volatile periods

Why this matters:
- Crypto microstructure drifts quickly.
- This is often a stability improvement rather than a new source of alpha.

### 9. Nonlinear Interaction Models
Use interactions only after the above are in place.

Add:
- return x volatility
- imbalance x time_to_close
- spread x volatility
- Gradient-boosted trees with strong regularization

Why this ranks lower:
- It is useful, but usually only after the core signal families are present.
- Without orthogonal inputs, nonlinear models often just memorize noise.

### 10. Deployment Threshold Optimization
This does not improve the classifier itself, but it improves trading value.

Add:
- Thresholds optimized for expected value instead of accuracy
- Fee and slippage-aware cutoffs
- Trade/no-trade gating by regime

Why this is lower priority:
- It is downstream of prediction quality.
- It helps monetization more than raw predictive power.

## Recommended Build Order

1. Cross-market BTC signals
2. Order-flow and liquidity microstructure
3. Regime-specific experts
4. Multi-horizon feature stack
5. Time-aware calibration and mixture logic
6. Label cleanup and recency weighting
7. Nonlinear interaction models
8. Threshold optimization for deployment

## Practical Rule

If a new feature mostly restates the Polymarket mid-price, spread, or a transformed version of the same book state, it is probably not orthogonal enough to add much.

The best next gains are likely to come from information that Polymarket does not already know, especially:
- Binance spot and perp behavior
- Funding / basis / open interest
- Real order-flow imbalance
- Regime separation before classification
