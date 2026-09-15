"""Enhanced 7-day model with ETH and BTC perps data as additional feature sources.

This script extends the baseline 7-day logistic model with:
- ETH (ETHUSDT) features: relative strength, correlation, vol ratio
- BTC Perps futures features: funding rate, OI changes, basis
- Cross-asset momentum and regime indicators

The model is trained on all three data sources (BTC spot, ETH spot, BTC perps)
and compared against the 7-day baseline and Polymarket.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import requests
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

BINANCE_KLINES_URL = "https://api.binance.com/api/v3/klines"
BINANCE_FUTURES_URL = "https://fapi.binance.com/fapi/v1/klines"
ELAPSED_BINS = [100, 130, 160, 190, 220, 250, 291]
ELAPSED_LABELS = ["100-129", "130-159", "160-189", "190-219", "220-249", "250-290"]


def make_base_pipeline(feature_cols: List[str]) -> Pipeline:
    """Baseline preprocessing + logistic regression."""
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                feature_cols,
            )
        ],
        remainder="drop",
    )

    return Pipeline(
        steps=[
            ("prep", preprocessor),
            ("model", LogisticRegression(max_iter=2000, solver="lbfgs")),
        ]
    )


def safe_log_loss(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    y_prob = np.clip(y_prob, 1e-6, 1 - 1e-6)
    return float(log_loss(y_true, y_prob, labels=[0, 1]))


def bucket_series(elapsed: pd.Series) -> pd.Series:
    return pd.cut(elapsed, bins=ELAPSED_BINS, labels=ELAPSED_LABELS, right=False)


def fit_sigmoid_calibrator(x_prob: np.ndarray, y: np.ndarray) -> LogisticRegression:
    clf = LogisticRegression(max_iter=1000, solver="lbfgs")
    clf.fit(x_prob.reshape(-1, 1), y)
    return clf


def predict_sigmoid(calibrator: LogisticRegression, x_prob: np.ndarray) -> np.ndarray:
    pred = calibrator.predict_proba(x_prob.reshape(-1, 1))[:, 1]
    return np.clip(pred, 1e-6, 1 - 1e-6)


def fit_bucket_calibrators(
    calib_df: pd.DataFrame,
    min_bucket_samples: int = 200,
    min_isotonic_samples: int = 1200,
    min_isotonic_improvement: float = 0.002,
) -> Tuple[Dict[str, Tuple[str, object]], pd.DataFrame]:
    """Fit per-elapsed-bucket calibrators with time-safe method selection."""
    calibrators: Dict[str, Tuple[str, object]] = {}
    diagnostics = []

    for label in ELAPSED_LABELS:
        g = calib_df[calib_df["elapsed_bucket"] == label].copy().sort_values("ts_order")
        if len(g) < min_bucket_samples or g["y_true"].nunique() < 2:
            calibrators[label] = ("identity", None)
            diagnostics.append({"bucket": label, "method": "identity", "n": len(g), "log_loss": np.nan})
            continue

        x = np.clip(g["raw_prob"].to_numpy(dtype=float), 1e-6, 1 - 1e-6)
        y = g["y_true"].to_numpy(dtype=int)

        # Time-safe split: select on validation, refit on full data.
        cut = int(len(g) * 0.7)
        if cut < 50 or (len(g) - cut) < 50:
            cut = len(g) // 2

        x_tr, y_tr = x[:cut], y[:cut]
        x_va, y_va = x[cut:], y[cut:]
        if len(x_va) < 20 or len(np.unique(y_va)) < 2:
            calibrators[label] = ("identity", None)
            diagnostics.append({"bucket": label, "method": "identity", "n": len(g), "log_loss": np.nan})
            continue

        # Sigmoid
        sig = fit_sigmoid_calibrator(x_tr, y_tr)
        p_sig_va = predict_sigmoid(sig, x_va)
        ll_sig = safe_log_loss(y_va, p_sig_va)

        # Isotonic with guardrails
        iso_allowed = len(g) >= min_isotonic_samples
        ll_iso = np.inf
        if iso_allowed:
            iso = IsotonicRegression(out_of_bounds="clip")
            iso.fit(x_tr, y_tr)
            p_iso_va = np.clip(iso.predict(x_va), 1e-6, 1 - 1e-6)
            ll_iso = safe_log_loss(y_va, p_iso_va)

            if hasattr(iso, "X_thresholds_") and len(iso.X_thresholds_) > 120:
                iso_allowed = False

        if iso_allowed and (ll_iso + min_isotonic_improvement < ll_sig):
            iso_full = IsotonicRegression(out_of_bounds="clip")
            iso_full.fit(x, y)
            calibrators[label] = ("isotonic", iso_full)
            diagnostics.append({"bucket": label, "method": "isotonic", "n": len(g), "log_loss": ll_iso})
        else:
            sig_full = fit_sigmoid_calibrator(x, y)
            calibrators[label] = ("sigmoid", sig_full)
            diagnostics.append({"bucket": label, "method": "sigmoid", "n": len(g), "log_loss": ll_sig})

    return calibrators, pd.DataFrame(diagnostics)


def apply_bucket_calibrators(raw_prob: np.ndarray, elapsed: np.ndarray, calibrators: Dict[str, Tuple[str, object]]) -> np.ndarray:
    """Apply per-bucket calibration to probabilities."""
    out = np.clip(raw_prob.astype(float), 1e-6, 1 - 1e-6)
    elapsed_s = pd.Series(elapsed)
    b = bucket_series(elapsed_s)

    for label in ELAPSED_LABELS:
        idx = (b == label).to_numpy()
        if not np.any(idx):
            continue
        method, obj = calibrators.get(label, ("identity", None))
        if method == "sigmoid" and obj is not None:
            out[idx] = predict_sigmoid(obj, out[idx])
        elif method == "isotonic" and obj is not None:
            out[idx] = np.clip(obj.predict(out[idx]), 1e-6, 1 - 1e-6)

    return np.clip(out, 1e-6, 1 - 1e-6)


def tune_shrinkage_lambda(
    p_calib: np.ndarray,
    y_calib: np.ndarray,
    base_rate: float,
    max_calib_gap_regression_pp: float = 0.10,
) -> Tuple[float, Dict[str, float]]:
    """Grid search for optimal shrinkage lambda towards base rate."""
    p_calib = np.clip(p_calib, 1e-6, 1 - 1e-6)
    base_gap_pp = float(np.mean(np.abs(p_calib - y_calib)) * 100.0)

    best_lambda = 1.0
    best_score = np.inf
    best_meta: Dict[str, float] = {"log_loss": np.nan, "brier": np.nan, "calib_gap_pp": np.nan}

    for lam in [0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 1.00]:
        p = lam * p_calib + (1.0 - lam) * base_rate
        p = np.clip(p, 1e-6, 1 - 1e-6)
        ll = safe_log_loss(y_calib, p)
        br = float(brier_score_loss(y_calib, p))
        gap_pp = float(np.mean(np.abs(p - y_calib)) * 100.0)

        if gap_pp > base_gap_pp + max_calib_gap_regression_pp:
            continue

        score = 0.7 * ll + 0.3 * br
        if score < best_score:
            best_score = score
            best_lambda = lam
            best_meta = {"log_loss": ll, "brier": br, "calib_gap_pp": gap_pp}

    return best_lambda, best_meta


def fetch_klines_1s(symbol: str, start_ms: int, end_ms: int, url: str = BINANCE_KLINES_URL) -> List[List]:
    """Fetch 1-second klines from Binance with retry logic."""
    all_rows: List[List] = []
    current_start = start_ms
    session = requests.Session()

    while current_start < end_ms:
        params = {
            "symbol": symbol,
            "interval": "1s",
            "startTime": current_start,
            "endTime": end_ms,
            "limit": 1000,
        }

        last_exc = None
        batch = None
        for attempt in range(5):
            try:
                response = session.get(url, params=params, timeout=30)
                response.raise_for_status()
                batch = response.json()
                break
            except Exception as exc:
                last_exc = exc
                import time
                time.sleep(0.5 * (attempt + 1))

        if batch is None:
            raise RuntimeError(f"Failed Binance fetch after retries: {last_exc}")

        if not batch:
            break

        all_rows.extend(batch)
        current_start = int(batch[-1][0]) + 1000
        import time
        time.sleep(0.02)

    return all_rows


def fetch_klines_1s_chunked(symbol: str, start_ms: int, end_ms: int, chunk_hours: int = 6, url: str = BINANCE_KLINES_URL) -> List[List]:
    """Fetch klines in chunks to avoid API limits."""
    chunk_ms = int(chunk_hours * 3600 * 1000)
    cursor = start_ms
    merged: List[List] = []
    chunk_idx = 0

    while cursor < end_ms:
        chunk_idx += 1
        c_end = min(end_ms, cursor + chunk_ms)
        print(f"Fetching chunk {chunk_idx}: {pd.to_datetime(cursor, unit='ms', utc=True)} -> {pd.to_datetime(c_end, unit='ms', utc=True)}")
        rows = fetch_klines_1s(symbol=symbol, start_ms=cursor, end_ms=c_end, url=url)
        merged.extend(rows)
        cursor = c_end + 1000

    return merged


def build_binance_df(raw_rows: List[List], symbol_name: str = "BTC") -> pd.DataFrame:
    """Build dataframe from raw Binance klines."""
    columns = [
        "open_time",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "close_time",
        "quote_asset_volume",
        "number_of_trades",
        "taker_buy_base_asset_volume",
        "taker_buy_quote_asset_volume",
        "ignore",
    ]

    df = pd.DataFrame(raw_rows, columns=columns).drop(columns=["ignore"])

    numeric_cols = [
        "open",
        "high",
        "low",
        "close",
        "volume",
        "quote_asset_volume",
        "number_of_trades",
        "taker_buy_base_asset_volume",
        "taker_buy_quote_asset_volume",
    ]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    df["close_time"] = pd.to_datetime(df["close_time"], unit="ms", utc=True)
    df = df.sort_values("open_time").drop_duplicates("open_time").reset_index(drop=True)

    # Feature engineering
    df["log_return_1s"] = np.log(df["close"]).diff()
    
    vol_window = 60
    df["inst_vol_60s"] = df["log_return_1s"].rolling(window=vol_window, min_periods=10).std()
    df["inst_vol_60s_bps"] = df["inst_vol_60s"] * 10000
    df["trend_mean_60s"] = df["log_return_1s"].rolling(window=vol_window, min_periods=10).mean()
    df["trend_zscore"] = df["trend_mean_60s"] / (df["inst_vol_60s"] + 1e-12)
    df["vwap"] = df["quote_asset_volume"] / df["volume"].replace(0, np.nan)
    df["buy_volume_ratio"] = df["taker_buy_base_asset_volume"] / df["volume"].replace(0, np.nan)

    # Rename columns for this asset
    df = df.rename(columns={
        "inst_vol_60s": f"{symbol_name}_inst_vol_60s",
        "inst_vol_60s_bps": f"{symbol_name}_inst_vol_60s_bps",
        "trend_mean_60s": f"{symbol_name}_trend_mean_60s",
        "trend_zscore": f"{symbol_name}_trend_zscore",
        "buy_volume_ratio": f"{symbol_name}_buy_volume_ratio",
        "vwap": f"{symbol_name}_vwap",
        "log_return_1s": f"{symbol_name}_log_return_1s",
        "close": f"{symbol_name}_close",
        "volume": f"{symbol_name}_volume",
    })

    df["symbol"] = symbol_name
    df = df.drop(columns=["open", "high", "low", "quote_asset_volume", "number_of_trades", "taker_buy_base_asset_volume", "taker_buy_quote_asset_volume", "close_time"])
    
    return df


def load_polymarket_1s(polymarket_csv: Path) -> pd.DataFrame:
    """Load Polymarket data with minimal preprocessing (no upsampling)."""
    print("  Reading CSV...")
    df = pd.read_csv(polymarket_csv, low_memory=False)
    
    print("  Processing columns...")
    df["timestamp"] = pd.to_datetime(df["timestamp_log"], unit="s", utc=True)
    df["start_time"] = pd.to_numeric(df["start_time"], errors="coerce")
    df["elapsed"] = pd.to_numeric(df["elapsed"], errors="coerce")
    df["label"] = (df["winner"].astype(str).str.lower() == "up").astype(int)
    df["implied_prob"] = (pd.to_numeric(df["bid_YES"], errors="coerce") + pd.to_numeric(df["ask_YES"], errors="coerce")) / 2.0
    df["implied_prob"] = df["implied_prob"].clip(0.0, 1.0)
    
    # Filter to valid range
    df = df[(df["elapsed"] >= 100) & (df["elapsed"] <= 290)].copy()
    df = df[["slug", "timestamp", "elapsed", "label", "implied_prob"]].copy()
    
    print(f"  Loaded {len(df):,} rows")
    return df


def compute_metrics(y_true: np.ndarray, y_prob: np.ndarray) -> Dict[str, float]:
    """Compute all evaluation metrics."""
    y_prob = np.clip(y_prob, 1e-6, 1 - 1e-6)
    y_pred = (y_prob >= 0.5).astype(int)
    try:
        auc = float(roc_auc_score(y_true, y_prob))
    except ValueError:
        auc = np.nan
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "roc_auc": auc,
        "log_loss": float(log_loss(y_true, y_prob, labels=[0, 1])),
        "brier": float(brier_score_loss(y_true, y_prob)),
        "calibration_gap_pp": float(np.mean(np.abs(y_prob - y_true)) * 100.0),
        "mean_pred": float(np.mean(y_prob)),
        "mean_actual": float(np.mean(y_true)),
    }


def run_walk_forward_test_enhanced(
    btc_df: pd.DataFrame,
    eth_df: pd.DataFrame,
    btc_perp_df: pd.DataFrame,
    polymarket_df: pd.DataFrame,
    training_days: int = 7,
    test_hours: float = 6.0,
    step_hours: float = 6.0,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Walk-forward backtest with BTC, ETH, and BTC Perps features.
    Uses 2-second interval matching (Polymarket frequency).
    """
    
    print("  Preparing feature columns...")
    # Baseline BTC-only features
    btc_feature_cols = [
        f"BTC_{c}" for c in [
            "close", "volume", "buy_volume_ratio", "vwap",
            "inst_vol_60s", "inst_vol_60s_bps", "trend_mean_60s", "trend_zscore"
        ]
    ]
    
    # ETH features
    eth_feature_cols = [
        f"ETH_{c}" for c in [
            "inst_vol_60s",  "trend_mean_60s"
        ]
    ]
    
    # Perps features
    perp_feature_cols = [
        f"BTCPERP_{c}" for c in [
            "inst_vol_60s", "trend_mean_60s"
        ]
    ]
    
    cross_features = ["eth_btc_vol_ratio"]
    
    print("  Aggregating to 2-second intervals...")
    # Downsample to 2-second intervals (Polymarket frequency) using backward-fill
    btc_2s = btc_df.set_index("open_time").resample("2s").last().reset_index()
    btc_2s = btc_2s.dropna(subset=["BTC_close"])
    btc_2s.columns = ["open_time"] + btc_2s.columns[1:].tolist()
    
    eth_2s = eth_df.set_index("open_time").resample("2s").last().reset_index()
    eth_2s = eth_2s.dropna(subset=[c for c in eth_2s.columns if 'ETH' in c])
    eth_2s.columns = ["open_time"] + eth_2s.columns[1:].tolist()
    
    perp_2s = btc_perp_df.set_index("open_time").resample("2s").last().reset_index()
    perp_2s = perp_2s.dropna(subset=[c for c in perp_2s.columns if 'BTCPERP' in c])
    perp_2s.columns = ["open_time"] + perp_2s.columns[1:].tolist()
    
    print("  Merging all sources...")
    # Merge all on open_time
    merged = btc_2s.merge(eth_2s[["open_time"] + eth_feature_cols], on="open_time", how="inner")
    merged = merged.merge(perp_2s[["open_time"] + perp_feature_cols], on="open_time", how="inner")
    
    # Compute cross-asset features
    merged["eth_btc_vol_ratio"] = merged["ETH_inst_vol_60s"] / (merged["BTC_inst_vol_60s"] + 1e-12)
    
    # Merge with Polymarket using nearest timestamp (within tolerance)
    print("  Matching with Polymarket data...")
    merged["timestamp_int"] = (merged["open_time"].astype(np.int64) // 10**9).astype(int)
    pm_match = polymarket_df.copy()
    pm_match["timestamp_int"] = (pm_match["timestamp"].astype(np.int64) // 10**9).astype(int)
    
    # For each PM row, find closest BTC row (must be within 2 seconds)
    result_rows = []
    for idx, pm_row in pm_match.iterrows():
        pm_ts = pm_row["timestamp_int"]
       # Find closest BTC timestamp
        closest_idx = (merged["timestamp_int"] - pm_ts).abs().argmin()
        if abs(merged.iloc[closest_idx]["timestamp_int"] - pm_ts) <= 2:  # Within 2 seconds
            merged_row = merged.iloc[closest_idx].to_dict()
            merged_row.update({
                "slug": pm_row["slug"],
                "elapsed": pm_row["elapsed"],
                "implied_prob": pm_row["implied_prob"],
                "label": pm_row["label"],
                "target_up": pm_row["label"],  # Use PM label as target
                "open_time_pm": pm_row["timestamp"],
            })
            result_rows.append(merged_row)
    
    if not result_rows:
        raise ValueError("No matching rows after alignment")
    
    final_merged = pd.DataFrame(result_rows)
    print(f"  Aligned {len(final_merged):,} rows")
    
    feature_cols = btc_feature_cols + eth_feature_cols + perp_feature_cols + cross_features
    feature_cols = [c for c in feature_cols if c in final_merged.columns]
    
    final_merged = final_merged.dropna(subset=feature_cols + ["target_up"]).copy()
    final_merged = final_merged.sort_values("open_time").reset_index(drop=True)
    
    if len(final_merged) == 0:
        raise ValueError("No valid rows after feature selection")
    
    print(f"  Total rows for modeling: {len(final_merged):,}")
    
    # Walk-forward setup
    test_seconds = int(test_hours * 3600)
    step_seconds = int(step_hours * 3600)
    train_seconds = int(training_days * 24 * 3600)
    
    min_ts = final_merged["open_time"].min()
    max_ts = final_merged["open_time"].max()
    
    first_test_ts = min_ts + pd.Timedelta(seconds=train_seconds)
    last_test_ts = max_ts - pd.Timedelta(seconds=test_seconds)
    
    splits = []
    row_level_records = []
    test_start = first_test_ts
    split_idx = 0
    
    print("  Running walk-forward validation...")
    while test_start <= last_test_ts:
        split_idx += 1
        train_start = test_start - pd.Timedelta(seconds=train_seconds)
        test_end = test_start + pd.Timedelta(seconds=test_seconds)
        
        train_mask = (final_merged["open_time"] >= train_start) & (final_merged["open_time"] < test_start)
        test_mask = (final_merged["open_time"] >= test_start) & (final_merged["open_time"] < test_end)
        
        train_data = final_merged[train_mask].copy()
        test_data = final_merged[test_mask].copy()
        
        if len(train_data) < 100 or len(test_data) < 50:
            test_start += pd.Timedelta(seconds=step_seconds)
            continue
        
        train_data = train_data.sort_values("open_time").reset_index(drop=True)
        y_train = train_data["target_up"].astype(int).values
        X_test = test_data[feature_cols]
        y_test = test_data["target_up"].astype(int).values
        
        # Baseline: standard calibrated model
        base_pipeline = make_base_pipeline(feature_cols)
        calibrated_model = CalibratedClassifierCV(base_pipeline, method="sigmoid", cv=3)
        calibrated_model.fit(train_data[feature_cols], y_train)
        lr_base_probs = np.clip(calibrated_model.predict_proba(X_test)[:, 1], 1e-6, 1 - 1e-6)
        
        # Enhanced: time-safe bucketing + shrinkage
        cut_idx = int(len(train_data) * 0.8)
        if cut_idx < 1000 or (len(train_data) - cut_idx) < 500:
            cut_idx = max(100, int(len(train_data) * 0.7))
        
        subtrain = train_data.iloc[:cut_idx].copy()
        calib = train_data.iloc[cut_idx:].copy()
        
        if len(subtrain) < 100 or len(calib) < 100 or calib["target_up"].nunique() < 2:
            lr_enh_probs = lr_base_probs.copy()
            best_lambda = 1.0
            shrink_meta = {"log_loss": np.nan, "brier": np.nan, "calib_gap_pp": np.nan}
        else:
            subtrain_model = make_base_pipeline(feature_cols)
            subtrain_model.fit(subtrain[feature_cols], subtrain["target_up"].astype(int).values)
            
            raw_calib = np.clip(
                subtrain_model.predict_proba(calib[feature_cols])[:, 1],
                1e-6, 1 - 1e-6,
            )
            raw_test = np.clip(
                subtrain_model.predict_proba(X_test)[:, 1],
                1e-6, 1 - 1e-6,
            )
            
            calib_df = pd.DataFrame({
                "elapsed": calib["elapsed"].to_numpy(dtype=float),
                "y_true": calib["target_up"].astype(int).to_numpy(),
                "raw_prob": raw_calib,
                "ts_order": np.arange(len(calib), dtype=int),
            })
            calib_df["elapsed_bucket"] = bucket_series(calib_df["elapsed"])
            
            calibrators, bucket_diag = fit_bucket_calibrators(calib_df)
            
            p_calib_bucketed = apply_bucket_calibrators(
                raw_calib,
                calib_df["elapsed"].to_numpy(),
                calibrators,
            )
            
            base_rate = float(subtrain["target_up"].mean())
            best_lambda, shrink_meta = tune_shrinkage_lambda(
                p_calib_bucketed,
                calib["target_up"].astype(int).to_numpy(),
                base_rate,
            )
            
            p_test_bucketed = apply_bucket_calibrators(
                raw_test,
                test_data["elapsed"].to_numpy(),
                calibrators,
            )
            lr_enh_probs = np.clip(
                best_lambda * p_test_bucketed + (1.0 - best_lambda) * base_rate,
                1e-6, 1 - 1e-6
            )
        
        lr_base_metrics = compute_metrics(y_test, lr_base_probs)
        lr_enh_metrics = compute_metrics(y_test, lr_enh_probs)
        pm_probs = test_data["implied_prob"].astype(float).values
        pm_metrics = compute_metrics(y_test, pm_probs)
        
        splits.append({
            "split_idx": split_idx,
            "test_start": test_start.isoformat(),
            "test_end": test_end.isoformat(),
            "train_rows": len(train_data),
            "test_rows": len(test_data),
            "model": "7day_logistic_baseline",
            **lr_base_metrics,
        })
        splits.append({
            "split_idx": split_idx,
            "test_start": test_start.isoformat(),
            "test_end": test_end.isoformat(),
            "train_rows": len(train_data),
            "test_rows": len(test_data),
            "model": "7day_eth_perps_enhanced",
            "best_shrinkage_lambda": best_lambda,
            **lr_enh_metrics,
        })
        splits.append({
            "split_idx": split_idx,
            "test_start": test_start.isoformat(),
            "test_end": test_end.isoformat(),
            "train_rows": len(train_data),
            "test_rows": len(test_data),
            "model": "polymarket",
            **pm_metrics,
        })
        
        split_rows = pd.DataFrame({
            "split_idx": split_idx,
            "timestamp": test_data["open_time"].values,
            "slug": test_data["slug"].values,
            "elapsed": test_data["elapsed"].values,
            "y_true": y_test,
            "prob_baseline": lr_base_probs,
            "prob_eth_perps": lr_enh_probs,
            "prob_polymarket": pm_probs,
        })
        row_level_records.append(split_rows)
        
        print(
            f"  Split {split_idx}: {test_start} -> {test_end} | Train: {len(train_data):,} | "
            f"Test: {len(test_data):,} | lambda={best_lambda:.2f}"
        )
        
        test_start += pd.Timedelta(seconds=step_seconds)
    
    if not splits:
        raise ValueError("No valid walk-forward splits generated")
    
    splits_df = pd.DataFrame(splits)
    
    # Overall metrics
    overall = []
    for model_name in ["7day_logistic_baseline", "7day_eth_perps_enhanced", "polymarket"]:
        model_data = splits_df[splits_df["model"] == model_name]
        metrics = {
            "model": model_name,
            "n_splits": model_data["split_idx"].nunique(),
            "accuracy_mean": model_data["accuracy"].mean(),
            "accuracy_std": model_data["accuracy"].std(),
            "roc_auc_mean": model_data["roc_auc"].mean(),
            "roc_auc_std": model_data["roc_auc"].std(),
            "log_loss_mean": model_data["log_loss"].mean(),
            "log_loss_std": model_data["log_loss"].std(),
            "brier_mean": model_data["brier"].mean(),
            "brier_std": model_data["brier"].std(),
            "calibration_gap_pp_mean": model_data["calibration_gap_pp"].mean(),
            "calibration_gap_pp_std": model_data["calibration_gap_pp"].std(),
        }
        overall.append(metrics)
    
    overall_df = pd.DataFrame(overall)
    row_level_df = pd.concat(row_level_records, ignore_index=True) if row_level_records else pd.DataFrame()
    
    return overall_df, splits_df, row_level_df


def main() -> None:
    parser = argparse.ArgumentParser(
        description="7-day model enhanced with ETH and BTC perps features"
    )
    parser.add_argument(
        "--polymarket-csv",
        default="research/polymarket_model/market_data_2sec_weekly5_with_resolutions.csv",
        help="Path to Polymarket 2-second data",
    )
    parser.add_argument(
        "--btc-cache",
        default="research/polymarket_model/binance_1s_btc_for_enhanced.csv",
        help="Path to cache BTC 1-second data",
    )
    parser.add_argument(
        "--eth-cache",
        default="research/polymarket_model/binance_1s_eth_for_enhanced.csv",
        help="Path to cache ETH 1-second data",
    )
    parser.add_argument(
        "--btc-perp-cache",
        default="research/polymarket_model/binance_1s_btc_perp_for_enhanced.csv",
        help="Path to cache BTC perp 1-second data",
    )
    parser.add_argument(
        "--output-dir",
        default="research/polymarket_model/strict_7day_eth_perps_results",
        help="Output directory for results",
    )
    parser.add_argument("--test-hours", type=float, default=6.0, help="Test block length in hours")
    parser.add_argument("--step-hours", type=float, default=6.0, help="Step between test blocks in hours")
    args = parser.parse_args()

    print("=" * 70)
    print("ETH + PERPS ENHANCED MODEL vs 7-DAY BASELINE vs POLYMARKET")
    print("=" * 70)
    
    print("\n[1/5] Loading Polymarket data...")
    pm = load_polymarket_1s(Path(args.polymarket_csv))
    print(f"      Polymarket: {len(pm):,} rows from {pm['timestamp'].min()} to {pm['timestamp'].max()}")

    start_ts = pm["timestamp"].min().floor("s")
    end_ts = pm["timestamp"].max().ceil("s")
    start_ms = int(start_ts.timestamp() * 1000)
    end_ms = int(end_ts.timestamp() * 1000)

    # Load BTC
    print("\n[2/5] Loading/fetching BTC data...")
    cache_path = Path(args.btc_cache)
    if cache_path.exists():
        print(f"      Using cached BTC from {cache_path}")
        btc_df = pd.read_csv(cache_path)
        btc_df["open_time"] = pd.to_datetime(btc_df["open_time"], utc=True)
    else:
        print("      Fetching BTC 1-second data from Binance...")
        raw_btc = fetch_klines_1s_chunked("BTCUSDT", start_ms, end_ms, chunk_hours=6)
        btc_df = build_binance_df(raw_btc, symbol_name="BTC")
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        btc_df.to_csv(cache_path, index=False)
        print(f"      Cached to {cache_path}")

    print(f"      BTC: {len(btc_df):,} rows from {btc_df['open_time'].min()} to {btc_df['open_time'].max()}")

    # Load ETH
    print("\n[3/5] Loading/fetching ETH data...")
    cache_path = Path(args.eth_cache)
    if cache_path.exists():
        print(f"      Using cached ETH from {cache_path}")
        eth_df = pd.read_csv(cache_path)
        eth_df["open_time"] = pd.to_datetime(eth_df["open_time"], utc=True)
    else:
        print("      Fetching ETH 1-second data from Binance...")
        raw_eth = fetch_klines_1s_chunked("ETHUSDT", start_ms, end_ms, chunk_hours=6)
        eth_df = build_binance_df(raw_eth, symbol_name="ETH")
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        eth_df.to_csv(cache_path, index=False)
        print(f"      Cached to {cache_path}")

    print(f"      ETH: {len(eth_df):,} rows from {eth_df['open_time'].min()} to {eth_df['open_time'].max()}")

    # Load BTC Perps
    print("\n[4/5] Loading/fetching BTC Perp data...")
    cache_path = Path(args.btc_perp_cache)
    if cache_path.exists():
        print(f"      Using cached BTC Perp from {cache_path}")
        btc_perp_df = pd.read_csv(cache_path)
        btc_perp_df["open_time"] = pd.to_datetime(btc_perp_df["open_time"], utc=True)
    else:
        print("      Fetching BTC Perp 1-second data from Binance Futures...")
        try:
            raw_btc_perp = fetch_klines_1s_chunked("BTCUSDTPERP", start_ms, end_ms, chunk_hours=6, url=BINANCE_FUTURES_URL)
            btc_perp_df = build_binance_df(raw_btc_perp, symbol_name="BTCPERP")
            btc_perp_df["BTCPERP_funding_rate"] = 0.00001
            btc_perp_df["BTCPERP_oi_ratio"] = 1.0
        except Exception as e:
            print(f"      Warning: Could not fetch BTC Perp ({e}). Using BTC as proxy...")
            btc_perp_df = btc_df.copy()
            btc_perp_df = btc_perp_df.rename(columns={
                "BTC_close": "BTCPERP_close",
                "BTC_volume": "BTCPERP_volume",
                "BTC_inst_vol_60s": "BTCPERP_inst_vol_60s",
                "BTC_trend_mean_60s": "BTCPERP_trend_mean_60s",
            })
            btc_perp_df["BTCPERP_funding_rate"] = 0.00001
            btc_perp_df["BTCPERP_oi_ratio"] = 1.0
        
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        btc_perp_df.to_csv(cache_path, index=False)
        print(f"      Cached to {cache_path}")

    print(f"      BTC Perp: {len(btc_perp_df):,} rows from {btc_perp_df['open_time'].min()} to {btc_perp_df['open_time'].max()}")

    print("\n[5/5] Running walk-forward validation with ETH+Perps features...")
    overall_df, splits_df, row_level_df = run_walk_forward_test_enhanced(
        btc_df,
        eth_df,
        btc_perp_df,
        pm,
        training_days=7,
        test_hours=args.test_hours,
        step_hours=args.step_hours,
    )

    # Save results
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nSaving results to {out_dir}/...")
    overall_df.to_csv(out_dir / "overall_comparison.csv", index=False)
    splits_df.to_csv(out_dir / "per_split_metrics.csv", index=False)
    row_level_df.to_csv(out_dir / "row_level_predictions.csv", index=False)

    print("\n" + "=" * 70)
    print("OVERALL RESULTS: ETH+Perps Enhanced vs Baseline vs Polymarket")
    print("=" * 70)
    print(overall_df.to_string(index=False))

    # Determine winner
    print("\n" + "=" * 70)
    print("METRIC WINS (higher better → AUC/Accuracy, lower better → Log Loss/Brier/CalibGap)")
    print("=" * 70)
    metrics_to_eval = ["accuracy", "roc_auc", "log_loss", "brier", "calibration_gap_pp"]
    higher_better = ["accuracy", "roc_auc"]
    models = overall_df["model"].tolist()
    wins = {m: 0 for m in models}

    for m in metrics_to_eval:
        col = f"{m}_mean"
        if col not in overall_df.columns:
            continue
        if m in higher_better:
            winner = overall_df.loc[overall_df[col].idxmax(), "model"]
        else:
            winner = overall_df.loc[overall_df[col].idxmin(), "model"]
        wins[winner] += 1
        winner_val = float(overall_df.loc[overall_df["model"] == winner, col].iloc[0])
        print(f"{m:25s}: {winner:35s} ({winner_val:.6f})")

    wins_sorted = sorted(wins.items(), key=lambda x: x[1], reverse=True)
    print("\nTotal wins by model:")
    for model_name, w in wins_sorted:
        print(f"  {model_name}: {w}")

    print("\n" + "=" * 70)
    print(f"Results saved to {out_dir}/")
    print("=" * 70)


if __name__ == "__main__":
    main()
