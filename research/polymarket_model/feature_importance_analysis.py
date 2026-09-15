"""Feature importance analysis for 7-day baseline model.

This script:
1. Ranks all 13 baseline features by importance using permutation importance
2. Builds a reduced model using only important features (>threshold)
3. Walk-forward validates baseline, reduced, and Polymarket models
4. Saves metrics to CSV for comparison
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
from sklearn.inspection import permutation_importance
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

BINANCE_KLINES_URL = "https://api.binance.com/api/v3/klines"
ELAPSED_BINS = [100, 130, 160, 190, 220, 250, 291]
ELAPSED_LABELS = ["100-129", "130-159", "160-189", "190-219", "220-249", "250-290"]

ALL_FEATURES = [
    "close",
    "volume",
    "number_of_trades",
    "log_return_1s",
    "inst_vol_60s",
    "inst_vol_60s_bps",
    "trend_mean_60s",
    "trend_zscore",
    "buy_volume_ratio",
    "vwap",
    "return_from_candle_open",
    "seconds_to_5m_close",
    "fraction_of_candle_elapsed",
]


def fetch_klines_1s_chunked(symbol: str, start_ms: int, end_ms: int, chunk_hours: int = 6) -> List[List]:
    """Fetch 1-second klines from Binance in chunks to avoid timeout."""
    chunk_ms = chunk_hours * 3600 * 1000
    all_rows = []
    current_ms = start_ms

    while current_ms < end_ms:
        chunk_end_ms = min(current_ms + chunk_ms, end_ms)
        params = {
            "symbol": symbol,
            "interval": "1s",
            "startTime": current_ms,
            "endTime": chunk_end_ms,
            "limit": 1000,
        }
        print(f"  Fetching {symbol} {current_ms} to {chunk_end_ms}...")

        try:
            response = requests.get(BINANCE_KLINES_URL, params=params, timeout=10)
            response.raise_for_status()
            rows = response.json()
            if not rows:
                break
            all_rows.extend(rows)
        except requests.RequestException as e:
            print(f"    Warning: Failed to fetch chunk: {e}")
            break

        current_ms = chunk_end_ms

    return all_rows


def build_binance_df(raw_rows: List[List]) -> pd.DataFrame:
    """Build BTC DataFrame with feature engineering from raw Binance data."""
    columns = [
        "open_time", "open", "high", "low", "close", "volume", "close_time",
        "quote_asset_volume", "number_of_trades", "taker_buy_base_asset_volume",
        "taker_buy_quote_asset_volume", "ignore",
    ]
    df = pd.DataFrame(raw_rows, columns=columns).drop(columns=["ignore"])

    numeric_cols = [
        "open", "high", "low", "close", "volume", "quote_asset_volume",
        "number_of_trades", "taker_buy_base_asset_volume", "taker_buy_quote_asset_volume",
    ]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    df["close_time"] = pd.to_datetime(df["close_time"], unit="ms", utc=True)
    df = df.sort_values("open_time").drop_duplicates("open_time").reset_index(drop=True)

    # Feature engineering matching original model
    df["log_return_1s"] = np.log(df["close"]).diff()
    vol_window = 60
    df["inst_vol_60s"] = df["log_return_1s"].rolling(window=vol_window, min_periods=10).std()
    df["inst_vol_60s_bps"] = df["inst_vol_60s"] * 10000
    df["trend_mean_60s"] = df["log_return_1s"].rolling(window=vol_window, min_periods=10).mean()
    df["trend_zscore"] = df["trend_mean_60s"] / (df["inst_vol_60s"] + 1e-12)
    df["vwap"] = df["quote_asset_volume"] / df["volume"].replace(0, np.nan)
    df["buy_volume_ratio"] = df["taker_buy_base_asset_volume"] / df["volume"].replace(0, np.nan)

    # 5-minute candle labels (target_up computed from actual price movement)
    df["candle_5m_start"] = df["open_time"].dt.floor("5min")
    candle = (
        df.groupby("candle_5m_start", as_index=False)
        .agg(candle_open=("open", "first"), candle_close=("close", "last"))
        .copy()
    )
    candle["target_up"] = (candle["candle_close"] > candle["candle_open"]).astype(int)
    df = df.merge(candle[["candle_5m_start", "target_up"]], on="candle_5m_start", how="left")

    # Temporal features within candle
    candle_end = df["candle_5m_start"] + pd.Timedelta(minutes=5)
    df["seconds_to_5m_close"] = (candle_end - df["open_time"]).dt.total_seconds().clip(lower=0)
    df["fraction_of_candle_elapsed"] = 1.0 - (df["seconds_to_5m_close"] / 300.0)
    df["return_from_candle_open"] = (
        df["close"] / df.groupby("candle_5m_start")["open"].transform("first")
    ) - 1.0

    return df


def build_btc_target(btc_df: pd.DataFrame) -> pd.DataFrame:
    """Compute 5-minute candle targets from actual BTC price movement."""
    btc_df = btc_df.copy()
    btc_df["candle_5m_start"] = btc_df["open_time"].dt.floor("5min")
    candle = (
        btc_df.groupby("candle_5m_start", as_index=False)
        .agg(candle_open=("open", "first"), candle_close=("close", "last"))
        .copy()
    )
    candle["target_up"] = (candle["candle_close"] > candle["candle_open"]).astype(int)
    btc_df = btc_df.merge(candle[["candle_5m_start", "target_up"]], on="candle_5m_start", how="left")
    return btc_df


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


def fetch_klines_1s(symbol: str, start_ms: int, end_ms: int) -> List[List]:
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
                response = session.get(BINANCE_KLINES_URL, params=params, timeout=30)
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


def fetch_klines_1s_chunked(symbol: str, start_ms: int, end_ms: int, chunk_hours: int = 6) -> List[List]:
    """Fetch klines in chunks to avoid API limits."""
    chunk_ms = int(chunk_hours * 3600 * 1000)
    cursor = start_ms
    merged: List[List] = []
    chunk_idx = 0

    while cursor < end_ms:
        chunk_idx += 1
        c_end = min(end_ms, cursor + chunk_ms)
        print(f"Fetching chunk {chunk_idx}: {pd.to_datetime(cursor, unit='ms', utc=True)} -> {pd.to_datetime(c_end, unit='ms', utc=True)}")
        rows = fetch_klines_1s(symbol=symbol, start_ms=cursor, end_ms=c_end)
        merged.extend(rows)
        cursor = c_end + 1000

    return merged


def build_binance_df(raw_rows: List[List], symbol_name: str = "BTC") -> pd.DataFrame:
    """Build dataframe from raw Binance klines."""
    columns = [
        "open_time", "open", "high", "low", "close", "volume",
        "close_time", "quote_asset_volume", "number_of_trades",
        "taker_buy_base_asset_volume", "taker_buy_quote_asset_volume", "unused",
    ]

    df = pd.DataFrame(raw_rows, columns=columns)
    df["open_time"] = pd.to_datetime(df["open_time"].astype(int), unit="ms", utc=True)
    df["close_time"] = pd.to_datetime(df["close_time"].astype(int), unit="ms", utc=True)

    for col in ["open", "high", "low", "close", "volume", "quote_asset_volume", "number_of_trades", "taker_buy_base_asset_volume", "taker_buy_quote_asset_volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["close", "volume"]).copy()

    df["log_return_1s"] = np.log(df["close"] / df["close"].shift(1))
    df["inst_vol_60s"] = df["log_return_1s"].rolling(60, min_periods=1).std()
    df["inst_vol_60s_bps"] = df["inst_vol_60s"] * 10000
    df["trend_mean_60s"] = df["log_return_1s"].rolling(60, min_periods=1).mean()
    df["trend_zscore"] = (df["log_return_1s"] - df["trend_mean_60s"]) / (df["inst_vol_60s"] + 1e-8)
    df["buy_volume_ratio"] = df["taker_buy_base_asset_volume"] / (df["volume"] + 1e-8)
    df["vwap"] = (df["close"] * df["volume"]).rolling(60, min_periods=1).sum() / (df["volume"].rolling(60, min_periods=1).sum() + 1e-8)
    df["return_from_candle_open"] = (df["close"] - df["open"]) / (df["open"] + 1e-8)
    df["seconds_to_5m_close"] = (300 - (df["open_time"].astype(np.int64) // 10**9 % 300)).astype(float)
    df["fraction_of_candle_elapsed"] = 1.0 - (df["seconds_to_5m_close"] / 300.0)

    return df


def load_polymarket_csv(pm_path: str) -> pd.DataFrame:
    """Load Polymarket data from CSV with upsampling to 1-second."""
    df = pd.read_csv(pm_path)
    df["timestamp"] = pd.to_datetime(df["timestamp_log"], unit="s", utc=True)
    df["start_time"] = pd.to_numeric(df["start_time"], errors="coerce")
    df["elapsed"] = pd.to_numeric(df["elapsed"], errors="coerce")
    df["label"] = (df["winner"].astype(str).str.lower() == "up").astype(int)
    df["implied_prob"] = (pd.to_numeric(df["bid_YES"], errors="coerce") + pd.to_numeric(df["ask_YES"], errors="coerce")) / 2.0
    df["implied_prob"] = df["implied_prob"].clip(0.0, 1.0)

    # Upsample 2s -> 1s by carry-forward within each market slug.
    upsampled = []
    for slug, g in df.groupby("slug", sort=False):
        g = g.sort_values("timestamp").copy()
        idx = pd.date_range(start=g["timestamp"].min(), end=g["timestamp"].max(), freq="1s", tz="UTC")
        u = g.set_index("timestamp").reindex(idx)
        for col in ["slug", "start_time", "label", "implied_prob"]:
            u[col] = u[col].ffill().bfill()
        u = u.reset_index().rename(columns={"index": "timestamp"})
        u["elapsed"] = (u["timestamp"].astype("int64") // 10**9 - u["start_time"]).astype(int)
        upsampled.append(u[["slug", "timestamp", "elapsed", "label", "implied_prob"]])

    pm = pd.concat(upsampled, ignore_index=True)
    pm = pm[(pm["elapsed"] >= 100) & (pm["elapsed"] <= 290)].copy()
    
    return pm


def rank_features_by_importance(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    feature_names: List[str],
    feature_cols: List[str],
) -> pd.DataFrame:
    """Rank features by permutation importance on a trained baseline model."""
    print("\n  Computing permutation importance on training split...")

    # Train a simple model to evaluate importance
    pipe = make_base_pipeline(feature_cols)
    pipe.fit(X_train, y_train)

    # Compute permutation importance
    result = permutation_importance(
        pipe, X_train, y_train,
        n_repeats=10,
        random_state=42,
        n_jobs=-1,
        scoring='neg_log_loss'
    )

    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance_mean': result.importances_mean,
        'importance_std': result.importances_std,
    }).sort_values('importance_mean', ascending=False)

    print("  Feature Importance Ranking:")
    for idx, row in importance_df.iterrows():
        print(f"    {row['feature']:35s} {row['importance_mean']:8.4f} ± {row['importance_std']:6.4f}")

    return importance_df


def run_walk_forward_with_feature_sets(
    btc_df: pd.DataFrame,
    pm: pd.DataFrame,
    all_feature_cols: List[str],
    selected_feature_cols: List[str],
    training_days: int = 7,
    test_hours: int = 6,
    step_hours: int = 6,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Walk-forward validation comparing baseline (all features) vs reduced (selected features) vs Polymarket."""

    # Align timestamps
    btc_df = btc_df.sort_values("open_time").copy()
    pm = pm.sort_values("timestamp").copy()

    min_ts = max(btc_df["open_time"].min(), pm["timestamp"].min())
    max_ts = min(btc_df["open_time"].max(), pm["timestamp"].max())

    btc_df = btc_df[(btc_df["open_time"] >= min_ts) & (btc_df["open_time"] <= max_ts)].copy()
    pm = pm[(pm["timestamp"] >= min_ts) & (pm["timestamp"] <= max_ts)].copy()

    # Merge: BTC features + target from BTC + Polymarket data (implied_prob)
    merged = btc_df.merge(
        pm[["timestamp", "slug", "elapsed", "implied_prob"]],
        left_on="open_time",
        right_on="timestamp",
        how="inner",
    )
    merged = merged.dropna(subset=all_feature_cols + ["target_up"]).copy()

    if len(merged) == 0:
        raise ValueError("No merged rows after alignment")

    # Walk-forward setup
    test_seconds = int(test_hours * 3600)
    step_seconds = int(step_hours * 3600)
    training_seconds = int(training_days * 24 * 3600)

    merged["ts_order"] = (merged["timestamp"] - merged["timestamp"].min()).dt.total_seconds()
    max_ts_order = merged["ts_order"].max()

    results = []
    splits_data = []
    split_idx = 0

    start_time = merged["timestamp"].min() + pd.Timedelta(seconds=training_seconds)

    while start_time < merged["timestamp"].max() - pd.Timedelta(seconds=test_seconds):
        split_idx += 1
        end_time = start_time + pd.Timedelta(seconds=test_seconds)

        train_start = start_time - pd.Timedelta(seconds=training_seconds)
        train_data = merged[(merged["timestamp"] >= train_start) & (merged["timestamp"] < start_time)]
        test_data = merged[(merged["timestamp"] >= start_time) & (merged["timestamp"] < end_time)]

        if len(test_data) < 100:
            start_time += pd.Timedelta(seconds=step_seconds)
            continue

        y_test = test_data["target_up"].astype(int).values
        base_rate = float(y_test.mean())

        # === Model 1: Baseline (all features) ===
        if len(train_data) >= 500:
            X_train_baseline = train_data[all_feature_cols]
            pipe_baseline = make_base_pipeline(all_feature_cols)
            pipe_baseline.fit(X_train_baseline, train_data["target_up"].astype(int).values)

            X_test_baseline = test_data[all_feature_cols]
            raw_prob_baseline = pipe_baseline.predict_proba(X_test_baseline)[:, 1]

            # Calibration + shrinkage
            subtrain_cut = int(len(train_data) * 0.8)
            subtrain = train_data.iloc[:subtrain_cut]
            calib_data_raw = train_data.iloc[subtrain_cut:]

            pipe_subtrain = make_base_pipeline(all_feature_cols)
            pipe_subtrain.fit(subtrain[all_feature_cols], subtrain["target_up"].astype(int).values)
            p_calib_raw = pipe_subtrain.predict_proba(calib_data_raw[all_feature_cols])[:, 1]

            calib_df = pd.DataFrame({
                "raw_prob": p_calib_raw,
                "y_true": calib_data_raw["target_up"].astype(int).values,
                "elapsed_bucket": bucket_series(calib_data_raw["elapsed"].values),
                "ts_order": calib_data_raw["ts_order"].values,
            })

            calibrators, _ = fit_bucket_calibrators(calib_df, min_bucket_samples=50)
            p_calib = apply_bucket_calibrators(p_calib_raw, calib_data_raw["elapsed"].values, calibrators)
            lambda_b, _ = tune_shrinkage_lambda(p_calib, calib_data_raw["target_up"].astype(int).values, base_rate)

            p_final_baseline = apply_bucket_calibrators(raw_prob_baseline, test_data["elapsed"].values, calibrators)
            p_final_baseline = lambda_b * p_final_baseline + (1 - lambda_b) * base_rate
            p_final_baseline = np.clip(p_final_baseline, 1e-6, 1 - 1e-6)

            acc_baseline = float(accuracy_score(y_test, (p_final_baseline > 0.5).astype(int)))
            auc_baseline = float(roc_auc_score(y_test, p_final_baseline))
            ll_baseline = safe_log_loss(y_test, p_final_baseline)
            brier_baseline = float(brier_score_loss(y_test, p_final_baseline))
            gap_baseline = float(np.mean(np.abs(p_final_baseline - y_test)) * 100.0)
        else:
            acc_baseline, auc_baseline, ll_baseline, brier_baseline, gap_baseline = np.nan, np.nan, np.nan, np.nan, np.nan

        # === Model 2: Reduced (selected features) ===
        if len(train_data) >= 500 and selected_feature_cols:
            X_train_reduced = train_data[selected_feature_cols]
            pipe_reduced = make_base_pipeline(selected_feature_cols)
            pipe_reduced.fit(X_train_reduced, train_data["target_up"].astype(int).values)

            X_test_reduced = test_data[selected_feature_cols]
            raw_prob_reduced = pipe_reduced.predict_proba(X_test_reduced)[:, 1]

            # Calibration + shrinkage
            pipe_subtrain_r = make_base_pipeline(selected_feature_cols)
            pipe_subtrain_r.fit(subtrain[selected_feature_cols], subtrain["target_up"].astype(int).values)
            p_calib_raw_r = pipe_subtrain_r.predict_proba(calib_data_raw[selected_feature_cols])[:, 1]

            calib_df_r = pd.DataFrame({
                "raw_prob": p_calib_raw_r,
                "y_true": calib_data_raw["target_up"].astype(int).values,
                "elapsed_bucket": bucket_series(calib_data_raw["elapsed"].values),
                "ts_order": calib_data_raw["ts_order"].values,
            })

            calibrators_r, _ = fit_bucket_calibrators(calib_df_r, min_bucket_samples=50)
            p_calib_r = apply_bucket_calibrators(p_calib_raw_r, calib_data_raw["elapsed"].values, calibrators_r)
            lambda_r, _ = tune_shrinkage_lambda(p_calib_r, calib_data_raw["target_up"].astype(int).values, base_rate)

            p_final_reduced = apply_bucket_calibrators(raw_prob_reduced, test_data["elapsed"].values, calibrators_r)
            p_final_reduced = lambda_r * p_final_reduced + (1 - lambda_r) * base_rate
            p_final_reduced = np.clip(p_final_reduced, 1e-6, 1 - 1e-6)

            acc_reduced = float(accuracy_score(y_test, (p_final_reduced > 0.5).astype(int)))
            auc_reduced = float(roc_auc_score(y_test, p_final_reduced))
            ll_reduced = safe_log_loss(y_test, p_final_reduced)
            brier_reduced = float(brier_score_loss(y_test, p_final_reduced))
            gap_reduced = float(np.mean(np.abs(p_final_reduced - y_test)) * 100.0)
        else:
            acc_reduced, auc_reduced, ll_reduced, brier_reduced, gap_reduced = np.nan, np.nan, np.nan, np.nan, np.nan

        # === Model 3: Polymarket ===
        p_polymarket = test_data["implied_prob"].values
        p_polymarket = np.clip(p_polymarket, 1e-6, 1 - 1e-6)
        acc_polymarket = float(accuracy_score(y_test, (p_polymarket > 0.5).astype(int)))
        auc_polymarket = float(roc_auc_score(y_test, p_polymarket))
        ll_polymarket = safe_log_loss(y_test, p_polymarket)
        brier_polymarket = float(brier_score_loss(y_test, p_polymarket))
        gap_polymarket = float(np.mean(np.abs(p_polymarket - y_test)) * 100.0)

        splits_data.append({
            'split': split_idx,
            'train_start': train_start.isoformat(),
            'test_start': start_time.isoformat(),
            'test_end': end_time.isoformat(),
            'n_train': len(train_data),
            'n_test': len(test_data),
            'baseline_accuracy': acc_baseline,
            'baseline_auc': auc_baseline,
            'baseline_log_loss': ll_baseline,
            'baseline_brier': brier_baseline,
            'baseline_calib_gap_pp': gap_baseline,
            'reduced_accuracy': acc_reduced,
            'reduced_auc': auc_reduced,
            'reduced_log_loss': ll_reduced,
            'reduced_brier': brier_reduced,
            'reduced_calib_gap_pp': gap_reduced,
            'polymarket_accuracy': acc_polymarket,
            'polymarket_auc': auc_polymarket,
            'polymarket_log_loss': ll_polymarket,
            'polymarket_brier': brier_polymarket,
            'polymarket_calib_gap_pp': gap_polymarket,
        })

        start_time += pd.Timedelta(seconds=step_seconds)

    splits_df = pd.DataFrame(splits_data)

    # Aggregate metrics
    models = ['baseline', 'reduced', 'polymarket']
    overall = []
    for model in models:
        for metric in ['accuracy', 'auc', 'log_loss', 'brier', 'calib_gap_pp']:
            col = f'{model}_{metric}'
            mean_val = splits_df[col].mean()
            std_val = splits_df[col].std()
            overall.append({
                'model': model,
                f'{metric}_mean': mean_val,
                f'{metric}_std': std_val,
            })

    # Merge overall results
    overall_df = pd.DataFrame(overall).groupby('model').agg({
        'accuracy_mean': 'first', 'accuracy_std': 'first',
        'auc_mean': 'first', 'auc_std': 'first',
        'log_loss_mean': 'first', 'log_loss_std': 'first',
        'brier_mean': 'first', 'brier_std': 'first',
        'calib_gap_pp_mean': 'first', 'calib_gap_pp_std': 'first',
    }).reset_index()

    # Fill in any missing columns
    for metric in ['accuracy', 'auc', 'log_loss', 'brier', 'calib_gap_pp']:
        for suffix in ['_mean', '_std']:
            col = f'{metric}{suffix}'
            if col not in overall_df.columns:
                overall_df[col] = np.nan

    return overall_df, splits_df


def main():
    parser = argparse.ArgumentParser(description="Feature importance analysis and model comparison")
    parser.add_argument("--pm-csv", default="research/polymarket_model/market_data_2sec_weekly5_with_resolutions.csv", help="Polymarket CSV path")
    parser.add_argument("--btc-cache", default="research/polymarket_model/binance_1s_btc_for_feature_analysis.csv", help="BTC cache path")
    parser.add_argument("--output-dir", default="research/polymarket_model/feature_importance_results", help="Output directory")
    parser.add_argument("--test-hours", type=int, default=6, help="Hours per test window")
    parser.add_argument("--step-hours", type=int, default=6, help="Hours between test windows")
    parser.add_argument("--importance-threshold", type=float, default=0.0005, help="Permutation importance threshold for feature selection")
    args = parser.parse_args()

    print("=" * 70)
    print("FEATURE IMPORTANCE ANALYSIS & MODEL COMPARISON")
    print("=" * 70)

    # Load Polymarket
    print("\n[1/4] Loading Polymarket data...")
    pm = load_polymarket_csv(args.pm_csv)
    start_ms = int(pm["timestamp"].min().timestamp() * 1000)
    end_ms = int(pm["timestamp"].max().timestamp() * 1000)
    print(f"  Loaded {len(pm):,} rows from {pm['timestamp'].min()} to {pm['timestamp'].max()}")

    # Load/fetch BTC
    print("\n[2/4] Loading/fetching BTC data...")
    cache_path = Path(args.btc_cache)
    if cache_path.exists():
        print(f"  Using cached BTC from {cache_path}")
        btc_df = pd.read_csv(cache_path)
        btc_df["open_time"] = pd.to_datetime(btc_df["open_time"], utc=True)
        btc_df["open"] = pd.to_numeric(btc_df["open"], errors="coerce")
        btc_df["close"] = pd.to_numeric(btc_df["close"], errors="coerce")
        
        # Compute target_up if not already present
        if "target_up" not in btc_df.columns:
            btc_df["candle_5m_start"] = btc_df["open_time"].dt.floor("5min")
            candle = (
                btc_df.groupby("candle_5m_start", as_index=False)
                .agg(candle_open=("open", "first"), candle_close=("close", "last"))
                .copy()
            )
            candle["target_up"] = (candle["candle_close"] > candle["candle_open"]).astype(int)
            btc_df = btc_df.merge(candle[["candle_5m_start", "target_up"]], on="candle_5m_start", how="left")
    else:
        print("  Fetching BTC 1-second data from Binance...")
        raw_btc = fetch_klines_1s_chunked("BTCUSDT", start_ms, end_ms, chunk_hours=6)
        btc_df = build_binance_df(raw_btc)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        btc_df.to_csv(cache_path, index=False)
        print(f"  Cached to {cache_path}")

    print(f"  BTC: {len(btc_df):,} rows from {btc_df['open_time'].min()} to {btc_df['open_time'].max()}")

    # Merge and run initial walk-forward to identify important features
    print("\n[3/4] Ranking features by importance...")
    btc_df_aligned = btc_df.sort_values("open_time").copy()
    pm_aligned = pm.sort_values("timestamp").copy()

    min_ts = max(btc_df_aligned["open_time"].min(), pm_aligned["timestamp"].min())
    max_ts = min(btc_df_aligned["open_time"].max(), pm_aligned["timestamp"].max())

    btc_df_aligned = btc_df_aligned[(btc_df_aligned["open_time"] >= min_ts) & (btc_df_aligned["open_time"] <= max_ts)].copy()
    pm_aligned = pm_aligned[(pm_aligned["timestamp"] >= min_ts) & (pm_aligned["timestamp"] <= max_ts)].copy()

    # Merge: BTC features + Polymarket implied_prob + BTC target
    merged_init = btc_df_aligned.merge(
        pm_aligned[["timestamp", "slug", "elapsed", "implied_prob"]],
        left_on="open_time",
        right_on="timestamp",
        how="inner",
    )
    merged_init = merged_init.dropna(subset=ALL_FEATURES + ["target_up"]).copy()

    # Build initial train/test split for importance ranking
    train_init = merged_init.iloc[:int(len(merged_init) * 0.7)]
    X_train_init = train_init[ALL_FEATURES]
    y_train_init = train_init["target_up"].astype(int).values

    importance_df = rank_features_by_importance(
        X_train_init, y_train_init, ALL_FEATURES, ALL_FEATURES
    )

    # Select features above threshold
    selected_features = importance_df[importance_df['importance_mean'] > args.importance_threshold]['feature'].tolist()
    print(f"\n  Selected {len(selected_features)}/{len(ALL_FEATURES)} features above threshold {args.importance_threshold}:")
    print(f"  {selected_features}")

    # Save feature importance report
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    importance_df.to_csv(out_dir / "feature_importance.csv", index=False)
    print(f"\n  Saved feature importance to {out_dir / 'feature_importance.csv'}")

    # Walk-forward comparison
    print("\n[4/4] Running walk-forward validation (baseline vs reduced vs Polymarket)...")
    overall_df, splits_df = run_walk_forward_with_feature_sets(
        btc_df, pm,
        all_feature_cols=ALL_FEATURES,
        selected_feature_cols=selected_features,
        training_days=7,
        test_hours=args.test_hours,
        step_hours=args.step_hours,
    )

    # Save results
    overall_df.to_csv(out_dir / "overall_comparison.csv", index=False)
    splits_df.to_csv(out_dir / "per_split_metrics.csv", index=False)

    print("\n" + "=" * 70)
    print("OVERALL RESULTS: 7-Day Baseline vs Reduced Features vs Polymarket")
    print("=" * 70)
    print(overall_df.to_string(index=False))

    # Determine metric winners
    print("\n" + "=" * 70)
    print("METRIC WINS (higher better → AUC/Accuracy, lower better → Log Loss/Brier/CalibGap)")
    print("=" * 70)
    metrics_to_eval = ["accuracy", "auc", "log_loss", "brier", "calib_gap_pp"]
    higher_better = ["accuracy", "auc"]
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
        print(f"{m:20s}: {winner:20s} ({winner_val:.6f})")

    wins_sorted = sorted(wins.items(), key=lambda x: x[1], reverse=True)
    print("\nTotal wins by model:")
    for model_name, w in wins_sorted:
        print(f"  {model_name}: {w}")

    print("\n" + "=" * 70)
    print(f"Results saved to {out_dir}/")
    print("=" * 70)


if __name__ == "__main__":
    main()
