"""Strict walk-forward comparison of calibrated 7-day logistic models vs Polymarket.

This script compares three models on identical test rows:
- 7day_logistic_baseline: standard sigmoid calibration (CalibratedClassifierCV)
- 7day_logistic_enhanced: time-safe holdout calibration + bucketed method selection
    (sigmoid vs isotonic) + shrinkage to base rate
- polymarket: implied probabilities from Polymarket order book midpoint
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
ELAPSED_BINS = [100, 130, 160, 190, 220, 250, 291]
ELAPSED_LABELS = ["100-129", "130-159", "160-189", "190-219", "220-249", "250-290"]


def make_base_pipeline(feature_cols: List[str]) -> Pipeline:
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

        # Time-safe split inside bucket: select method on validation segment only.
        cut = int(len(g) * 0.7)
        if cut < 50 or (len(g) - cut) < 50:
            cut = len(g) // 2

        x_tr, y_tr = x[:cut], y[:cut]
        x_va, y_va = x[cut:], y[cut:]
        if len(x_va) < 20 or len(np.unique(y_va)) < 2:
            calibrators[label] = ("identity", None)
            diagnostics.append({"bucket": label, "method": "identity", "n": len(g), "log_loss": np.nan})
            continue

        # Sigmoid option.
        sig = fit_sigmoid_calibrator(x_tr, y_tr)
        p_sig_va = predict_sigmoid(sig, x_va)
        ll_sig = safe_log_loss(y_va, p_sig_va)

        # Isotonic option with guardrails.
        iso_allowed = len(g) >= min_isotonic_samples
        ll_iso = np.inf
        if iso_allowed:
            iso = IsotonicRegression(out_of_bounds="clip")
            iso.fit(x_tr, y_tr)
            p_iso_va = np.clip(iso.predict(x_va), 1e-6, 1 - 1e-6)
            ll_iso = safe_log_loss(y_va, p_iso_va)

            # Smoothness guardrail: too many steps in isotonic is a common overfit symptom.
            if hasattr(iso, "X_thresholds_") and len(iso.X_thresholds_) > 120:
                iso_allowed = False

        if iso_allowed and (ll_iso + min_isotonic_improvement < ll_sig):
            # Refit selected method on full bucket calibration data.
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

        # Hard guardrail against making calibration gap worse on the calibration holdout.
        if gap_pp > base_gap_pp + max_calib_gap_regression_pp:
            continue

        # Joint objective: prefer better log loss while accounting for Brier.
        score = 0.7 * ll + 0.3 * br
        if score < best_score:
            best_score = score
            best_lambda = lam
            best_meta = {"log_loss": ll, "brier": br, "calib_gap_pp": gap_pp}

    return best_lambda, best_meta


def fetch_klines_1s(symbol: str, start_ms: int, end_ms: int) -> List[List]:
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
    chunk_ms = int(chunk_hours * 3600 * 1000)
    cursor = start_ms
    merged: List[List] = []
    chunk_idx = 0

    while cursor < end_ms:
        chunk_idx += 1
        c_end = min(end_ms, cursor + chunk_ms)
        print(f"Fetching Binance chunk {chunk_idx}: {pd.to_datetime(cursor, unit='ms', utc=True)} -> {pd.to_datetime(c_end, unit='ms', utc=True)}")
        rows = fetch_klines_1s(symbol=symbol, start_ms=cursor, end_ms=c_end)
        merged.extend(rows)
        cursor = c_end + 1000

    return merged


def build_binance_df(raw_rows: List[List]) -> pd.DataFrame:
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

    # Feature engineering matching window generalization tests
    df["log_return_1s"] = np.log(df["close"]).diff()

    vol_window = 60
    df["inst_vol_60s"] = df["log_return_1s"].rolling(window=vol_window, min_periods=10).std()
    df["inst_vol_60s_bps"] = df["inst_vol_60s"] * 10000
    df["trend_mean_60s"] = df["log_return_1s"].rolling(window=vol_window, min_periods=10).mean()
    df["trend_zscore"] = df["trend_mean_60s"] / (df["inst_vol_60s"] + 1e-12)
    df["vwap"] = df["quote_asset_volume"] / df["volume"].replace(0, np.nan)
    df["buy_volume_ratio"] = df["taker_buy_base_asset_volume"] / df["volume"].replace(0, np.nan)

    # 5-minute candle labels
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


def load_polymarket_1s(polymarket_csv: Path) -> pd.DataFrame:
    """Load Polymarket data upsampled to 1-second resolution."""
    df = pd.read_csv(polymarket_csv)
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


def run_walk_forward_test(
    binance_df: pd.DataFrame,
    polymarket_df: pd.DataFrame,
    training_days: int = 7,
    test_hours: float = 6.0,
    step_hours: float = 6.0,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Run walk-forward backtest comparing baseline/enhanced 7d models vs Polymarket.
    
    Returns:
        (overall metrics DataFrame, per-split metrics DataFrame, row-level predictions DataFrame)
    """
    
    feature_cols = [
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
    feature_cols = [c for c in feature_cols if c in binance_df.columns]
    
    # Align timestamps
    binance_df = binance_df.sort_values("open_time").copy()
    polymarket_df = polymarket_df.sort_values("timestamp").copy()
    
    min_ts = max(binance_df["open_time"].min(), polymarket_df["timestamp"].min())
    max_ts = min(binance_df["open_time"].max(), polymarket_df["timestamp"].max())
    
    binance_df = binance_df[(binance_df["open_time"] >= min_ts) & (binance_df["open_time"] <= max_ts)].copy()
    polymarket_df = polymarket_df[(polymarket_df["timestamp"] >= min_ts) & (polymarket_df["timestamp"] <= max_ts)].copy()
    
    # Merge on exact second
    merged = binance_df.merge(
        polymarket_df[["timestamp", "slug", "elapsed", "implied_prob", "label"]],
        left_on="open_time",
        right_on="timestamp",
        how="inner",
    )
    merged = merged.dropna(subset=feature_cols + ["target_up"]).copy()
    
    if len(merged) == 0:
        raise ValueError("No merged rows after alignment")
    
    # Walk-forward setup
    test_seconds = int(test_hours * 3600)
    step_seconds = int(step_hours * 3600)
    train_seconds = int(training_days * 24 * 3600)
    
    first_test_ts = binance_df["open_time"].min() + pd.Timedelta(seconds=train_seconds)
    last_test_ts = binance_df["open_time"].max() - pd.Timedelta(seconds=test_seconds)
    
    splits = []
    row_level_records = []
    test_start = first_test_ts
    split_idx = 0
    
    while test_start <= last_test_ts:
        split_idx += 1
        train_start = test_start - pd.Timedelta(seconds=train_seconds)
        test_end = test_start + pd.Timedelta(seconds=test_seconds)
        
        train_mask = (merged["open_time"] >= train_start) & (merged["open_time"] < test_start)
        test_mask = (merged["open_time"] >= test_start) & (merged["open_time"] < test_end)
        
        train_data = merged[train_mask].copy()
        test_data = merged[test_mask].copy()
        
        if len(train_data) < 100 or len(test_data) < 50:
            test_start += pd.Timedelta(seconds=step_seconds)
            continue
        
        train_data = train_data.sort_values("open_time").reset_index(drop=True)
        y_train = train_data["target_up"].astype(int).values

        X_test = test_data[feature_cols]
        y_test = test_data["target_up"].astype(int).values

        # Baseline model: standard sigmoid calibration via CV on full training fold.
        base_pipeline = make_base_pipeline(feature_cols)
        calibrated_model = CalibratedClassifierCV(base_pipeline, method="sigmoid", cv=3)
        calibrated_model.fit(train_data[feature_cols], y_train)
        lr_base_probs = np.clip(calibrated_model.predict_proba(X_test)[:, 1], 1e-6, 1 - 1e-6)

        # Enhanced model: time-safe subtrain/calib split + bucket calibrators + shrinkage.
        cut_idx = int(len(train_data) * 0.8)
        if cut_idx < 1000 or (len(train_data) - cut_idx) < 500:
            cut_idx = max(100, int(len(train_data) * 0.7))

        subtrain = train_data.iloc[:cut_idx].copy()
        calib = train_data.iloc[cut_idx:].copy()

        if len(subtrain) < 100 or len(calib) < 100 or calib["target_up"].nunique() < 2:
            lr_enh_probs = lr_base_probs.copy()
            bucket_diag = pd.DataFrame()
            best_lambda = 1.0
            shrink_meta = {"log_loss": np.nan, "brier": np.nan, "calib_gap_pp": np.nan}
        else:
            subtrain_model = make_base_pipeline(feature_cols)
            subtrain_model.fit(subtrain[feature_cols], subtrain["target_up"].astype(int).values)

            raw_calib = np.clip(
                subtrain_model.predict_proba(calib[feature_cols])[:, 1],
                1e-6,
                1 - 1e-6,
            )
            raw_test = np.clip(
                subtrain_model.predict_proba(X_test)[:, 1],
                1e-6,
                1 - 1e-6,
            )

            calib_df = pd.DataFrame(
                {
                    "elapsed": calib["elapsed"].to_numpy(dtype=float),
                    "y_true": calib["target_up"].astype(int).to_numpy(),
                    "raw_prob": raw_calib,
                    "ts_order": np.arange(len(calib), dtype=int),
                }
            )
            calib_df["elapsed_bucket"] = bucket_series(calib_df["elapsed"])

            calibrators, bucket_diag = fit_bucket_calibrators(calib_df)

            p_calib_bucketed = apply_bucket_calibrators(
                raw_calib,
                calib["elapsed"].to_numpy(dtype=float),
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
                test_data["elapsed"].to_numpy(dtype=float),
                calibrators,
            )
            lr_enh_probs = np.clip(best_lambda * p_test_bucketed + (1.0 - best_lambda) * base_rate, 1e-6, 1 - 1e-6)

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
            "model": "7day_logistic_enhanced",
            "best_shrinkage_lambda": best_lambda,
            "enh_calib_holdout_log_loss": shrink_meta.get("log_loss", np.nan) if len(subtrain) >= 100 and len(calib) >= 100 else np.nan,
            "enh_calib_holdout_brier": shrink_meta.get("brier", np.nan) if len(subtrain) >= 100 and len(calib) >= 100 else np.nan,
            "enh_calib_holdout_gap_pp": shrink_meta.get("calib_gap_pp", np.nan) if len(subtrain) >= 100 and len(calib) >= 100 else np.nan,
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

        split_rows = pd.DataFrame(
            {
                "split_idx": split_idx,
                "timestamp": test_data["open_time"].values,
                "slug": test_data["slug"].values,
                "elapsed": test_data["elapsed"].values,
                "y_true": y_test,
                "prob_7d_baseline": lr_base_probs,
                "prob_7d_enhanced": lr_enh_probs,
                "prob_polymarket": pm_probs,
            }
        )
        row_level_records.append(split_rows)

        if not bucket_diag.empty:
            chosen = ", ".join(
                f"{r.bucket}:{r.method}" for r in bucket_diag.itertuples(index=False)
            )
        else:
            chosen = "fallback"

        print(
            f"Split {split_idx}: {test_start} -> {test_end} | Train: {len(train_data):,} | "
            f"Test: {len(test_data):,} | lambda={best_lambda:.2f} | bucket_methods={chosen}"
        )
        
        test_start += pd.Timedelta(seconds=step_seconds)
    
    if not splits:
        raise ValueError("No valid walk-forward splits generated")
    
    splits_df = pd.DataFrame(splits)
    
    # Overall metrics averaged across splits
    overall = []
    for model_name in ["7day_logistic_baseline", "7day_logistic_enhanced", "polymarket"]:
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
        description="Strict walk-forward comparison: baseline/enhanced 7-day logistic vs Polymarket"
    )
    parser.add_argument(
        "--polymarket-csv",
        default=r"C:\Users\fiona\Desktop\polymarket_bot\notes\resources\market_data_2sec_weekly5_with_resolutions.csv",
        help="Path to Polymarket 2-second data",
    )
    parser.add_argument(
        "--binance-cache",
        default=r"C:\Users\fiona\Desktop\polymarket_bot\notes\resources\binance_1s_for_polymarket_window.csv",
        help="Path to cache Binance 1-second data",
    )
    parser.add_argument(
        "--output-dir",
        default="research/polymarket_model/strict_7day_vs_pm_resultsv2",
        help="Output directory for results",
    )
    parser.add_argument("--symbol", default="BTCUSDT", help="Binance symbol")
    parser.add_argument("--test-hours", type=float, default=6.0, help="Test block length in hours")
    parser.add_argument("--step-hours", type=float, default=6.0, help="Step between test blocks in hours")
    args = parser.parse_args()

    print("Loading Polymarket data...")
    pm = load_polymarket_1s(Path(args.polymarket_csv))
    print(f"Polymarket: {len(pm):,} rows from {pm['timestamp'].min()} to {pm['timestamp'].max()}")

    # Load or fetch Binance data
    cache_path = Path(args.binance_cache)
    start_ts = pm["timestamp"].min().floor("s")
    end_ts = pm["timestamp"].max().ceil("s")

    if cache_path.exists():
        print(f"Loading Binance cache from {cache_path}...")
        bn = pd.read_csv(cache_path)
        bn["open_time"] = pd.to_datetime(bn["open_time"], utc=True)
        bn = bn[(bn["open_time"] >= start_ts) & (bn["open_time"] <= end_ts)].copy()
    else:
        print("Fetching Binance 1-second data...")
        start_ms = int(start_ts.timestamp() * 1000)
        end_ms = int(end_ts.timestamp() * 1000)
        raw = fetch_klines_1s_chunked(args.symbol, start_ms, end_ms, chunk_hours=6)
        bn = build_binance_df(raw)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        bn.to_csv(cache_path, index=False)
        print(f"Cached Binance data to {cache_path}")

    print(f"Binance: {len(bn):,} rows from {bn['open_time'].min()} to {bn['open_time'].max()}")

    print("\nRunning walk-forward validation (7-day training windows)...")
    overall_df, splits_df, row_level_df = run_walk_forward_test(
        bn,
        pm,
        training_days=7,
        test_hours=args.test_hours,
        step_hours=args.step_hours,
    )

    # Save results
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    overall_df.to_csv(out_dir / "overall_comparison.csv", index=False)
    splits_df.to_csv(out_dir / "per_split_metrics.csv", index=False)
    row_level_df.to_csv(out_dir / "row_level_predictions.csv", index=False)

    print("\n" + "=" * 60)
    print("OVERALL COMPARISON (7-day Training Window)")
    print("=" * 60)
    print(overall_df.to_string(index=False))

    # Determine winner
    print("\n" + "=" * 60)
    print("METRIC WINS (higher better → AUC/Accuracy, lower better → Log Loss/Brier/CalibGap)")
    print("=" * 60)
    metrics = ["accuracy", "roc_auc", "log_loss", "brier", "calibration_gap_pp"]
    higher_better = ["accuracy", "roc_auc"]
    models = overall_df["model"].tolist()
    wins = {m: 0 for m in models}

    for m in metrics:
        col = f"{m}_mean"
        if col not in overall_df.columns:
            continue
        if m in higher_better:
            winner = overall_df.loc[overall_df[col].idxmax(), "model"]
        else:
            winner = overall_df.loc[overall_df[col].idxmin(), "model"]
        wins[winner] += 1
        winner_val = float(overall_df.loc[overall_df["model"] == winner, col].iloc[0])
        print(f"{m:25s}: {winner:24s} ({winner_val:.6f})")

    wins_sorted = sorted(wins.items(), key=lambda x: x[1], reverse=True)
    print("\nTotal wins by model:")
    for model_name, w in wins_sorted:
        print(f"  {model_name}: {w}")

    top_model = wins_sorted[0][0]
    if top_model == "polymarket":
        print(">>> WINNER: Polymarket Implied Probabilities")
    elif top_model == "7day_logistic_enhanced":
        print(">>> WINNER: 7-day Logistic (Enhanced Calibration)")
    else:
        print(">>> WINNER: 7-day Logistic (Baseline Calibration)")

    print(f"\nResults saved to {out_dir}/")


if __name__ == "__main__":
    main()
