from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple
import traceback

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


TIME_BUCKETS = [100, 130, 160, 190, 220, 250, 291]
TIME_BUCKET_LABELS = ["100-129", "130-159", "160-189", "190-219", "220-249", "250-290"]
REGIME_VALUES = ["calm", "normal", "volatile_trend_up", "volatile_trend_down", "high_vol_chop"]


def load_polymarket_frame(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["timestamp"] = pd.to_datetime(df["timestamp_log"], unit="s", utc=True)
    df["elapsed"] = pd.to_numeric(df["elapsed"], errors="coerce")
    df["label"] = (df["winner"].astype(str).str.lower() == "up").astype(int)
    df["implied_prob"] = (
        pd.to_numeric(df["bid_YES"], errors="coerce") + pd.to_numeric(df["ask_YES"], errors="coerce")
    ) / 2.0
    df["implied_prob"] = df["implied_prob"].clip(0.0, 1.0)
    return df[["slug", "timestamp", "elapsed", "label", "implied_prob"]].copy()


def load_binance_frame(path: Path) -> pd.DataFrame:
    usecols = [
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
        "log_return_1s",
        "inst_vol_60s",
        "inst_vol_60s_bps",
        "trend_mean_60s",
        "trend_zscore",
        "vwap",
        "buy_volume_ratio",
        "candle_5m_start",
        "target_up",
        "seconds_to_5m_close",
        "fraction_of_candle_elapsed",
        "return_from_candle_open",
    ]
    df = pd.read_csv(path, usecols=usecols)
    df["open_time"] = pd.to_datetime(df["open_time"], utc=True)
    df["candle_5m_start"] = pd.to_datetime(df["candle_5m_start"], utc=True)

    if "market_regime" not in df.columns:
        vol_low = df["inst_vol_60s"].quantile(0.30)
        vol_high = df["inst_vol_60s"].quantile(0.70)
        conditions = [
            df["inst_vol_60s"] <= vol_low,
            (df["inst_vol_60s"] >= vol_high) & (df["trend_zscore"] >= 1.5),
            (df["inst_vol_60s"] >= vol_high) & (df["trend_zscore"] <= -1.5),
            (df["inst_vol_60s"] >= vol_high) & (df["trend_zscore"].abs() < 0.5),
        ]
        labels = ["calm", "volatile_trend_up", "volatile_trend_down", "high_vol_chop"]
        df["market_regime"] = np.select(conditions, labels, default="normal")

    return df.sort_values("open_time").reset_index(drop=True)


def merge_sample(pm: pd.DataFrame, bn: pd.DataFrame) -> pd.DataFrame:
    merged = pm.merge(bn, left_on="timestamp", right_on="open_time", how="inner")
    merged = merged[(merged["elapsed"] >= 100) & (merged["elapsed"] <= 290)].copy()
    merged = merged.sort_values(["candle_5m_start", "open_time"]).reset_index(drop=True)
    return merged


def add_multi_horizon_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values("open_time").copy()

    windows = [5, 15, 30, 60, 180]
    close = df["close"]
    log_ret = np.log(close).diff()

    for w in windows:
        df[f"ret_{w}s"] = close / close.shift(w) - 1.0
        df[f"vol_{w}s"] = log_ret.rolling(w, min_periods=max(3, min(w, 10))).std()
        df[f"range_{w}s"] = (df["high"].rolling(w, min_periods=max(3, min(w, 10))).max() - df["low"].rolling(w, min_periods=max(3, min(w, 10))).min()) / close
        df[f"trades_{w}s"] = df["number_of_trades"].rolling(w, min_periods=max(3, min(w, 10))).sum()
        df[f"volume_{w}s"] = df["volume"].rolling(w, min_periods=max(3, min(w, 10))).sum()
        df[f"volume_accel_{w}s"] = df["volume"] / (df["volume"].rolling(w, min_periods=max(3, min(w, 10))).mean() + 1e-12)

    df["pm_edge"] = df["implied_prob"] - 0.5
    df["pm_abs_edge"] = df["pm_edge"].abs()
    df["pm_logit"] = np.log(np.clip(df["implied_prob"], 1e-4, 1 - 1e-4) / np.clip(1 - df["implied_prob"], 1e-4, 1 - 1e-4))

    # Interaction features from notes.
    df["pm_x_time"] = df["implied_prob"] * df["fraction_of_candle_elapsed"]
    df["pm_x_vol"] = df["implied_prob"] * df["inst_vol_60s_bps"]
    df["pm_x_regime_vol"] = df["implied_prob"] * df["trend_zscore"].abs()
    df["return_x_vol"] = df["return_from_candle_open"] * df["inst_vol_60s_bps"]
    df["trend_x_vol"] = df["trend_mean_60s"] * df["inst_vol_60s_bps"]
    df["buy_ratio_x_time"] = df["buy_volume_ratio"] * df["fraction_of_candle_elapsed"]
    df["vol_x_time"] = df["inst_vol_60s_bps"] * df["fraction_of_candle_elapsed"]
    df["range_x_time"] = df["range_60s"] * df["fraction_of_candle_elapsed"]
    df["trades_x_time"] = df["trades_60s"] * df["fraction_of_candle_elapsed"]

    return df


def build_feature_columns(df: pd.DataFrame) -> List[str]:
    numeric = [
        "implied_prob",
        "pm_edge",
        "pm_abs_edge",
        "pm_logit",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "quote_asset_volume",
        "number_of_trades",
        "taker_buy_base_asset_volume",
        "taker_buy_quote_asset_volume",
        "log_return_1s",
        "inst_vol_60s",
        "inst_vol_60s_bps",
        "trend_mean_60s",
        "trend_zscore",
        "vwap",
        "buy_volume_ratio",
        "seconds_to_5m_close",
        "fraction_of_candle_elapsed",
        "return_from_candle_open",
        "pm_x_time",
        "pm_x_vol",
        "pm_x_regime_vol",
        "return_x_vol",
        "trend_x_vol",
        "buy_ratio_x_time",
        "vol_x_time",
        "range_x_time",
        "trades_x_time",
    ]
    numeric += [col for col in df.columns if col.startswith(("ret_", "vol_", "range_", "trades_", "volume_", "volume_accel_"))]
    numeric = [c for c in numeric if c in df.columns]
    seen = set()
    unique: List[str] = []
    for col in numeric:
        if col not in seen:
            seen.add(col)
            unique.append(col)
    return unique


def make_design_matrix(df: pd.DataFrame, feature_cols: List[str]) -> pd.DataFrame:
    X = df[feature_cols].copy()
    regime = pd.Categorical(df["market_regime"], categories=REGIME_VALUES)
    bucket = pd.cut(df["seconds_to_5m_close"], bins=TIME_BUCKETS, labels=TIME_BUCKET_LABELS, right=False)

    X = pd.get_dummies(X, columns=[], dummy_na=False)
    X = pd.concat(
        [
            X,
            pd.get_dummies(regime, prefix="regime", dummy_na=False),
            pd.get_dummies(bucket, prefix="bucket", dummy_na=False),
        ],
        axis=1,
    )
    return X


def sample_weights_from_recency(df: pd.DataFrame) -> np.ndarray:
    # Higher weight for more recent observations.
    order = df["open_time"].astype("int64").values
    rank = (order - order.min()) / max(1, order.max() - order.min())
    return 0.7 + 0.6 * rank


def make_pipeline(feature_names: List[str]) -> Pipeline:
    return Pipeline(
        steps=[
            (
                "prep",
                ColumnTransformer(
                    transformers=[
                        (
                            "num",
                            Pipeline(
                                steps=[
                                    ("imputer", SimpleImputer(strategy="median")),
                                    ("scaler", StandardScaler()),
                                ]
                            ),
                            feature_names,
                        )
                    ],
                    remainder="drop",
                ),
            ),
            (
                "model",
                LogisticRegression(
                    max_iter=3000,
                    solver="lbfgs",
                    C=0.6,
                ),
            ),
        ]
    )


def align_columns(frame: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    return frame.reindex(columns=columns, fill_value=0.0)


def fit_bucket_models(train_df: pd.DataFrame, feature_cols: List[str]) -> Dict[str, Pipeline]:
    models: Dict[str, Pipeline] = {}
    train_design = make_design_matrix(train_df, feature_cols)

    for bucket_label in TIME_BUCKET_LABELS:
        mask = pd.cut(train_df["seconds_to_5m_close"], bins=TIME_BUCKETS, labels=TIME_BUCKET_LABELS, right=False) == bucket_label
        part = train_df.loc[mask].copy()
        if len(part) < 1500 or part["target_up"].nunique() < 2:
            continue
        X_part = train_design.loc[part.index]
        y_part = part["target_up"].astype(int).values
        model = make_pipeline(list(X_part.columns))
        model.fit(X_part, y_part, model__sample_weight=sample_weights_from_recency(part))
        models[bucket_label] = model

    return models


def fit_global_model(train_df: pd.DataFrame, feature_cols: List[str]) -> Pipeline:
    design = make_design_matrix(train_df, feature_cols)
    model = make_pipeline(list(design.columns))
    model.fit(design, train_df["target_up"].astype(int).values, model__sample_weight=sample_weights_from_recency(train_df))
    return model


def predict_with_hierarchy(
    test_df: pd.DataFrame,
    feature_cols: List[str],
    global_model: Pipeline,
    bucket_models: Dict[str, Pipeline],
) -> np.ndarray:
    design = make_design_matrix(test_df, feature_cols)
    bucket = pd.cut(test_df["seconds_to_5m_close"], bins=TIME_BUCKETS, labels=TIME_BUCKET_LABELS, right=False)

    global_columns = list(global_model.named_steps["prep"].feature_names_in_)
    global_prob = global_model.predict_proba(align_columns(design, global_columns))[:, 1]
    final_prob = global_prob.copy()

    for bucket_label in TIME_BUCKET_LABELS:
        idx = bucket == bucket_label
        if not idx.any():
            continue
        if bucket_label in bucket_models:
            bucket_columns = list(bucket_models[bucket_label].named_steps["prep"].feature_names_in_)
            bucket_design = align_columns(design.loc[idx], bucket_columns)
            bucket_prob = bucket_models[bucket_label].predict_proba(bucket_design)[:, 1]
            # Blend the specialized bucket model with the global model.
            final_prob[idx] = 0.72 * bucket_prob + 0.28 * global_prob[idx]
        else:
            final_prob[idx] = global_prob[idx]

    return final_prob


def compute_metrics(y_true: np.ndarray, y_prob: np.ndarray) -> Dict[str, float]:
    y_pred = (y_prob >= 0.5).astype(int)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "log_loss": float(log_loss(y_true, y_prob, labels=[0, 1])),
        "brier": float(brier_score_loss(y_true, y_prob)),
        "calibration_gap_pp": float(np.mean(np.abs(y_prob - y_true)) * 100.0),
        "mean_pred": float(np.mean(y_prob)),
        "mean_actual": float(np.mean(y_true)),
    }


def cross_validate(df: pd.DataFrame, feature_cols: List[str], n_splits: int = 5) -> Tuple[pd.DataFrame, pd.DataFrame]:
    df = df.reset_index(drop=True).copy()
    groups = df["candle_5m_start"].astype(str).values
    y = df["target_up"].astype(int).values
    gkf = GroupKFold(n_splits=n_splits)

    fold_rows = []
    pred_frames = []

    for fold_idx, (train_idx, test_idx) in enumerate(gkf.split(df, y, groups=groups), start=1):
        train_df = df.iloc[train_idx].reset_index(drop=True).copy()
        test_df = df.iloc[test_idx].reset_index(drop=True).copy()

        global_model = fit_global_model(train_df, feature_cols)
        bucket_models = fit_bucket_models(train_df, feature_cols)
        y_prob = predict_with_hierarchy(test_df, feature_cols, global_model, bucket_models)

        metrics = compute_metrics(test_df["target_up"].astype(int).values, y_prob)
        fold_rows.append({"fold": fold_idx, "samples": len(test_df), **metrics})

        test_out = test_df[["slug", "open_time", "candle_5m_start", "seconds_to_5m_close", "market_regime", "target_up", "implied_prob"]].copy()
        test_out["v2_prob"] = y_prob
        pred_frames.append(test_out)

    return pd.DataFrame(fold_rows), pd.concat(pred_frames, ignore_index=True)


def compare_to_baselines(pred_df: pd.DataFrame) -> pd.DataFrame:
    y_true = pred_df["target_up"].astype(int).values
    rows = []
    rows.append({"model": "polymarket", **compute_metrics(y_true, pred_df["implied_prob"].astype(float).values)})
    rows.append({"model": "v2_hybrid", **compute_metrics(y_true, pred_df["v2_prob"].astype(float).values)})
    return pd.DataFrame(rows)


def build_bucket_report(pred_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    pred_df = pred_df.copy()
    pred_df["bucket"] = pd.cut(pred_df["seconds_to_5m_close"], bins=TIME_BUCKETS, labels=TIME_BUCKET_LABELS, right=False)
    for bucket, g in pred_df.groupby("bucket", observed=False):
        if g.empty:
            continue
        rows.append(
            {
                "bucket": str(bucket),
                "n_rows": len(g),
                "model": "polymarket",
                **compute_metrics(g["target_up"].astype(int).values, g["implied_prob"].astype(float).values),
            }
        )
        rows.append(
            {
                "bucket": str(bucket),
                "n_rows": len(g),
                "model": "v2_hybrid",
                **compute_metrics(g["target_up"].astype(int).values, g["v2_prob"].astype(float).values),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build and evaluate v2 hybrid model against Polymarket baseline")
    parser.add_argument("--polymarket-csv", default="research/polymarket_model/market_data_2sec_weekly5_with_resolutions.csv")
    parser.add_argument("--binance-csv", default="research/polymarket_model/binance_1s_for_polymarket_window.csv")
    parser.add_argument("--output-dir", default="research/polymarket_model/v2_hybrid_results")
    parser.add_argument("--n-splits", type=int, default=5)
    args = parser.parse_args()

    pm = load_polymarket_frame(Path(args.polymarket_csv))
    bn = load_binance_frame(Path(args.binance_csv))
    merged = merge_sample(pm, bn)
    merged = add_multi_horizon_features(merged)

    feature_cols = build_feature_columns(merged)
    if not feature_cols:
        raise RuntimeError("No feature columns available")

    # Keep only rows with complete features and labels.
    model_df = merged.dropna(subset=feature_cols + ["target_up", "candle_5m_start", "implied_prob"]).copy()
    fold_df, pred_df = cross_validate(model_df, feature_cols, n_splits=args.n_splits)

    overall = compare_to_baselines(pred_df)
    bucket = build_bucket_report(pred_df)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fold_df.to_csv(out_dir / "cv_metrics.csv", index=False)
    pred_df.to_csv(out_dir / "oof_predictions.csv", index=False)
    overall.to_csv(out_dir / "overall_comparison.csv", index=False)
    bucket.to_csv(out_dir / "bucket_comparison.csv", index=False)

    print("=== V2 HYBRID MODEL ===")
    print(f"Rows: {len(model_df):,}")
    print(f"Candles: {model_df['candle_5m_start'].nunique():,}")
    print(f"Features: {len(feature_cols)}")
    print("\nCV fold metrics:")
    print(fold_df.to_string(index=False, float_format=lambda x: f'{x:.6f}'))
    print("\nOverall comparison:")
    print(overall.to_string(index=False, float_format=lambda x: f'{x:.6f}'))

    wins = {"polymarket": 0, "v2_hybrid": 0}
    higher = ["accuracy", "roc_auc"]
    lower = ["log_loss", "brier", "calibration_gap_pp"]
    for metric in higher:
        wins[overall.sort_values(metric, ascending=False).iloc[0]["model"]] += 1
    for metric in lower:
        wins[overall.sort_values(metric, ascending=True).iloc[0]["model"]] += 1

    print("\nMetric wins:", wins)
    print("Fair winner:", "v2_hybrid" if wins["v2_hybrid"] > wins["polymarket"] else ("polymarket" if wins["polymarket"] > wins["v2_hybrid"] else "tie"))
    print(f"\nSaved results to {out_dir.resolve()}")


if __name__ == "__main__":
    try:
        main()
    except Exception:  # noqa: BLE001
        traceback.print_exc()
        raise
