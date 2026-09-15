"""Train and evaluate instantaneous 5-minute candle direction probability models.

At each 1-second timestamp t, estimate P(5m candle closes up).
Adds model comparison, probability calibration, and time-to-close diagnostics.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.calibration import CalibratedClassifierCV
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train a model to predict P(5m candle closes up) at each 1-second row "
            "using GroupKFold cross-validation."
        )
    )
    parser.add_argument(
        "--input",
        default="research/btcusdt_1s_last24h.csv",
        help="Path to enriched 1-second CSV",
    )
    parser.add_argument(
        "--n-splits",
        type=int,
        default=5,
        help="Number of GroupKFold splits",
    )
    parser.add_argument(
        "--models",
        default="logreg,hgb,xgboost,lightgbm",
        help=(
            "Comma-separated models to compare. Supported: "
            "logreg, hgb, xgboost, lightgbm"
        ),
    )
    parser.add_argument(
        "--calibration",
        choices=["none", "sigmoid", "isotonic"],
        default="sigmoid",
        help="Probability calibration method",
    )
    parser.add_argument(
        "--pred-output",
        default="research/btcusdt_1s_with_up_prob.csv",
        help="Output CSV with out-of-fold probabilities",
    )
    parser.add_argument(
        "--metrics-output",
        default="research/model_cv_metrics.csv",
        help="Output CSV for fold-level metrics across models",
    )
    parser.add_argument(
        "--bucket-output",
        default="research/time_to_close_bucket_report.csv",
        help="Output CSV for time-to-close bucket diagnostics",
    )
    return parser.parse_args()


def load_and_prepare(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Input CSV not found: {path}")

    df = pd.read_csv(path)
    required = [
        "open_time",
        "open",
        "close",
        "volume",
        "number_of_trades",
        "inst_vol_60s",
        "inst_vol_60s_bps",
        "log_return_1s",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df["open_time"] = pd.to_datetime(df["open_time"], utc=True)
    df = df.sort_values("open_time").reset_index(drop=True)

    # 5-minute candle bucket that contains each 1-second timestamp.
    df["candle_5m_start"] = df["open_time"].dt.floor("5min")

    # Candle-level open/close and direction target.
    candle = (
        df.groupby("candle_5m_start", as_index=False)
        .agg(candle_open=("open", "first"), candle_close=("close", "last"))
        .copy()
    )
    candle["target_up"] = (candle["candle_close"] > candle["candle_open"]).astype(int)
    candle = candle[candle["candle_close"] != candle["candle_open"]].copy()

    df = df.merge(candle[["candle_5m_start", "target_up"]], on="candle_5m_start", how="inner")

    # Time-position features inside the 5-minute candle.
    candle_end = df["candle_5m_start"] + pd.Timedelta(minutes=5)
    seconds_to_close = (candle_end - df["open_time"]).dt.total_seconds()
    df["seconds_to_5m_close"] = seconds_to_close.clip(lower=0)
    df["fraction_of_candle_elapsed"] = 1.0 - (df["seconds_to_5m_close"] / 300.0)

    # Price position and microstructure-like helpers.
    df["return_from_candle_open"] = (df["close"] / df.groupby("candle_5m_start")["open"].transform("first")) - 1.0
    if "buy_volume_ratio" in df.columns:
        df["buy_volume_ratio"] = df["buy_volume_ratio"].clip(lower=0, upper=1)

    return df


def parse_model_list(raw_models: str) -> List[str]:
    models = [m.strip().lower() for m in raw_models.split(",") if m.strip()]
    if not models:
        raise ValueError("No models requested")
    return models


def build_feature_list(df: pd.DataFrame) -> List[str]:
    candidates = [
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
    return [col for col in candidates if col in df.columns]


def make_preprocessor(features: List[str], use_scaler: bool) -> ColumnTransformer:
    steps = [("imputer", SimpleImputer(strategy="median"))]
    if use_scaler:
        steps.append(("scaler", StandardScaler()))

    return ColumnTransformer(
        transformers=[("num", Pipeline(steps=steps), features)],
        remainder="drop",
    )


def build_estimator(model_name: str, features: List[str], calibration: str):
    model_name = model_name.lower()

    if model_name == "logreg":
        preprocessor = make_preprocessor(features, use_scaler=True)
        base_model = LogisticRegression(max_iter=2000, solver="lbfgs")
    elif model_name == "hgb":
        preprocessor = make_preprocessor(features, use_scaler=False)
        base_model = HistGradientBoostingClassifier(
            max_depth=6,
            learning_rate=0.06,
            max_iter=250,
            random_state=42,
        )
    elif model_name == "xgboost":
        try:
            from xgboost import XGBClassifier
        except ImportError as exc:
            raise ImportError("xgboost is not installed") from exc
        preprocessor = make_preprocessor(features, use_scaler=False)
        base_model = XGBClassifier(
            n_estimators=350,
            learning_rate=0.05,
            max_depth=5,
            subsample=0.9,
            colsample_bytree=0.9,
            objective="binary:logistic",
            eval_metric="logloss",
            random_state=42,
            n_jobs=1,
        )
    elif model_name == "lightgbm":
        try:
            from lightgbm import LGBMClassifier
        except ImportError as exc:
            raise ImportError("lightgbm is not installed") from exc
        preprocessor = make_preprocessor(features, use_scaler=False)
        base_model = LGBMClassifier(
            n_estimators=500,
            learning_rate=0.03,
            num_leaves=31,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=42,
            n_jobs=1,
        )
    else:
        raise ValueError(f"Unsupported model: {model_name}")

    estimator = Pipeline(steps=[("prep", preprocessor), ("model", base_model)])
    if calibration != "none":
        estimator = CalibratedClassifierCV(estimator=estimator, method=calibration, cv=3)

    return estimator


def evaluate_model(
    df: pd.DataFrame,
    features: List[str],
    n_splits: int,
    model_name: str,
    calibration: str,
) -> Tuple[pd.DataFrame, np.ndarray]:
    X = df[features].copy()
    y = df["target_up"].astype(int).values
    groups = df["candle_5m_start"].astype(str).values

    unique_groups = np.unique(groups)
    if len(unique_groups) < n_splits:
        raise ValueError(
            f"Not enough 5m candles for n_splits={n_splits}. Available candles: {len(unique_groups)}"
        )

    estimator = build_estimator(model_name=model_name, features=features, calibration=calibration)

    gkf = GroupKFold(n_splits=n_splits)
    oof_prob_up = np.full(shape=len(df), fill_value=np.nan)

    fold_metrics = []
    for fold_idx, (train_idx, test_idx) in enumerate(gkf.split(X, y, groups=groups), start=1):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        estimator.fit(X_train, y_train)
        prob_up = estimator.predict_proba(X_test)[:, 1]
        pred_up = (prob_up >= 0.5).astype(int)
        oof_prob_up[test_idx] = prob_up

        fold_metrics.append(
            {
            "model": model_name,
            "calibration": calibration,
                "fold": fold_idx,
                "samples": int(len(test_idx)),
                "accuracy": float(accuracy_score(y_test, pred_up)),
                "roc_auc": float(roc_auc_score(y_test, prob_up)),
                "log_loss": float(log_loss(y_test, np.column_stack([1 - prob_up, prob_up]))),
                "brier": float(brier_score_loss(y_test, prob_up)),
            }
        )

    return pd.DataFrame(fold_metrics), oof_prob_up


def summarize_metrics(metrics_df: pd.DataFrame) -> pd.DataFrame:
    summary = (
        metrics_df.groupby(["model", "calibration"], as_index=False)
        .agg(
            accuracy_mean=("accuracy", "mean"),
            accuracy_std=("accuracy", "std"),
            roc_auc_mean=("roc_auc", "mean"),
            roc_auc_std=("roc_auc", "std"),
            log_loss_mean=("log_loss", "mean"),
            log_loss_std=("log_loss", "std"),
            brier_mean=("brier", "mean"),
            brier_std=("brier", "std"),
        )
        .sort_values(["roc_auc_mean", "log_loss_mean"], ascending=[False, True])
        .reset_index(drop=True)
    )
    return summary


def build_time_to_close_report(df: pd.DataFrame, prob_col: str) -> pd.DataFrame:
    bins = [0, 5, 15, 30, 60, 120, 180, 240, 300]
    labels = [
        "0-5s",
        "5-15s",
        "15-30s",
        "30-60s",
        "60-120s",
        "120-180s",
        "180-240s",
        "240-300s",
    ]

    report_df = df[["seconds_to_5m_close", "target_up", prob_col]].copy()
    report_df = report_df.rename(columns={prob_col: "prob_up"})
    report_df["time_to_close_bucket"] = pd.cut(
        report_df["seconds_to_5m_close"],
        bins=bins,
        labels=labels,
        include_lowest=True,
        right=True,
    )

    rows: List[Dict] = []
    for bucket, part in report_df.groupby("time_to_close_bucket", observed=False):
        if part.empty:
            continue
        y_true = part["target_up"].astype(int).values
        y_prob = part["prob_up"].astype(float).values
        y_pred = (y_prob >= 0.5).astype(int)

        auc = np.nan
        if len(np.unique(y_true)) == 2:
            auc = float(roc_auc_score(y_true, y_prob))

        rows.append(
            {
                "time_to_close_bucket": str(bucket),
                "samples": int(len(part)),
                "actual_up_rate": float(np.mean(y_true)),
                "avg_pred_up_prob": float(np.mean(y_prob)),
                "accuracy": float(accuracy_score(y_true, y_pred)),
                "brier": float(brier_score_loss(y_true, y_prob)),
                "roc_auc": auc,
            }
        )

    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.pred_output)
    metrics_path = Path(args.metrics_output)
    bucket_path = Path(args.bucket_output)

    df = load_and_prepare(input_path)
    features = build_feature_list(df)
    requested_models = parse_model_list(args.models)

    if not features:
        raise ValueError("No usable feature columns found")

    print(f"Loaded {len(df):,} rows")
    print(f"Using {len(features)} features: {features}")
    print(f"Requested models: {requested_models}")
    print(f"Calibration: {args.calibration}")

    all_fold_metrics: List[pd.DataFrame] = []
    model_prob_cols: Dict[str, str] = {}
    skipped_models: List[str] = []
    pred_df = df.copy()

    for model_name in requested_models:
        print(f"\nRunning model: {model_name}")
        try:
            fold_metrics, oof_prob = evaluate_model(
                df=df,
                features=features,
                n_splits=args.n_splits,
                model_name=model_name,
                calibration=args.calibration,
            )
        except Exception as exc:
            print(f"Skipping model '{model_name}' due to error: {exc}")
            skipped_models.append(model_name)
            continue

        prob_col = f"prob_up_oof_{model_name}"
        pred_df[prob_col] = oof_prob
        model_prob_cols[model_name] = prob_col
        all_fold_metrics.append(fold_metrics)

        print(fold_metrics.to_string(index=False, float_format=lambda x: f"{x:.5f}"))

    if not all_fold_metrics:
        raise RuntimeError("No model completed successfully")

    metrics_df = pd.concat(all_fold_metrics, axis=0, ignore_index=True)
    summary_df = summarize_metrics(metrics_df)

    print("\nCross-validation summary (sorted by ROC AUC desc, log loss asc)")
    print(summary_df.to_string(index=False, float_format=lambda x: f"{x:.5f}"))

    best_model = summary_df.iloc[0]["model"]
    best_prob_col = model_prob_cols[best_model]
    pred_df["prob_5m_close_up_oof"] = pred_df[best_prob_col]
    pred_df["pred_up_oof"] = (pred_df["prob_5m_close_up_oof"] >= 0.5).astype(int)
    pred_df["best_model"] = best_model
    pred_df["calibration_method"] = args.calibration

    time_bucket_report = build_time_to_close_report(pred_df, prob_col="prob_5m_close_up_oof")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    bucket_path.parent.mkdir(parents=True, exist_ok=True)

    pred_df.to_csv(output_path, index=False)
    metrics_df.to_csv(metrics_path, index=False)
    time_bucket_report.to_csv(bucket_path, index=False)

    print(f"\nBest model selected: {best_model}")
    if skipped_models:
        print(f"Skipped models: {skipped_models}")

    print(f"Saved out-of-fold predictions to {output_path.resolve()}")
    print(f"Saved CV fold metrics to {metrics_path.resolve()}")
    print(f"Saved time-to-close report to {bucket_path.resolve()}")


if __name__ == "__main__":
    main()
