"""Walk-forward comparison of training windows for 5m close-direction prediction.

Compares candidate training windows (default: 1, 2, 3, 7 days) using
strictly future test blocks to estimate real generalization.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


DEFAULT_WINDOWS = [1, 2, 3, 7]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare 1d/2d/3d/7d training windows with walk-forward testing on future candles."
        )
    )
    parser.add_argument(
        "--input",
        default="research/btcusdt_1s_last24h.csv",
        help="Path to enriched 1-second CSV",
    )
    parser.add_argument(
        "--windows-days",
        default=",".join(str(x) for x in DEFAULT_WINDOWS),
        help="Comma-separated training windows in days (e.g. 1,2,3,7)",
    )
    parser.add_argument(
        "--test-hours",
        type=float,
        default=6.0,
        help="Future-only test block length in hours for each walk-forward split",
    )
    parser.add_argument(
        "--step-hours",
        type=float,
        default=6.0,
        help="Gap between consecutive walk-forward anchors in hours",
    )
    parser.add_argument(
        "--calibration",
        choices=["none", "sigmoid", "isotonic"],
        default="sigmoid",
        help="Probability calibration on training data",
    )
    parser.add_argument(
        "--output-dir",
        default="research/window_generalization_tests/output",
        help="Directory for metrics outputs",
    )
    return parser.parse_args()


def parse_windows(raw: str) -> List[int]:
    values = [int(x.strip()) for x in raw.split(",") if x.strip()]
    values = sorted(set(values))
    if not values:
        raise ValueError("No valid windows were provided")
    if any(v <= 0 for v in values):
        raise ValueError("All training windows must be positive integers")
    return values


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

    df["candle_5m_start"] = df["open_time"].dt.floor("5min")

    candle = (
        df.groupby("candle_5m_start", as_index=False)
        .agg(candle_open=("open", "first"), candle_close=("close", "last"))
        .copy()
    )
    candle["target_up"] = (candle["candle_close"] > candle["candle_open"]).astype(int)
    candle = candle[candle["candle_close"] != candle["candle_open"]].copy()

    df = df.merge(candle[["candle_5m_start", "target_up"]], on="candle_5m_start", how="inner")

    candle_end = df["candle_5m_start"] + pd.Timedelta(minutes=5)
    seconds_to_close = (candle_end - df["open_time"]).dt.total_seconds()
    df["seconds_to_5m_close"] = seconds_to_close.clip(lower=0)
    df["fraction_of_candle_elapsed"] = 1.0 - (df["seconds_to_5m_close"] / 300.0)

    df["return_from_candle_open"] = (
        df["close"] / df.groupby("candle_5m_start")["open"].transform("first")
    ) - 1.0
    if "buy_volume_ratio" in df.columns:
        df["buy_volume_ratio"] = df["buy_volume_ratio"].clip(lower=0, upper=1)

    return df


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


def build_estimator(features: List[str], calibration: str):
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
                features,
            )
        ],
        remainder="drop",
    )

    base = Pipeline(
        steps=[
            ("prep", preprocessor),
            ("model", LogisticRegression(max_iter=2000, solver="lbfgs")),
        ]
    )

    if calibration == "none":
        return base

    return CalibratedClassifierCV(estimator=base, method=calibration, cv=3)


def make_anchors(
    df: pd.DataFrame,
    windows_days: List[int],
    test_hours: float,
    step_hours: float,
) -> List[pd.Timestamp]:
    max_window = max(windows_days)
    first_anchor = df["open_time"].min() + pd.Timedelta(days=max_window)
    last_anchor = df["open_time"].max() - pd.Timedelta(hours=test_hours)

    if first_anchor >= last_anchor:
        return []

    anchors = []
    current = first_anchor
    step = pd.Timedelta(hours=step_hours)
    while current <= last_anchor:
        anchors.append(current)
        current = current + step

    return anchors


def evaluate_window_on_anchor(
    df: pd.DataFrame,
    features: List[str],
    window_days: int,
    anchor: pd.Timestamp,
    test_hours: float,
    calibration: str,
) -> tuple[Dict, pd.DataFrame]:
    train_start = anchor - pd.Timedelta(days=window_days)
    train_end = anchor
    test_end = anchor + pd.Timedelta(hours=test_hours)

    train_mask = (df["open_time"] >= train_start) & (df["open_time"] < train_end)
    test_mask = (df["open_time"] >= anchor) & (df["open_time"] < test_end)

    train_df = df.loc[train_mask]
    test_df = df.loc[test_mask]

    if train_df.empty or test_df.empty:
        return (
            {
                "window_days": window_days,
                "anchor_start": anchor,
                "anchor_end": test_end,
                "train_rows": int(len(train_df)),
                "test_rows": int(len(test_df)),
                "train_candles": int(train_df["candle_5m_start"].nunique()),
                "test_candles": int(test_df["candle_5m_start"].nunique()),
                "status": "skipped_empty_split",
            },
            pd.DataFrame(),
        )

    y_train = train_df["target_up"].astype(int).values
    y_test = test_df["target_up"].astype(int).values

    if len(np.unique(y_train)) < 2:
        return (
            {
                "window_days": window_days,
                "anchor_start": anchor,
                "anchor_end": test_end,
                "train_rows": int(len(train_df)),
                "test_rows": int(len(test_df)),
                "train_candles": int(train_df["candle_5m_start"].nunique()),
                "test_candles": int(test_df["candle_5m_start"].nunique()),
                "status": "skipped_single_class_train",
            },
            pd.DataFrame(),
        )

    model = build_estimator(features=features, calibration=calibration)
    model.fit(train_df[features], y_train)

    prob_up = model.predict_proba(test_df[features])[:, 1]
    pred_up = (prob_up >= 0.5).astype(int)

    row = {
        "window_days": window_days,
        "anchor_start": anchor,
        "anchor_end": test_end,
        "train_rows": int(len(train_df)),
        "test_rows": int(len(test_df)),
        "train_candles": int(train_df["candle_5m_start"].nunique()),
        "test_candles": int(test_df["candle_5m_start"].nunique()),
        "status": "ok",
        "accuracy": float(accuracy_score(y_test, pred_up)),
        "brier": float(brier_score_loss(y_test, prob_up)),
        "log_loss": float(log_loss(y_test, np.column_stack([1 - prob_up, prob_up]))),
        "actual_up_rate": float(np.mean(y_test)),
        "avg_pred_up_prob": float(np.mean(prob_up)),
        "calibration_gap": float(abs(np.mean(prob_up) - np.mean(y_test))),
    }

    if len(np.unique(y_test)) == 2:
        row["roc_auc"] = float(roc_auc_score(y_test, prob_up))
    else:
        row["roc_auc"] = np.nan

    pred_df = test_df[["open_time", "candle_5m_start", "seconds_to_5m_close", "target_up"]].copy()
    pred_df["window_days"] = window_days
    pred_df["anchor_start"] = anchor
    pred_df["anchor_end"] = test_end
    pred_df["prob_up"] = prob_up

    return row, pred_df


def build_time_to_close_bucket_report(pred_df: pd.DataFrame) -> pd.DataFrame:
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

    work = pred_df.copy()
    work["time_to_close_bucket"] = pd.cut(
        work["seconds_to_5m_close"],
        bins=bins,
        labels=labels,
        include_lowest=True,
        right=True,
    )

    rows: List[Dict] = []
    for bucket, part in work.groupby("time_to_close_bucket", observed=False):
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
                "log_loss": float(log_loss(y_true, np.column_stack([1 - y_prob, y_prob]))),
                "roc_auc": auc,
            }
        )

    return pd.DataFrame(rows)


def build_summary(per_split: pd.DataFrame) -> pd.DataFrame:
    ok = per_split[per_split["status"] == "ok"].copy()
    if ok.empty:
        return pd.DataFrame()

    summary = (
        ok.groupby("window_days", as_index=False)
        .agg(
            splits=("window_days", "size"),
            accuracy_mean=("accuracy", "mean"),
            accuracy_std=("accuracy", "std"),
            roc_auc_mean=("roc_auc", "mean"),
            roc_auc_std=("roc_auc", "std"),
            brier_mean=("brier", "mean"),
            brier_std=("brier", "std"),
            log_loss_mean=("log_loss", "mean"),
            log_loss_std=("log_loss", "std"),
            calibration_gap_mean=("calibration_gap", "mean"),
            calibration_gap_std=("calibration_gap", "std"),
        )
        .sort_values(["log_loss_mean", "brier_mean", "accuracy_mean"], ascending=[True, True, False])
        .reset_index(drop=True)
    )

    summary["rank_log_loss"] = summary["log_loss_mean"].rank(method="dense", ascending=True).astype(int)
    summary["rank_brier"] = summary["brier_mean"].rank(method="dense", ascending=True).astype(int)
    summary["rank_accuracy"] = summary["accuracy_mean"].rank(method="dense", ascending=False).astype(int)
    return summary


def build_anchor_wins(per_split: pd.DataFrame) -> pd.DataFrame:
    ok = per_split[per_split["status"] == "ok"].copy()
    if ok.empty:
        return pd.DataFrame()

    winners = []
    for anchor, part in ok.groupby("anchor_start", observed=False):
        part = part.dropna(subset=["log_loss", "brier", "accuracy"]).copy()
        if part.empty:
            continue

        best_log_loss = int(part.sort_values("log_loss", ascending=True).iloc[0]["window_days"])
        best_brier = int(part.sort_values("brier", ascending=True).iloc[0]["window_days"])
        best_accuracy = int(part.sort_values("accuracy", ascending=False).iloc[0]["window_days"])

        winners.append(
            {
                "anchor_start": anchor,
                "best_window_by_log_loss": best_log_loss,
                "best_window_by_brier": best_brier,
                "best_window_by_accuracy": best_accuracy,
            }
        )

    wins = pd.DataFrame(winners)
    if wins.empty:
        return wins

    counts = []
    for metric_col, metric_name in [
        ("best_window_by_log_loss", "log_loss"),
        ("best_window_by_brier", "brier"),
        ("best_window_by_accuracy", "accuracy"),
    ]:
        tmp = wins[metric_col].value_counts().sort_index()
        for window_days, n in tmp.items():
            counts.append(
                {
                    "metric": metric_name,
                    "window_days": int(window_days),
                    "anchor_wins": int(n),
                }
            )

    return pd.DataFrame(counts).sort_values(["metric", "anchor_wins"], ascending=[True, False]).reset_index(drop=True)


def main() -> None:
    args = parse_args()
    windows = parse_windows(args.windows_days)

    if args.test_hours <= 0 or args.step_hours <= 0:
        raise ValueError("test-hours and step-hours must be positive")

    df = load_and_prepare(Path(args.input))
    features = build_feature_list(df)
    if not features:
        raise ValueError("No usable features found in input file")

    anchors = make_anchors(
        df=df,
        windows_days=windows,
        test_hours=args.test_hours,
        step_hours=args.step_hours,
    )

    if not anchors:
        min_needed = max(windows) + (args.test_hours / 24.0)
        raise RuntimeError(
            "Not enough chronological span for requested test setup. "
            f"Need at least about {min_needed:.2f} days of data."
        )

    rows: List[Dict] = []
    pred_frames: List[pd.DataFrame] = []
    for anchor in anchors:
        for window_days in windows:
            split_row, split_preds = evaluate_window_on_anchor(
                df=df,
                features=features,
                window_days=window_days,
                anchor=anchor,
                test_hours=args.test_hours,
                calibration=args.calibration,
            )
            rows.append(split_row)
            if not split_preds.empty:
                pred_frames.append(split_preds)

    per_split = pd.DataFrame(rows)
    summary = build_summary(per_split)
    wins = build_anchor_wins(per_split)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    per_split_path = output_dir / "per_split_metrics.csv"
    summary_path = output_dir / "window_summary.csv"
    wins_path = output_dir / "anchor_wins.csv"
    per_window_bucket_dir = output_dir / "time_to_close_bucket_reports"

    per_split.to_csv(per_split_path, index=False)
    summary.to_csv(summary_path, index=False)
    wins.to_csv(wins_path, index=False)

    all_preds = pd.concat(pred_frames, ignore_index=True) if pred_frames else pd.DataFrame()
    if not all_preds.empty:
        per_window_bucket_dir.mkdir(parents=True, exist_ok=True)
        combined_rows: List[pd.DataFrame] = []
        for window_days in sorted(all_preds["window_days"].unique()):
            window_preds = all_preds[all_preds["window_days"] == window_days].copy()
            bucket_report = build_time_to_close_bucket_report(window_preds)
            if bucket_report.empty:
                continue
            bucket_report.insert(0, "window_days", int(window_days))
            combined_rows.append(bucket_report)
            bucket_path = per_window_bucket_dir / f"time_to_close_bucket_report_{int(window_days)}d.csv"
            bucket_report.to_csv(bucket_path, index=False)

        if combined_rows:
            combined_bucket = pd.concat(combined_rows, ignore_index=True)
            combined_bucket_path = output_dir / "time_to_close_bucket_report_by_window.csv"
            combined_bucket.to_csv(combined_bucket_path, index=False)

    ok_rows = int((per_split["status"] == "ok").sum())
    skipped_rows = int((per_split["status"] != "ok").sum())

    print(f"Rows loaded: {len(df):,}")
    print(f"Feature count: {len(features)}")
    print(f"Anchors evaluated: {len(anchors)}")
    print(f"Successful window-split evaluations: {ok_rows}")
    print(f"Skipped window-split evaluations: {skipped_rows}")
    print(f"Saved per-split metrics: {per_split_path.resolve()}")
    print(f"Saved summary: {summary_path.resolve()}")
    print(f"Saved anchor wins: {wins_path.resolve()}")
    if not all_preds.empty:
        print(f"Saved per-window bucket reports: {per_window_bucket_dir.resolve()}")
        print(
            "Saved combined bucket report: "
            f"{(output_dir / 'time_to_close_bucket_report_by_window.csv').resolve()}"
        )

    if not summary.empty:
        best_row = summary.iloc[0]
        print(
            "Best window by summary ranking: "
            f"{int(best_row['window_days'])} day(s) "
            f"(log_loss={best_row['log_loss_mean']:.4f}, "
            f"brier={best_row['brier_mean']:.4f}, "
            f"accuracy={best_row['accuracy_mean']:.4f})"
        )


if __name__ == "__main__":
    main()
