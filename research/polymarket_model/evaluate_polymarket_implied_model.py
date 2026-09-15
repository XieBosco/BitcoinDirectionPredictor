import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import GroupKFold


def load_data(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)

    # Ensure numeric fields are parsed
    numeric_cols = [
        "start_time",
        "elapsed",
        "ask_YES",
        "bid_YES",
        "ask_NO",
        "bid_NO",
        "timestamp_log",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Parse timestamps from unix seconds
    df["timestamp"] = pd.to_datetime(df["timestamp_log"], unit="s", utc=True)
    df["candle_start_ts"] = pd.to_datetime(df["start_time"], unit="s", utc=True)

    # Binary label for whether 5m candle closes up
    df["label"] = (df["winner"].str.lower() == "up").astype(int)

    return df.dropna(subset=["elapsed", "ask_YES", "bid_YES", "timestamp_log", "label"])


def add_implied_probability(df: pd.DataFrame) -> pd.DataFrame:
    # Midprice of YES best bid/ask represents implied probability of BTC closing up.
    # Example: yes bid=0.74 and yes ask=0.76 -> implied prob = 0.75
    df = df.copy()
    df["implied_prob"] = (df["bid_YES"] + df["ask_YES"]) / 2.0
    df["implied_prob"] = df["implied_prob"].clip(0.0, 1.0)
    return df


def upsample_to_1s(df: pd.DataFrame) -> pd.DataFrame:
    # Data arrives every 2s; convert to 1s by carry-forward previous implied probability
    all_rows = []

    for slug, g in df.groupby("slug", sort=False):
        g = g.sort_values("timestamp").copy()
        start_ts = g["timestamp"].min()
        end_ts = g["timestamp"].max()

        full_index = pd.date_range(start=start_ts, end=end_ts, freq="1s", tz="UTC")
        u = g.set_index("timestamp").reindex(full_index)

        # Carry forward all contextual columns; backfill very first row if needed
        for col in [
            "slug",
            "start_time",
            "candle_start_ts",
            "winner",
            "label",
            "bid_YES",
            "ask_YES",
            "bid_NO",
            "ask_NO",
            "implied_prob",
            "elapsed",
        ]:
            if col in u.columns:
                u[col] = u[col].ffill().bfill()

        u = u.reset_index().rename(columns={"index": "timestamp"})

        # Recompute elapsed from known candle start so exact 1-second grid is valid
        u["elapsed"] = (u["timestamp"].astype("int64") // 10**9 - u["start_time"]).astype(int)
        all_rows.append(u)

    out = pd.concat(all_rows, ignore_index=True)
    return out


def metrics_for_frame(frame: pd.DataFrame) -> dict:
    y_true = frame["label"].values.astype(int)
    y_prob = frame["implied_prob"].values.astype(float)
    y_pred = (y_prob >= 0.5).astype(int)

    return {
        "n_rows": int(len(frame)),
        "n_candles": int(frame["slug"].nunique()),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "log_loss": float(log_loss(y_true, y_prob, labels=[0, 1])),
        "brier": float(brier_score_loss(y_true, y_prob)),
        "calibration_gap_pp": float(np.mean(np.abs(y_prob - y_true)) * 100.0),
        "mean_implied_prob": float(np.mean(y_prob)),
        "mean_actual_prob": float(np.mean(y_true)),
    }


def cross_validate_by_candle(df: pd.DataFrame, n_splits: int = 5) -> pd.DataFrame:
    # Group K-Fold by candle slug to mirror no-leakage evaluation
    candles = (
        df[["slug", "label"]]
        .drop_duplicates("slug")
        .reset_index(drop=True)
    )

    if len(candles) < n_splits:
        n_splits = max(2, len(candles) // 2)

    gkf = GroupKFold(n_splits=n_splits)
    groups = candles["slug"].values

    rows = []
    for fold, (_, test_idx) in enumerate(gkf.split(candles, candles["label"], groups=groups), start=1):
        test_slugs = set(candles.iloc[test_idx]["slug"])
        test_frame = df[df["slug"].isin(test_slugs)].copy()
        m = metrics_for_frame(test_frame)
        m["fold"] = fold
        rows.append(m)

    return pd.DataFrame(rows)


def evaluate_by_elapsed_window(df: pd.DataFrame) -> pd.DataFrame:
    # Evaluate at each elapsed second to compare implied vs actual probability over time
    rows = []
    for sec, g in df.groupby("elapsed"):
        if sec < 100 or sec > 290:
            continue
        if g["label"].nunique() < 2:
            continue
        m = metrics_for_frame(g)
        m["elapsed"] = int(sec)
        rows.append(m)
    return pd.DataFrame(rows).sort_values("elapsed")


def evaluate_by_buckets(df: pd.DataFrame) -> pd.DataFrame:
    bins = [100, 130, 160, 190, 220, 250, 291]
    labels = ["100-129", "130-159", "160-189", "190-219", "220-249", "250-290"]

    w = df.copy()
    w["window_bucket"] = pd.cut(w["elapsed"], bins=bins, labels=labels, right=False)

    rows = []
    for b, g in w.groupby("window_bucket", observed=False):
        if pd.isna(b) or len(g) == 0 or g["label"].nunique() < 2:
            continue
        m = metrics_for_frame(g)
        m["bucket"] = str(b)
        rows.append(m)

    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Polymarket implied probabilities as predictor")
    parser.add_argument(
        "--input-csv",
        type=str,
        default="market_data_2sec_weekly5_with_resolutions.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="eval_results",
    )
    args = parser.parse_args()

    input_csv = Path(args.input_csv)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = load_data(input_csv)
    df = add_implied_probability(df)
    df_1s = upsample_to_1s(df)

    # Keep only the requested time window within each 5m candle
    eval_df = df_1s[(df_1s["elapsed"] >= 100) & (df_1s["elapsed"] <= 290)].copy()

    overall = metrics_for_frame(eval_df)
    fold_df = cross_validate_by_candle(eval_df, n_splits=5)
    elapsed_df = evaluate_by_elapsed_window(eval_df)
    bucket_df = evaluate_by_buckets(eval_df)

    # Calibration table (predicted deciles vs actual)
    cal = eval_df.copy()
    cal["prob_bin"] = pd.qcut(cal["implied_prob"], q=10, duplicates="drop")
    calibration_df = (
        cal.groupby("prob_bin", observed=False)
        .agg(
            n=("label", "size"),
            mean_pred=("implied_prob", "mean"),
            mean_actual=("label", "mean"),
        )
        .reset_index()
    )
    calibration_df["gap_pp"] = (calibration_df["mean_pred"] - calibration_df["mean_actual"]).abs() * 100

    pd.DataFrame([overall]).to_csv(output_dir / "overall_metrics.csv", index=False)
    fold_df.to_csv(output_dir / "groupkfold_metrics.csv", index=False)
    elapsed_df.to_csv(output_dir / "metrics_by_elapsed_second.csv", index=False)
    bucket_df.to_csv(output_dir / "metrics_by_window_bucket.csv", index=False)
    calibration_df.to_csv(output_dir / "calibration_table.csv", index=False)

    print("=== Polymarket Implied Probability Evaluation ===")
    print(f"Input rows (2s): {len(df):,}")
    print(f"Upsampled rows (1s): {len(df_1s):,}")
    print(f"Eval rows (100-290s): {len(eval_df):,}")
    print(f"Unique 5m candles: {eval_df['slug'].nunique():,}")
    print("\nOverall Metrics (100-290s window):")
    for k in [
        "accuracy",
        "roc_auc",
        "log_loss",
        "brier",
        "calibration_gap_pp",
        "mean_implied_prob",
        "mean_actual_prob",
    ]:
        print(f"  {k}: {overall[k]:.6f}")

    if not fold_df.empty:
        print("\nGroupKFold Mean ± Std:")
        for k in ["accuracy", "roc_auc", "log_loss", "brier", "calibration_gap_pp"]:
            print(f"  {k}: {fold_df[k].mean():.6f} ± {fold_df[k].std():.6f}")

    print("\nSaved outputs:")
    for name in [
        "overall_metrics.csv",
        "groupkfold_metrics.csv",
        "metrics_by_elapsed_second.csv",
        "metrics_by_window_bucket.csv",
        "calibration_table.csv",
    ]:
        print(f"  - {output_dir / name}")


if __name__ == "__main__":
    main()
