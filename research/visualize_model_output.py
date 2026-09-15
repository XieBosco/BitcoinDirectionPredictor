"""Visualize model output probabilities and realized 5-minute candle outcomes."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Visualize model probabilities versus realized candle direction "
            "from train_5m_close_direction_model.py output."
        )
    )
    parser.add_argument(
        "--input",
        default="research/btcusdt_1s_with_up_prob.csv",
        help="Input prediction CSV",
    )
    parser.add_argument(
        "--output",
        default="research/model_prediction_visualization.png",
        help="Output image path",
    )
    parser.add_argument(
        "--lookback-minutes",
        type=int,
        default=180,
        help="Recent window shown in top panel (minutes)",
    )
    return parser.parse_args()


def make_reliability_curve(df: pd.DataFrame, prob_col: str, n_bins: int = 10) -> pd.DataFrame:
    part = df[[prob_col, "target_up"]].dropna().copy()
    part["bin"] = pd.qcut(part[prob_col], q=n_bins, duplicates="drop")
    rel = (
        part.groupby("bin", observed=True)
        .agg(predicted_prob=(prob_col, "mean"), actual_up_rate=("target_up", "mean"), count=("target_up", "size"))
        .reset_index(drop=True)
    )
    return rel


def make_time_bucket_table(df: pd.DataFrame, prob_col: str) -> pd.DataFrame:
    bins = [0, 5, 15, 30, 60, 120, 180, 240, 300]
    labels = ["0-5s", "5-15s", "15-30s", "30-60s", "60-120s", "120-180s", "180-240s", "240-300s"]

    part = df[["seconds_to_5m_close", "target_up", prob_col]].dropna().copy()
    part["bucket"] = pd.cut(part["seconds_to_5m_close"], bins=bins, labels=labels, include_lowest=True)

    out = (
        part.groupby("bucket", observed=False)
        .agg(actual_up_rate=("target_up", "mean"), pred_up_prob=(prob_col, "mean"), count=("target_up", "size"))
        .reset_index()
    )
    out = out.dropna(subset=["bucket"])
    return out


def plot_dashboard(df: pd.DataFrame, output_path: Path, lookback_minutes: int) -> None:
    required_cols = ["open_time", "target_up", "prob_5m_close_up_oof", "candle_5m_start", "seconds_to_5m_close"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df = df.copy()
    df["open_time"] = pd.to_datetime(df["open_time"], utc=True)
    df = df.sort_values("open_time").reset_index(drop=True)

    end_time = df["open_time"].iloc[-1]
    start_time = end_time - pd.Timedelta(minutes=lookback_minutes)
    recent = df[df["open_time"] >= start_time].copy()
    if recent.empty:
        recent = df.copy()

    prob_col = "prob_5m_close_up_oof"
    y_true = df["target_up"].astype(int).values
    y_prob = df[prob_col].astype(float).values
    auc = roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) == 2 else np.nan

    reliability = make_reliability_curve(df, prob_col=prob_col)
    bucket_table = make_time_bucket_table(df, prob_col=prob_col)

    fig, axes = plt.subplots(2, 2, figsize=(18, 11))
    ax_prob = axes[0, 0]
    ax_cal = axes[0, 1]
    ax_hist = axes[1, 0]
    ax_bucket = axes[1, 1]

    up_mask = recent["target_up"] == 1
    down_mask = ~up_mask
    ax_prob.scatter(
        recent.loc[down_mask, "open_time"],
        recent.loc[down_mask, prob_col],
        s=7,
        alpha=0.6,
        color="#d62828",
        label="Actual candle down",
    )
    ax_prob.scatter(
        recent.loc[up_mask, "open_time"],
        recent.loc[up_mask, prob_col],
        s=7,
        alpha=0.6,
        color="#2a9d8f",
        label="Actual candle up",
    )
    ax_prob.axhline(0.5, color="#444444", linestyle="--", linewidth=0.9, label="0.5 threshold")
    ax_prob.set_ylim(-0.02, 1.02)
    ax_prob.set_title(f"Recent Probabilities (last {lookback_minutes} min) | ROC AUC={auc:.3f}")
    ax_prob.set_ylabel("Predicted P(5m close up)")
    ax_prob.grid(alpha=0.2)
    ax_prob.legend(loc="lower right")

    locator = mdates.AutoDateLocator(minticks=6, maxticks=10)
    formatter = mdates.ConciseDateFormatter(locator)
    ax_prob.xaxis.set_major_locator(locator)
    ax_prob.xaxis.set_major_formatter(formatter)

    ax_cal.plot([0, 1], [0, 1], linestyle="--", color="#666666", label="Perfect calibration")
    ax_cal.plot(
        reliability["predicted_prob"],
        reliability["actual_up_rate"],
        marker="o",
        linewidth=1.5,
        color="#1d3557",
        label="Model reliability",
    )
    ax_cal.set_xlabel("Mean predicted up probability")
    ax_cal.set_ylabel("Observed up frequency")
    ax_cal.set_title("Calibration Curve (quantile bins)")
    ax_cal.grid(alpha=0.2)
    ax_cal.legend(loc="lower right")

    ax_hist.hist(df.loc[df["target_up"] == 0, prob_col], bins=30, alpha=0.6, color="#d62828", label="Actual down")
    ax_hist.hist(df.loc[df["target_up"] == 1, prob_col], bins=30, alpha=0.6, color="#2a9d8f", label="Actual up")
    ax_hist.set_xlabel("Predicted up probability")
    ax_hist.set_ylabel("Count")
    ax_hist.set_title("Probability Distribution by True Class")
    ax_hist.grid(alpha=0.2)
    ax_hist.legend(loc="upper center")

    x = np.arange(len(bucket_table))
    ax_bucket.plot(x, bucket_table["actual_up_rate"], marker="o", color="#d62828", label="Actual up rate")
    ax_bucket.plot(x, bucket_table["pred_up_prob"], marker="o", color="#2a9d8f", label="Avg predicted up prob")
    ax_bucket.set_xticks(x)
    ax_bucket.set_xticklabels(bucket_table["bucket"], rotation=30)
    ax_bucket.set_ylim(0, 1)
    ax_bucket.set_ylabel("Probability")
    ax_bucket.set_title("Time-to-Close: Actual vs Predicted")
    ax_bucket.grid(alpha=0.2)
    ax_bucket.legend(loc="best")

    title = "Model Output Diagnostics: Per-Second Up Probability vs Realized 5m Direction"
    fig.suptitle(title, fontsize=14, y=1.02)
    plt.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=190, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    df = pd.read_csv(input_path)
    plot_dashboard(df, output_path=output_path, lookback_minutes=args.lookback_minutes)
    print(f"Saved model visualization to {output_path.resolve()}")


if __name__ == "__main__":
    main()
