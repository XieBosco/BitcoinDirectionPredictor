"""Visualize model probability predictions for the most recent 5-minute candle window."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Visualize model up-probability predictions within the most recent "
            "5-minute candle from train_5m_close_direction_model.py output."
        )
    )
    parser.add_argument(
        "--input",
        default="research/btcusdt_1s_with_up_prob.csv",
        help="Input prediction CSV",
    )
    parser.add_argument(
        "--output",
        default="research/model_recent_5min.png",
        help="Output image path",
    )
    return parser.parse_args()


def load_and_slice(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = ["open_time", "close", "prob_5m_close_up_oof", "target_up", "inst_vol_60s_bps"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df["open_time"] = pd.to_datetime(df["open_time"], utc=True)
    df = df.sort_values("open_time").reset_index(drop=True)

    end_time = df["open_time"].iloc[-1]
    start_time = end_time - pd.Timedelta(minutes=5)
    recent = df[df["open_time"] >= start_time].copy()

    if recent.empty:
        raise ValueError("No rows found in the most recent 5-minute window")

    return recent


def plot_recent_5min_probabilities(df_recent: pd.DataFrame, output_path: Path) -> None:
    fig, axes = plt.subplots(
        3,
        1,
        figsize=(16, 10),
        sharex=True,
        gridspec_kw={"height_ratios": [2.0, 1.5, 0.8]},
    )
    ax_prob, ax_vol, ax_outcome = axes

    up_mask = df_recent["target_up"] == 1
    down_mask = ~up_mask

    ax_prob.scatter(
        df_recent.loc[down_mask, "open_time"],
        df_recent.loc[down_mask, "prob_5m_close_up_oof"],
        s=15,
        alpha=0.7,
        color="#d62828",
        label="Actual candle down",
        zorder=3,
    )
    ax_prob.scatter(
        df_recent.loc[up_mask, "open_time"],
        df_recent.loc[up_mask, "prob_5m_close_up_oof"],
        s=15,
        alpha=0.7,
        color="#2a9d8f",
        label="Actual candle up",
        zorder=3,
    )

    ax_prob.plot(
        df_recent["open_time"],
        df_recent["prob_5m_close_up_oof"],
        color="#264653",
        linewidth=1.2,
        alpha=0.4,
        zorder=1,
    )

    ax_prob.axhline(0.5, color="#888888", linestyle="--", linewidth=0.9, alpha=0.6, label="50% threshold")
    ax_prob.fill_between(
        df_recent["open_time"],
        0,
        1,
        alpha=0.05,
        color="#2a9d8f",
        label="Up-probability region",
    )

    ax_prob.set_ylim(-0.05, 1.05)
    ax_prob.set_ylabel("P(5m candle closes up)")
    ax_prob.set_title("Recent 5-Minute Window: Model Up Probability")
    ax_prob.grid(alpha=0.25, linestyle=":")
    ax_prob.legend(loc="upper left", fontsize=9)

    ax_vol.plot(
        df_recent["open_time"],
        df_recent["inst_vol_60s_bps"],
        color="#9d4edd",
        linewidth=1.2,
        label="Inst. Vol (60s, bps)",
    )
    ax_vol.fill_between(
        df_recent["open_time"],
        0,
        df_recent["inst_vol_60s_bps"],
        alpha=0.3,
        color="#9d4edd",
    )
    ax_vol.set_ylabel("Volatility (bps)")
    ax_vol.grid(alpha=0.25, linestyle=":")
    ax_vol.legend(loc="upper left", fontsize=9)

    ax_outcome.scatter(
        df_recent.loc[down_mask, "open_time"],
        np.zeros(len(df_recent.loc[down_mask])),
        s=20,
        color="#d62828",
        marker="v",
        label="Down",
        zorder=3,
    )
    ax_outcome.scatter(
        df_recent.loc[up_mask, "open_time"],
        np.ones(len(df_recent.loc[up_mask])),
        s=20,
        color="#2a9d8f",
        marker="^",
        label="Up",
        zorder=3,
    )
    ax_outcome.set_yticks([0, 1])
    ax_outcome.set_yticklabels(["Down", "Up"])
    ax_outcome.set_ylabel("Outcome")
    ax_outcome.set_ylim(-0.5, 1.5)
    ax_outcome.grid(alpha=0.25, linestyle=":")
    ax_outcome.legend(loc="upper left", fontsize=9)

    locator = mdates.AutoDateLocator(minticks=6, maxticks=12)
    formatter = mdates.ConciseDateFormatter(locator)
    ax_outcome.xaxis.set_major_locator(locator)
    ax_outcome.xaxis.set_major_formatter(formatter)
    ax_outcome.set_xlabel("UTC Time")

    fig.suptitle("Recent 5-Minute Window: Model Probabilities + Volatility + Actual Outcome", fontsize=13, y=0.995)
    plt.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    recent = load_and_slice(input_path)
    plot_recent_5min_probabilities(recent, output_path=output_path)
    print(f"Saved recent 5-minute model visualization to {output_path.resolve()}")
    print(f"Rows displayed: {len(recent)}")


if __name__ == "__main__":
    main()
