"""Visualize enriched 1-second BTCUSDT market data from CSV.

Produces a static PNG with:
- price and rolling VWAP
- instantaneous volatility (bps)
- market regime timeline
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd


REGIME_COLORS = {
    "calm": "#2e8b57",
    "normal": "#6c757d",
    "high_vol_chop": "#f4a261",
    "volatile_trend_up": "#1971c2",
    "volatile_trend_down": "#d62828",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a chart from the enriched BTCUSDT 1s CSV dataset."
    )
    parser.add_argument(
        "--input",
        default="btcusdt_1s_last24h.csv",
        help="Input CSV produced by fetch_data.py",
    )
    parser.add_argument(
        "--output",
        default="btcusdt_1s_visualization.png",
        help="Output image file path",
    )
    return parser.parse_args()


def load_data(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "open_time" not in df.columns:
        raise ValueError("CSV is missing required 'open_time' column")

    df["open_time"] = pd.to_datetime(df["open_time"], utc=True)
    df = df.sort_values("open_time").reset_index(drop=True)
    return df


def plot_dataset(df: pd.DataFrame, output_path: Path) -> None:
    required_cols = ["close", "inst_vol_60s_bps", "market_regime"]
    missing_cols = [c for c in required_cols if c not in df.columns]
    if missing_cols:
        raise ValueError(f"CSV is missing required columns: {missing_cols}")

    fig, axes = plt.subplots(
        3,
        1,
        figsize=(16, 10),
        sharex=True,
        gridspec_kw={"height_ratios": [2.2, 1.4, 0.8]},
    )
    ax_price, ax_vol, ax_regime = axes

    # Price panel.
    ax_price.plot(df["open_time"], df["close"], color="#0b7285", linewidth=0.8, label="Close")
    if "vwap" in df.columns:
        ax_price.plot(df["open_time"], df["vwap"], color="#faa307", linewidth=0.9, label="VWAP")
    ax_price.set_ylabel("Price (USDT)")
    ax_price.set_title("BTCUSDT 1s Data: Price, Instantaneous Volatility, and Regime")
    ax_price.grid(alpha=0.2)
    ax_price.legend(loc="upper left")

    # Volatility panel.
    ax_vol.plot(
        df["open_time"],
        df["inst_vol_60s_bps"],
        color="#9d4edd",
        linewidth=0.8,
        label="Inst. Vol (60s, bps)",
    )
    ax_vol.set_ylabel("Volatility (bps)")
    ax_vol.grid(alpha=0.2)
    ax_vol.legend(loc="upper left")

    # Regime panel.
    regime_order = list(REGIME_COLORS.keys())
    y_map = {name: idx for idx, name in enumerate(regime_order)}
    y_values = df["market_regime"].map(y_map).fillna(y_map["normal"])
    point_colors = df["market_regime"].map(REGIME_COLORS).fillna(REGIME_COLORS["normal"])
    ax_regime.scatter(df["open_time"], y_values, c=point_colors, s=4, alpha=0.9)
    ax_regime.set_yticks(list(y_map.values()))
    ax_regime.set_yticklabels(regime_order)
    ax_regime.set_ylabel("Regime")
    ax_regime.grid(alpha=0.2)

    locator = mdates.AutoDateLocator()
    formatter = mdates.ConciseDateFormatter(locator)
    ax_regime.xaxis.set_major_locator(locator)
    ax_regime.xaxis.set_major_formatter(formatter)
    ax_regime.set_xlabel("UTC Time")

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)

    if not input_path.exists():
        raise FileNotFoundError(f"Input CSV not found: {input_path}")

    df = load_data(input_path)
    plot_dataset(df, output_path)
    print(f"Visualization saved to {output_path.resolve()}")


if __name__ == "__main__":
    main()
