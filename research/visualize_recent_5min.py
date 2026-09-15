"""Visualize only the most recent 5-minute interval from enriched 1s BTCUSDT data."""

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
        description="Create a chart for the most recent 5-minute interval from BTCUSDT 1s CSV."
    )
    parser.add_argument(
        "--input",
        default="btcusdt_1s_last24h.csv",
        help="Input CSV produced by fetch_data.py",
    )
    parser.add_argument(
        "--output",
        default="btcusdt_1s_recent5m.png",
        help="Output image file path",
    )
    return parser.parse_args()


def load_recent_window(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "open_time" not in df.columns:
        raise ValueError("CSV is missing required 'open_time' column")

    required_cols = ["close", "inst_vol_60s_bps", "market_regime"]
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(f"CSV is missing required columns: {missing}")

    df["open_time"] = pd.to_datetime(df["open_time"], utc=True)
    df = df.sort_values("open_time").reset_index(drop=True)

    end_time = df["open_time"].iloc[-1]
    start_time = end_time - pd.Timedelta(minutes=5)
    recent = df[df["open_time"] >= start_time].copy()

    if recent.empty:
        raise ValueError("No rows found in the most recent 5-minute interval")

    return recent


def plot_recent_window(df_recent: pd.DataFrame, output_path: Path) -> None:
    fig, axes = plt.subplots(
        3,
        1,
        figsize=(16, 10),
        sharex=True,
        gridspec_kw={"height_ratios": [2.2, 1.4, 0.8]},
    )
    ax_price, ax_vol, ax_regime = axes

    ax_price.plot(
        df_recent["open_time"],
        df_recent["close"],
        color="#0b7285",
        linewidth=1.1,
        label="Close",
    )
    if "vwap" in df_recent.columns:
        ax_price.plot(
            df_recent["open_time"],
            df_recent["vwap"],
            color="#faa307",
            linewidth=1.1,
            label="VWAP",
        )
    ax_price.set_ylabel("Price (USDT)")
    ax_price.set_title("BTCUSDT Most Recent 5 Minutes: Price, Volatility, and Regime")
    ax_price.grid(alpha=0.2)
    ax_price.legend(loc="upper left")

    ax_vol.plot(
        df_recent["open_time"],
        df_recent["inst_vol_60s_bps"],
        color="#9d4edd",
        linewidth=1.1,
        label="Inst. Vol (60s, bps)",
    )
    ax_vol.set_ylabel("Volatility (bps)")
    ax_vol.grid(alpha=0.2)
    ax_vol.legend(loc="upper left")

    regime_order = list(REGIME_COLORS.keys())
    y_map = {name: idx for idx, name in enumerate(regime_order)}
    y_values = df_recent["market_regime"].map(y_map).fillna(y_map["normal"])
    point_colors = df_recent["market_regime"].map(REGIME_COLORS).fillna(REGIME_COLORS["normal"])
    ax_regime.scatter(df_recent["open_time"], y_values, c=point_colors, s=12, alpha=0.95)
    ax_regime.set_yticks(list(y_map.values()))
    ax_regime.set_yticklabels(regime_order)
    ax_regime.set_ylabel("Regime")
    ax_regime.grid(alpha=0.2)

    locator = mdates.AutoDateLocator(minticks=6, maxticks=12)
    formatter = mdates.ConciseDateFormatter(locator)
    ax_regime.xaxis.set_major_locator(locator)
    ax_regime.xaxis.set_major_formatter(formatter)
    ax_regime.set_xlabel("UTC Time")

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)

    if not input_path.exists():
        raise FileNotFoundError(f"Input CSV not found: {input_path}")

    recent = load_recent_window(input_path)
    plot_recent_window(recent, output_path)
    print(f"5-minute visualization saved to {output_path.resolve()}")


if __name__ == "__main__":
    main()
