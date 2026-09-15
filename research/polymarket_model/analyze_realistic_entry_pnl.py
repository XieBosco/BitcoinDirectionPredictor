from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def load_orderbook_1s(polymarket_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(polymarket_csv)
    df["timestamp"] = pd.to_datetime(df["timestamp_log"], unit="s", utc=True)

    keep_cols = ["slug", "timestamp", "ask_YES", "bid_YES", "ask_NO", "bid_NO", "start_time"]
    for c in ["ask_YES", "bid_YES", "ask_NO", "bid_NO", "start_time"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df[keep_cols].copy()

    upsampled = []
    for slug, g in df.groupby("slug", sort=False):
        g = g.sort_values("timestamp").copy()
        idx = pd.date_range(start=g["timestamp"].min(), end=g["timestamp"].max(), freq="1s", tz="UTC")
        u = g.set_index("timestamp").reindex(idx)
        for col in ["slug", "start_time", "ask_YES", "bid_YES", "ask_NO", "bid_NO"]:
            u[col] = u[col].ffill().bfill()
        u = u.reset_index().rename(columns={"index": "timestamp"})
        upsampled.append(u[["slug", "timestamp", "ask_YES", "bid_YES", "ask_NO", "bid_NO"]])

    ob = pd.concat(upsampled, ignore_index=True)
    return ob


def compute_realistic_pnl(row_df: pd.DataFrame, threshold_pp: int, fee_per_side: float = 0.0) -> dict:
    th = threshold_pp / 100.0

    p_model = np.clip(row_df["prob_7d_baseline"].to_numpy(float), 1e-6, 1 - 1e-6)
    y = row_df["y_true"].to_numpy(int)
    ask_yes = np.clip(row_df["ask_YES"].to_numpy(float), 1e-6, 1 - 1e-6)
    ask_no = np.clip(row_df["ask_NO"].to_numpy(float), 1e-6, 1 - 1e-6)

    edge_yes = p_model - ask_yes
    edge_no = (1.0 - p_model) - ask_no

    buy_yes = edge_yes >= th
    buy_no = (edge_no >= th) & (~buy_yes)
    trade = buy_yes | buy_no

    pnl = np.zeros(len(row_df), dtype=float)
    pnl[buy_yes] = y[buy_yes] - ask_yes[buy_yes] - fee_per_side
    pnl[buy_no] = (1 - y[buy_no]) - ask_no[buy_no] - fee_per_side

    trades = int(trade.sum())
    return {
        "threshold_pp": threshold_pp,
        "trades": trades,
        "trade_rate": float(trades / len(row_df)),
        "buy_yes_trades": int(buy_yes.sum()),
        "buy_no_trades": int(buy_no.sum()),
        "total_net_pnl_per_1_notional": float(pnl[trade].sum()),
        "avg_net_pnl_per_trade": float(pnl[trade].mean()) if trades else np.nan,
        "win_rate": float((pnl[trade] > 0).mean()) if trades else np.nan,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate realistic entry PnL using ask-side fills")
    parser.add_argument(
        "--row-predictions",
        default="research/polymarket_model/strict_7day_vs_pm_results/row_level_predictions.csv",
    )
    parser.add_argument(
        "--polymarket-csv",
        default="research/polymarket_model/market_data_2sec_weekly5_with_resolutions.csv",
    )
    parser.add_argument(
        "--output",
        default="research/polymarket_model/strict_7day_vs_pm_results/realistic_entry_pnl.csv",
    )
    parser.add_argument(
        "--fee-per-side",
        type=float,
        default=0.0,
    )
    args = parser.parse_args()

    row_path = Path(args.row_predictions)
    pm_path = Path(args.polymarket_csv)

    row_df = pd.read_csv(row_path)
    row_df["timestamp"] = pd.to_datetime(row_df["timestamp"], utc=True)

    ob = load_orderbook_1s(pm_path)

    merged = row_df.merge(ob, on=["slug", "timestamp"], how="inner")
    merged = merged.dropna(subset=["ask_YES", "ask_NO", "prob_7d_baseline", "y_true"]).copy()

    thresholds = [0, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    rows = [compute_realistic_pnl(merged, t, fee_per_side=args.fee_per_side) for t in thresholds]
    out = pd.DataFrame(rows)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)

    print(f"Rows in merged realistic dataset: {len(merged):,}")
    print(out.to_string(index=False))
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
