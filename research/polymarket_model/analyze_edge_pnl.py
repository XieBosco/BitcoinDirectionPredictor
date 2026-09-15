from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd


def compute_trade_pnl_per_dollar(y_true: np.ndarray, market_prob: np.ndarray, model_prob: np.ndarray, threshold: float, fee_per_side: float) -> Dict[str, float]:
    """
    Binary contract PnL per $1 notional with thresholded edge execution.

    Trade rule:
    - Buy YES when model_prob - market_prob >= threshold
      PnL = y_true - market_prob - fee_per_side
    - Buy NO when model_prob - market_prob <= -threshold
      PnL = market_prob - y_true - fee_per_side

    Returns aggregate and per-trade statistics.
    """
    edge = model_prob - market_prob
    yes_mask = edge >= threshold
    no_mask = edge <= -threshold
    trade_mask = yes_mask | no_mask

    if not np.any(trade_mask):
        return {
            "trades": 0,
            "trade_rate": 0.0,
            "gross_pnl": 0.0,
            "net_pnl": 0.0,
            "avg_net_pnl_per_trade": np.nan,
            "avg_edge_abs": np.nan,
            "win_rate": np.nan,
        }

    pnl = np.zeros_like(market_prob, dtype=float)
    pnl[yes_mask] = y_true[yes_mask] - market_prob[yes_mask] - fee_per_side
    pnl[no_mask] = market_prob[no_mask] - y_true[no_mask] - fee_per_side

    gross = np.zeros_like(market_prob, dtype=float)
    gross[yes_mask] = y_true[yes_mask] - market_prob[yes_mask]
    gross[no_mask] = market_prob[no_mask] - y_true[no_mask]

    traded = pnl[trade_mask]
    traded_gross = gross[trade_mask]
    traded_edge_abs = np.abs(edge[trade_mask])

    return {
        "trades": int(trade_mask.sum()),
        "trade_rate": float(trade_mask.mean()),
        "gross_pnl": float(traded_gross.sum()),
        "net_pnl": float(traded.sum()),
        "avg_net_pnl_per_trade": float(traded.mean()),
        "avg_edge_abs": float(traded_edge_abs.mean()),
        "win_rate": float((traded > 0).mean()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze PnL from model-vs-market probability edge")
    parser.add_argument(
        "--predictions",
        default="research/polymarket_model/strict_7day_vs_pm_results/row_level_predictions.csv",
        help="Path to row-level predictions CSV",
    )
    parser.add_argument(
        "--output-dir",
        default="research/polymarket_model/strict_7day_vs_pm_results",
        help="Directory to save analysis CSV files",
    )
    parser.add_argument(
        "--fee-per-side",
        type=float,
        default=0.0,
        help="Fixed fee per trade side in $ per $1 notional",
    )
    args = parser.parse_args()

    path = Path(args.predictions)
    if not path.exists():
        raise FileNotFoundError(f"Missing predictions file: {path}")

    df = pd.read_csv(path)
    required = ["y_true", "prob_polymarket", "prob_7d_baseline", "prob_7d_enhanced"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    y_true = df["y_true"].astype(int).to_numpy()
    p_mkt = df["prob_polymarket"].astype(float).to_numpy()

    thresholds_pp = [2] + list(range(3, 11))
    models = {
        "7day_baseline": df["prob_7d_baseline"].astype(float).to_numpy(),
        "7day_enhanced": df["prob_7d_enhanced"].astype(float).to_numpy(),
    }

    rows: List[Dict[str, float]] = []
    for model_name, p_model in models.items():
        for pp in thresholds_pp:
            threshold = pp / 100.0
            stats = compute_trade_pnl_per_dollar(
                y_true=y_true,
                market_prob=p_mkt,
                model_prob=p_model,
                threshold=threshold,
                fee_per_side=args.fee_per_side,
            )
            rows.append(
                {
                    "model": model_name,
                    "edge_threshold_pp": pp,
                    "fee_per_side": args.fee_per_side,
                    **stats,
                }
            )

    out = pd.DataFrame(rows)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "edge_threshold_pnl.csv"
    out.to_csv(out_path, index=False)

    # Compact summary focused on user ask.
    ask_rows = out[out["edge_threshold_pp"].isin([2, 3, 4, 5, 6, 7, 8, 9, 10])].copy()
    ask_rows.to_csv(out_dir / "edge_threshold_pnl_2_to_10pp.csv", index=False)

    print("=== EDGE THRESHOLD PNL ANALYSIS ===")
    print(f"Rows analyzed: {len(df):,}")
    print(f"Fee per side: {args.fee_per_side:.6f}")
    print("\nPnL by threshold (2pp and 3-10pp):")
    print(ask_rows.to_string(index=False))
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
