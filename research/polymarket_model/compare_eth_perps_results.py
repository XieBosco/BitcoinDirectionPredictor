"""Compare ETH+Perps enhanced model results with baseline and Polymarket."""

import pandas as pd
import numpy as np
from pathlib import Path

def main():
    # Load results
    results_dir = Path("research/polymarket_model/strict_7day_eth_perps_results")
    
    if not results_dir.exists():
        print("Results directory not found. Has the model run completed?")
        return
    
    overall_csv = results_dir / "overall_comparison.csv"
    splits_csv = results_dir / "per_split_metrics.csv"
    
    if not overall_csv.exists():
        print(f"Overall metrics file not found at {overall_csv}")
        return
    
    print("\n" + "="*80)
    print("ETH+PERPS ENHANCED MODEL - COMPARISON RESULTS")
    print("="*80)
    
    # Read overall metrics
    overall = pd.read_csv(overall_csv)
    
    print("\n[OVERALL METRICS SUMMARY]")
    print(overall.to_string(index=False))
    
    # Read split metrics if available
    if splits_csv.exists():
        splits = pd.read_csv(splits_csv)
        
        print("\n" + "="*80)
        print("PER-SPLIT COMPARISON (selected windows)")
        print("="*80)
        
        split_ids = splits["split_idx"].unique()
        print(f"\nTotal splits evaluated: {len(split_ids)}")
        
        # Show first few and last few splits
        for split_id in list(split_ids[:2]) + list(split_ids[-2:]):
            split_data = splits[splits["split_idx"] == split_id]
            print(f"\nSplit {split_id}:")
            print(f"  Test window: {split_data['test_start'].iloc[0]} to {split_data['test_end'].iloc[0]}")
            print(f"  Train rows: {split_data['train_rows'].iloc[0]:,}")
            print(f"  Test rows: {split_data['test_rows'].iloc[0]:,}")
            print()
            
            for _, row in split_data.iterrows():
                model = row["model"]
                auc = row["roc_auc"]
                ll = row["log_loss"]
                br = row["brier"]
                gap = row["calibration_gap_pp"]
                acc = row["accuracy"]
                
                print(f"  {model:35s} | AUC: {auc:.4f} | LL: {ll:.4f} | Brier: {br:.4f} | Gap: {gap:.2f}pp | Acc: {acc:.4f}")
    
    # Key metrics comparison
    print("\n" + "="*80)
    print("KEY FINDINGS")
    print("="*80)
    
    for metric in ["calibration_gap_pp", "log_loss", "brier", "roc_auc", "accuracy"]:
        col = f"{metric}_mean"
        if col not in overall.columns:
            continue
        
        print(f"\n{metric.upper()}:")
        for _, row in overall.iterrows():
            model = row["model"]
            mean_val = row[col]
            std_val = row[f"{metric}_std"]
            print(f"  {model:35s}: {mean_val:.6f} ± {std_val:.6f}")
    
    # Comparison of improvements
    print("\n" + "="*80)
    print("IMPROVEMENT ANALYSIS")
    print("="*80)
    
    baseline_gap = overall[overall["model"] == "7day_logistic_baseline"]["calibration_gap_pp_mean"].values[0]
    enhanced_gap = overall[overall["model"] == "7day_eth_perps_enhanced"]["calibration_gap_pp_mean"].values[0]
    polymarket_gap = overall[overall["model"] == "polymarket"]["calibration_gap_pp_mean"].values[0]
    
    gap_improvement = baseline_gap - enhanced_gap
    gap_vs_pm = polymarket_gap - enhanced_gap
    
    print(f"\nCalibration Gap (lower is better):")
    print(f"  Baseline 7-day:        {baseline_gap:.2f}pp")
    print(f"  ETH+Perps Enhanced:    {enhanced_gap:.2f}pp")
    print(f"  Polymarket:            {polymarket_gap:.2f}pp")
    print(f"\n  Improvement from baseline: {gap_improvement:.2f}pp ({100*gap_improvement/baseline_gap:.1f}%)")
    print(f"  Gap vs Polymarket:        {gap_vs_pm:.2f}pp")
    
    # Log Loss comparison
    baseline_ll = overall[overall["model"] == "7day_logistic_baseline"]["log_loss_mean"].values[0]
    enhanced_ll = overall[overall["model"] == "7day_eth_perps_enhanced"]["log_loss_mean"].values[0]
    
    ll_improvement = baseline_ll - enhanced_ll
    
    print(f"\nLog Loss (lower is better):")
    print(f"  Baseline 7-day:        {baseline_ll:.4f}")
    print(f"  ETH+Perps Enhanced:    {enhanced_ll:.4f}")
    print(f"  Improvement:           {ll_improvement:.4f} ({100*ll_improvement/baseline_ll:.1f}%)")
    
    # AUC comparison
    baseline_auc = overall[overall["model"] == "7day_logistic_baseline"]["roc_auc_mean"].values[0]
    enhanced_auc = overall[overall["model"] == "7day_eth_perps_enhanced"]["roc_auc_mean"].values[0]
    polymarket_auc = overall[overall["model"] == "polymarket"]["roc_auc_mean"].values[0]
    
    print(f"\nROC AUC (higher is better):")
    print(f"  Baseline 7-day:        {baseline_auc:.4f}")
    print(f"  ETH+Perps Enhanced:    {enhanced_auc:.4f}")
    print(f"  Polymarket:            {polymarket_auc:.4f}")
    
    print("\n" + "="*80)

if __name__ == "__main__":
    main()
