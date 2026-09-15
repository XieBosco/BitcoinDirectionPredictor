import pandas as pd
import numpy as np

df = pd.read_csv('research/time_to_close_bucket_report.csv')
row_120_180 = df[df['time_to_close_bucket'] == '120-180s'].iloc[0]
row_180_240 = df[df['time_to_close_bucket'] == '180-240s'].iloc[0]

print("\n=== LAST 2-3 MINUTES OF 5-MINUTE CANDLE ===")
print("(120-180 seconds remaining)\n")
print(f"Actual candle close up rate:   {row_120_180['actual_up_rate']*100:.2f}%")
print(f"Model avg predicted up prob:   {row_120_180['avg_pred_up_prob']*100:.2f}%")
print(f"Average prediction difference: {(row_120_180['actual_up_rate'] - row_120_180['avg_pred_up_prob'])*100:.2f} pp")
print(f"Accuracy:                      {row_120_180['accuracy']*100:.2f}%")
print(f"Brier score:                   {row_120_180['brier']:.4f}")
print(f"ROC AUC:                       {row_120_180['roc_auc']:.4f}")
print(f"Sample size:                   {int(row_120_180['samples']):,} predictions\n")

mean_sq_error = row_120_180['brier']
estimated_mae = np.sqrt(mean_sq_error)
print(f"Estimated typical error: ±{estimated_mae*100:.2f} percentage points")
print(f"If model predicts 50% up: actual likely between {(0.5-estimated_mae)*100:.2f}% and {(0.5+estimated_mae)*100:.2f}%")

print("\n=== ALSO CHECK 180-240 SECONDS (LAST 3-4 MINUTES) ===\n")
print(f"Actual candle close up rate:   {row_180_240['actual_up_rate']*100:.2f}%")
print(f"Model avg predicted up prob:   {row_180_240['avg_pred_up_prob']*100:.2f}%")
print(f"Average prediction difference: {(row_180_240['actual_up_rate'] - row_180_240['avg_pred_up_prob'])*100:.2f} pp")
print(f"Accuracy:                      {row_180_240['accuracy']*100:.2f}%")
print(f"Brier score:                   {row_180_240['brier']:.4f}")

mean_sq_error_180 = row_180_240['brier']
estimated_mae_180 = np.sqrt(mean_sq_error_180)
print(f"\nEstimated typical error: ±{estimated_mae_180*100:.2f} percentage points")
