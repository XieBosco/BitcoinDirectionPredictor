import pandas as pd
import numpy as np

df = pd.read_csv('research/time_to_close_bucket_report.csv')
last_60s = df[df['time_to_close_bucket'].isin(['0-5s', '5-15s', '15-30s', '30-60s'])]

print('\nLAST 60 SECONDS OF 5-MINUTE CANDLE')
print('='*80)
print(last_60s[['time_to_close_bucket', 'samples', 'accuracy', 'brier', 'roc_auc']].to_string(index=False))

print('\n\nDETAILED METRICS BY BUCKET:\n')
for _, row in last_60s.iterrows():
    print(f"Time remaining: {row['time_to_close_bucket']}")
    print(f"  Accuracy:              {row['accuracy']*100:.2f}%")
    print(f"  Brier Score:           {row['brier']:.4f}")
    print(f"  ROC AUC:               {row['roc_auc']:.4f}")
    print(f"  Predicted up rate:     {row['avg_pred_up_prob']*100:.2f}%")
    print(f"  Actual up rate:        {row['actual_up_rate']*100:.2f}%")
    print(f"  Prediction difference: {(row['actual_up_rate']-row['avg_pred_up_prob'])*100:.2f} pp")
    print(f"  Samples:               {int(row['samples']):,}")
    print()

# Overall for last 60s
accuracy_avg = last_60s['accuracy'].mean()
brier_avg = last_60s['brier'].mean()
roc_auc_avg = last_60s['roc_auc'].mean()

print('='*80)
print('SUMMARY - LAST 60 SECONDS (averaged across all buckets):')
print(f"  Average accuracy:      {accuracy_avg*100:.2f}%")
print(f"  Average Brier score:   {brier_avg:.4f}")
print(f"  Average ROC AUC:       {roc_auc_avg:.4f}")
print(f"  Error rate:            {(1-accuracy_avg)*100:.2f}%")
