import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, roc_auc_score, log_loss, brier_score_loss

def make_base_pipeline(feature_cols):
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, feature_cols)
        ],
        remainder='drop'
    )
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', LogisticRegression(max_iter=2000, solver='lbfgs'))
    ])
    return pipeline

def load_polymarket_1s(polymarket_csv):
    pm = pd.read_csv(polymarket_csv)
    pm['timestamp'] = pm['timestamp_log'].astype(int)
    
    # Label is 1 if winner is up, else 0
    pm['label'] = (pm['winner'] == 'Up').astype(int)
    
    if 'bid_YES' in pm.columns and 'ask_YES' in pm.columns:
        pm['implied_prob'] = pm[['bid_YES', 'ask_YES']].mean(axis=1).clip(0, 1)
    else:
        pm['implied_prob'] = 0.5
        
    # 2 seconds to 1 second carry forward
    pm = pm.sort_values(['slug', 'timestamp'])
    pm.set_index('timestamp', inplace=True)
    all_slugs_dfs = []
    for slug, grp in pm.groupby('slug'):
        grp = grp[~grp.index.duplicated(keep='first')]
        if len(grp) > 0:
            reindexed = grp.reindex(range(grp.index.min(), grp.index.max() + 1), method='ffill')
            all_slugs_dfs.append(reindexed)
    if all_slugs_dfs:
        pm_1s = pd.concat(all_slugs_dfs).reset_index()
    else:
        pm_1s = pd.DataFrame()
        
    pm_1s = pm_1s[(pm_1s['elapsed'] >= 100) & (pm_1s['elapsed'] <= 290)]
    return pm_1s

def load_binance_data(binance_csv, pm_min_ts, pm_max_ts):
    df = pd.read_csv(binance_csv)
    df['open_time'] = pd.to_datetime(df['open_time'], utc=True)
    df['timestamp'] = df['open_time'].astype('int64') // 10**9
    
    # 4. Build the target from actual BTC price movement
    # For each 5-minute candle: candle_open, candle_close, target_up
    df['candle_5m_start'] = df['open_time'].dt.floor('5min')
    
    first_open = df.groupby('candle_5m_start')['open'].transform('first')
    last_close = df.groupby('candle_5m_start')['close'].transform('last')
    
    df['target_up'] = (last_close > first_open).astype(int)
    
    df = df[(df['timestamp'] >= pm_min_ts) & (df['timestamp'] <= pm_max_ts)]
    return df

def run_walk_forward_test(merged_df, feature_cols):
    # Sort by timestamp
    df = merged_df.sort_values('timestamp').reset_index(drop=True)
    
    min_time = df['open_time'].min()
    max_time = df['open_time'].max()
    
    split_results = []
    row_predictions = []
    
    current_time = min_time + pd.Timedelta(days=7)
    step = pd.Timedelta(hours=6)
    
    split_idx = 0
    while current_time < max_time:
        train_start = current_time - pd.Timedelta(days=7)
        test_end = current_time + pd.Timedelta(hours=6)
        
        train_mask = (df['open_time'] >= train_start) & (df['open_time'] < current_time)
        test_mask = (df['open_time'] >= current_time) & (df['open_time'] < test_end)
        
        train_data = df[train_mask]
        test_data = df[test_mask]
        
        if len(train_data) < 100 or len(test_data) < 50:
            current_time += step
            continue
            
        X_train = train_data[feature_cols]
        y_train = train_data['target_up']
        X_test = test_data[feature_cols]
        y_test = test_data['target_up']
        
        # Train baseline
        base_pipe = make_base_pipeline(feature_cols)
        calibrated_clf = CalibratedClassifierCV(estimator=base_pipe, method="sigmoid", cv=3)
        calibrated_clf.fit(X_train, y_train)
        
        # Predict baseline
        probs_baseline = calibrated_clf.predict_proba(X_test)[:, 1]
        probs_baseline = np.clip(probs_baseline, 1e-6, 1 - 1e-6)
        
        pm_probs = np.clip(test_data['implied_prob'].values, 1e-6, 1 - 1e-6)
        y_true = y_test.values
        
        # Metrics for baseline
        try:
            auc_base = roc_auc_score(y_true, probs_baseline)
        except:
            auc_base = 0.5
            
        base_metrics = {
            'model': '7day_logistic_baseline',
            'split': split_idx,
            'accuracy': accuracy_score(y_true, (probs_baseline >= 0.5).astype(int)),
            'roc_auc': auc_base,
            'log_loss': log_loss(y_true, probs_baseline, labels=[0,1]),
            'brier': brier_score_loss(y_true, probs_baseline),
            'calibration_gap_pp': np.mean(np.abs(probs_baseline - y_true)) * 100
        }
        split_results.append(base_metrics)
        
        # Metrics for polymarket
        try:
            auc_pm = roc_auc_score(y_true, pm_probs)
        except:
            auc_pm = 0.5
            
        pm_metrics = {
            'model': 'polymarket',
            'split': split_idx,
            'accuracy': accuracy_score(y_true, (pm_probs >= 0.5).astype(int)),
            'roc_auc': auc_pm,
            'log_loss': log_loss(y_true, pm_probs, labels=[0,1]),
            'brier': brier_score_loss(y_true, pm_probs),
            'calibration_gap_pp': np.mean(np.abs(pm_probs - y_true)) * 100
        }
        split_results.append(pm_metrics)
        
        # Row level predictions
        for i in range(len(test_data)):
            row_predictions.append({
                'split_index': split_idx,
                'timestamp': test_data.iloc[i]['timestamp'],
                'slug': test_data.iloc[i]['slug'],
                'elapsed': test_data.iloc[i]['elapsed'],
                'y_true': y_true[i],
                'prob_7d_baseline': probs_baseline[i],
                'prob_polymarket': pm_probs[i]
            })
            
        split_idx += 1
        current_time += step
        
    return pd.DataFrame(split_results), pd.DataFrame(row_predictions)

def main():
    res_dir = 'notes/resources'
    out_dir = 'researchv2/polymarket_model/strict_7day_vs_pm_results'
    os.makedirs(out_dir, exist_ok=True)
    
    print("Loading Polymarket data...")
    pm_1s = load_polymarket_1s(f"{res_dir}/market_data_2sec_weekly5_with_resolutions.csv")
    
    if len(pm_1s) == 0:
        print("No polymarket data in the given elapsed range.")
        return
        
    print("Loading Binance data...")
    pm_min_ts = pm_1s['timestamp'].min()
    pm_max_ts = pm_1s['timestamp'].max()
    binance_df = load_binance_data(f"{res_dir}/binance_1s_for_polymarket_window.csv", pm_min_ts, pm_max_ts)
    
    print("Merging datasets...")
    merged_df = pd.merge(binance_df, pm_1s, on='timestamp', how='inner')
    
    # Fill median for missing values of features where reasonable
    feature_cols = [
        'close', 'volume', 'number_of_trades', 'log_return_1s', 
        'inst_vol_60s', 'inst_vol_60s_bps', 'trend_mean_60s', 
        'trend_zscore', 'buy_volume_ratio', 'vwap', 
        'return_from_candle_open', 'seconds_to_5m_close', 'fraction_of_candle_elapsed'
    ]
    
    # Filter out where target is explicitly null
    merged_df = merged_df.dropna(subset=['target_up'])
    
    print(f"Merged dataset has {len(merged_df)} rows before evaluation.")
    
    print("Running walk-forward evaluation...")
    split_metrics_df, row_preds_df = run_walk_forward_test(merged_df, feature_cols)
    
    if len(split_metrics_df) == 0:
        print("No valid splits found.")
        return
        
    print("Computing summary metrics...")
    summary_metrics = split_metrics_df.groupby('model').agg({
        'accuracy': ['mean', 'std'],
        'roc_auc': ['mean', 'std'],
        'log_loss': ['mean', 'std'],
        'brier': ['mean', 'std'],
        'calibration_gap_pp': ['mean', 'std']
    })
    
    summary_metrics.columns = [f"{col[0]}_{col[1]}" for col in summary_metrics.columns]
    summary_metrics = summary_metrics.reset_index()
    
    print("Saving results...")
    summary_metrics.to_csv(f"{out_dir}/overall_comparison.csv", index=False)
    split_metrics_df.to_csv(f"{out_dir}/per_split_metrics.csv", index=False)
    row_preds_df.to_csv(f"{out_dir}/row_level_predictions.csv", index=False)
    
    print("\nOverall Summary:")
    print(summary_metrics.to_string(index=False))
    
    base_metrics = summary_metrics[summary_metrics['model'] == '7day_logistic_baseline'].iloc[0]
    pm_metrics = summary_metrics[summary_metrics['model'] == 'polymarket'].iloc[0]
    
    base_wins = 0
    pm_wins = 0
    
    if base_metrics['accuracy_mean'] > pm_metrics['accuracy_mean']:
        base_wins += 1
    elif pm_metrics['accuracy_mean'] > base_metrics['accuracy_mean']:
        pm_wins += 1
        
    if base_metrics['roc_auc_mean'] > pm_metrics['roc_auc_mean']:
        base_wins += 1
    elif pm_metrics['roc_auc_mean'] > base_metrics['roc_auc_mean']:
        pm_wins += 1
        
    if base_metrics['log_loss_mean'] < pm_metrics['log_loss_mean']:
        base_wins += 1
    elif pm_metrics['log_loss_mean'] < base_metrics['log_loss_mean']:
        pm_wins += 1
        
    if base_metrics['brier_mean'] < pm_metrics['brier_mean']:
        base_wins += 1
    elif pm_metrics['brier_mean'] < base_metrics['brier_mean']:
        pm_wins += 1
        
    if base_metrics['calibration_gap_pp_mean'] < pm_metrics['calibration_gap_pp_mean']:
        base_wins += 1
    elif pm_metrics['calibration_gap_pp_mean'] < base_metrics['calibration_gap_pp_mean']:
        pm_wins += 1
        
    print(f"\nMetric Wins - Baseline: {base_wins}, Polymarket: {pm_wins}")
    if base_wins > pm_wins:
        print("Winner: 7day_logistic_baseline")
    elif pm_wins > base_wins:
        print("Winner: polymarket")
    else:
        print("Winner: Tie")

if __name__ == "__main__":
    main()
