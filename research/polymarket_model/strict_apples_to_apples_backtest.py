from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import requests
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

BINANCE_KLINES_URL = "https://api.binance.com/api/v3/klines"


def fetch_klines_1s(symbol: str, start_ms: int, end_ms: int) -> List[List]:
    all_rows: List[List] = []
    current_start = start_ms
    session = requests.Session()

    while current_start < end_ms:
        params = {
            "symbol": symbol,
            "interval": "1s",
            "startTime": current_start,
            "endTime": end_ms,
            "limit": 1000,
        }

        last_exc = None
        batch = None
        for attempt in range(5):
            try:
                response = session.get(BINANCE_KLINES_URL, params=params, timeout=30)
                response.raise_for_status()
                batch = response.json()
                break
            except Exception as exc:  # noqa: BLE001
                last_exc = exc
                time.sleep(0.5 * (attempt + 1))

        if batch is None:
            raise RuntimeError(f"Failed Binance fetch after retries: {last_exc}")

        if not batch:
            break

        all_rows.extend(batch)
        current_start = int(batch[-1][0]) + 1000
        time.sleep(0.02)

    return all_rows


def fetch_klines_1s_chunked(symbol: str, start_ms: int, end_ms: int, chunk_hours: int = 6) -> List[List]:
    chunk_ms = int(chunk_hours * 3600 * 1000)
    cursor = start_ms
    merged: List[List] = []
    chunk_idx = 0

    while cursor < end_ms:
        chunk_idx += 1
        c_end = min(end_ms, cursor + chunk_ms)
        print(f"Fetching Binance chunk {chunk_idx}: {pd.to_datetime(cursor, unit='ms', utc=True)} -> {pd.to_datetime(c_end, unit='ms', utc=True)}")
        rows = fetch_klines_1s(symbol=symbol, start_ms=cursor, end_ms=c_end)
        merged.extend(rows)
        cursor = c_end + 1000

    return merged


def build_binance_df(raw_rows: List[List]) -> pd.DataFrame:
    columns = [
        "open_time",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "close_time",
        "quote_asset_volume",
        "number_of_trades",
        "taker_buy_base_asset_volume",
        "taker_buy_quote_asset_volume",
        "ignore",
    ]

    df = pd.DataFrame(raw_rows, columns=columns).drop(columns=["ignore"])

    numeric_cols = [
        "open",
        "high",
        "low",
        "close",
        "volume",
        "quote_asset_volume",
        "number_of_trades",
        "taker_buy_base_asset_volume",
        "taker_buy_quote_asset_volume",
    ]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    df["close_time"] = pd.to_datetime(df["close_time"], unit="ms", utc=True)
    df = df.sort_values("open_time").drop_duplicates("open_time").reset_index(drop=True)

    # Match feature engineering from the logistic workflow.
    df["log_return_1s"] = np.log(df["close"]).diff()

    vol_window = 60
    df["inst_vol_60s"] = df["log_return_1s"].rolling(window=vol_window, min_periods=10).std()
    df["inst_vol_60s_bps"] = df["inst_vol_60s"] * 10000
    df["trend_mean_60s"] = df["log_return_1s"].rolling(window=vol_window, min_periods=10).mean()
    df["trend_zscore"] = df["trend_mean_60s"] / (df["inst_vol_60s"] + 1e-12)
    df["vwap"] = df["quote_asset_volume"] / df["volume"].replace(0, np.nan)
    df["buy_volume_ratio"] = df["taker_buy_base_asset_volume"] / df["volume"].replace(0, np.nan)

    # Label and temporal position features.
    df["candle_5m_start"] = df["open_time"].dt.floor("5min")
    candle = (
        df.groupby("candle_5m_start", as_index=False)
        .agg(candle_open=("open", "first"), candle_close=("close", "last"))
        .copy()
    )
    candle["target_up"] = (candle["candle_close"] > candle["candle_open"]).astype(int)

    df = df.merge(candle[["candle_5m_start", "target_up"]], on="candle_5m_start", how="left")

    candle_end = df["candle_5m_start"] + pd.Timedelta(minutes=5)
    df["seconds_to_5m_close"] = (candle_end - df["open_time"]).dt.total_seconds().clip(lower=0)
    df["fraction_of_candle_elapsed"] = 1.0 - (df["seconds_to_5m_close"] / 300.0)
    df["return_from_candle_open"] = (
        df["close"] / df.groupby("candle_5m_start")["open"].transform("first")
    ) - 1.0

    return df


def load_polymarket_1s(polymarket_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(polymarket_csv)

    df["timestamp"] = pd.to_datetime(df["timestamp_log"], unit="s", utc=True)
    df["start_time"] = pd.to_numeric(df["start_time"], errors="coerce")
    df["elapsed"] = pd.to_numeric(df["elapsed"], errors="coerce")
    df["label"] = (df["winner"].astype(str).str.lower() == "up").astype(int)

    # Implied probability = midpoint between YES best bid and YES best ask.
    df["implied_prob"] = (pd.to_numeric(df["bid_YES"], errors="coerce") + pd.to_numeric(df["ask_YES"], errors="coerce")) / 2.0
    df["implied_prob"] = df["implied_prob"].clip(0.0, 1.0)

    # Upsample 2s -> 1s by carry-forward within each market slug.
    upsampled = []
    for slug, g in df.groupby("slug", sort=False):
        g = g.sort_values("timestamp").copy()
        idx = pd.date_range(start=g["timestamp"].min(), end=g["timestamp"].max(), freq="1s", tz="UTC")
        u = g.set_index("timestamp").reindex(idx)

        for col in ["slug", "start_time", "label", "implied_prob"]:
            u[col] = u[col].ffill().bfill()

        u = u.reset_index().rename(columns={"index": "timestamp"})
        u["elapsed"] = (u["timestamp"].astype("int64") // 10**9 - u["start_time"]).astype(int)
        upsampled.append(u[["slug", "timestamp", "elapsed", "label", "implied_prob"]])

    pm = pd.concat(upsampled, ignore_index=True)
    pm = pm[(pm["elapsed"] >= 100) & (pm["elapsed"] <= 290)].copy()
    return pm


def build_logistic_oof_probs(df: pd.DataFrame, n_splits: int = 5) -> pd.DataFrame:
    feature_cols = [
        "close",
        "volume",
        "number_of_trades",
        "log_return_1s",
        "inst_vol_60s",
        "inst_vol_60s_bps",
        "trend_mean_60s",
        "trend_zscore",
        "buy_volume_ratio",
        "vwap",
        "return_from_candle_open",
        "seconds_to_5m_close",
        "fraction_of_candle_elapsed",
    ]
    feature_cols = [c for c in feature_cols if c in df.columns]

    model_df = df.dropna(subset=feature_cols + ["target_up", "candle_5m_start"]).copy()

    X = model_df[feature_cols]
    y = model_df["target_up"].astype(int).values
    groups = model_df["candle_5m_start"].astype(str).values

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline(steps=[("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]),
                feature_cols,
            )
        ],
        remainder="drop",
    )

    estimator = Pipeline(steps=[("prep", preprocessor), ("model", LogisticRegression(max_iter=2000, solver="lbfgs"))])

    unique_groups = np.unique(groups)
    use_splits = min(n_splits, len(unique_groups))
    if use_splits < 2:
        raise ValueError("Not enough groups to run GroupKFold")

    gkf = GroupKFold(n_splits=use_splits)
    oof = np.full(len(model_df), np.nan)

    for train_idx, test_idx in gkf.split(X, y, groups=groups):
        estimator.fit(X.iloc[train_idx], y[train_idx])
        oof[test_idx] = estimator.predict_proba(X.iloc[test_idx])[:, 1]

    model_df["logreg_prob"] = oof
    return model_df


def compute_metrics(y_true: np.ndarray, y_prob: np.ndarray) -> Dict[str, float]:
    y_pred = (y_prob >= 0.5).astype(int)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "log_loss": float(log_loss(y_true, y_prob, labels=[0, 1])),
        "brier": float(brier_score_loss(y_true, y_prob)),
        "calibration_gap_pp": float(np.mean(np.abs(y_prob - y_true)) * 100.0),
        "mean_pred": float(np.mean(y_prob)),
        "mean_actual": float(np.mean(y_true)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Strict apples-to-apples Polymarket vs Logistic backtest")
    parser.add_argument("--polymarket-csv", default="research/polymarket_model/market_data_2sec_weekly5_with_resolutions.csv")
    parser.add_argument("--binance-cache", default="research/polymarket_model/binance_1s_for_polymarket_window.csv")
    parser.add_argument("--output-dir", default="research/polymarket_model/strict_compare_results")
    parser.add_argument("--symbol", default="BTCUSDT")
    args = parser.parse_args()

    pm = load_polymarket_1s(Path(args.polymarket_csv))

    start_ts = pm["timestamp"].min().floor("s")
    end_ts = pm["timestamp"].max().ceil("s")

    cache_path = Path(args.binance_cache)
    if cache_path.exists():
        bn = pd.read_csv(cache_path)
        bn["open_time"] = pd.to_datetime(bn["open_time"], utc=True)
        bn = bn[(bn["open_time"] >= start_ts) & (bn["open_time"] <= end_ts)].copy()
    else:
        start_ms = int(start_ts.timestamp() * 1000)
        end_ms = int(end_ts.timestamp() * 1000)
        raw = fetch_klines_1s_chunked(args.symbol, start_ms, end_ms, chunk_hours=6)
        if not raw:
            raise RuntimeError("No Binance rows fetched for required range")
        bn = build_binance_df(raw)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        bn.to_csv(cache_path, index=False)

    if "target_up" not in bn.columns:
        bn = build_binance_df(bn.values.tolist())

    # Join on exact second timestamps for strict same-row comparison.
    merged = pm.merge(
        bn,
        left_on="timestamp",
        right_on="open_time",
        how="inner",
        suffixes=("_pm", "_bn"),
    )

    # Keep only exact requested window by polymarket elapsed
    merged = merged[(merged["elapsed"] >= 100) & (merged["elapsed"] <= 290)].copy()

    # Build logistic OOF probs using this exact merged sample.
    model_df = build_logistic_oof_probs(merged, n_splits=5)
    model_df = model_df.dropna(subset=["logreg_prob", "implied_prob", "target_up"]).copy()

    y_true = model_df["target_up"].astype(int).values

    pm_metrics = compute_metrics(y_true, model_df["implied_prob"].astype(float).values)
    lr_metrics = compute_metrics(y_true, model_df["logreg_prob"].astype(float).values)

    comparison = pd.DataFrame(
        [
            {"model": "polymarket", **pm_metrics},
            {"model": "logistic_regression", **lr_metrics},
        ]
    )

    # Bucket comparison across same rows.
    bins = [100, 130, 160, 190, 220, 250, 291]
    labels = ["100-129", "130-159", "160-189", "190-219", "220-249", "250-290"]
    model_df["bucket"] = pd.cut(model_df["elapsed"], bins=bins, labels=labels, right=False)

    bucket_rows = []
    for bucket, g in model_df.groupby("bucket", observed=False):
        if pd.isna(bucket) or len(g) == 0:
            continue
        yb = g["target_up"].astype(int).values
        pm_b = compute_metrics(yb, g["implied_prob"].astype(float).values)
        lr_b = compute_metrics(yb, g["logreg_prob"].astype(float).values)
        bucket_rows.append({"bucket": str(bucket), "model": "polymarket", **pm_b, "n_rows": len(g)})
        bucket_rows.append({"bucket": str(bucket), "model": "logistic_regression", **lr_b, "n_rows": len(g)})

    bucket_cmp = pd.DataFrame(bucket_rows)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model_df[["slug", "timestamp", "elapsed", "target_up", "implied_prob", "logreg_prob"]].to_csv(
        out_dir / "strict_row_level_predictions.csv", index=False
    )
    comparison.to_csv(out_dir / "strict_overall_comparison.csv", index=False)
    bucket_cmp.to_csv(out_dir / "strict_bucket_comparison.csv", index=False)

    print("=== STRICT APPLES-TO-APPLES BACKTEST ===")
    print(f"Window: {start_ts} to {end_ts}")
    print(f"Compared rows: {len(model_df):,}")
    print(f"Compared candles: {model_df['slug'].nunique():,}")

    print("\nOverall comparison:")
    print(comparison.to_string(index=False))

    # Winner counts across key metrics.
    higher_better = ["accuracy", "roc_auc"]
    lower_better = ["log_loss", "brier", "calibration_gap_pp"]

    wins = {"polymarket": 0, "logistic_regression": 0}
    for m in higher_better:
        best = comparison.sort_values(m, ascending=False).iloc[0]["model"]
        wins[best] += 1
    for m in lower_better:
        best = comparison.sort_values(m, ascending=True).iloc[0]["model"]
        wins[best] += 1

    print("\nMetric wins (5 total):", wins)
    if wins["polymarket"] > wins["logistic_regression"]:
        print("Fair winner: Polymarket")
    elif wins["polymarket"] < wins["logistic_regression"]:
        print("Fair winner: Logistic Regression")
    else:
        print("Fair winner: Tie")

    print("\nSaved:")
    print(f"- {out_dir / 'strict_overall_comparison.csv'}")
    print(f"- {out_dir / 'strict_bucket_comparison.csv'}")
    print(f"- {out_dir / 'strict_row_level_predictions.csv'}")


if __name__ == "__main__":
    main()
