"""Live probability predictor for 5-minute candle close direction.

Fetches real-time 1-second Binance data for the current 5-minute candle,
computes features, and predicts P(candle closes up).
"""

from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import requests
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression


BINANCE_KLINES_URL = "https://api.binance.com/api/v3/klines"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Live predictor: fetches current 5-minute candle 1s data from Binance "
            "and predicts P(closes up)."
        )
    )
    parser.add_argument(
        "--symbol",
        default="BTCUSDT",
        help="Trading pair (default: BTCUSDT)",
    )
    parser.add_argument(
        "--training-csv",
        default="research/btcusdt_1s_last24h.csv",
        help="Path to training CSV for model calibration",
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=1,
        help="Update interval in seconds",
    )
    parser.add_argument(
        "--loop",
        action="store_true",
        help="Continuously predict; otherwise predict once and exit",
    )
    return parser.parse_args()


def fetch_current_5m_data(symbol: str) -> pd.DataFrame:
    """Fetch all 1-second candles from the current 5-minute window."""
    now = datetime.now(timezone.utc)
    candle_start = now.replace(second=0, microsecond=0) - timedelta(minutes=now.minute % 5)
    start_ms = int(candle_start.timestamp() * 1000)
    end_ms = int(now.timestamp() * 1000)

    params = {
        "symbol": symbol,
        "interval": "1s",
        "startTime": start_ms,
        "endTime": end_ms,
        "limit": 1000,
    }

    response = requests.get(BINANCE_KLINES_URL, params=params, timeout=10)
    response.raise_for_status()
    rows = response.json()

    if not rows:
        raise RuntimeError("No data returned from Binance for current 5-minute window")

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

    df = pd.DataFrame(rows, columns=columns)
    df = df.drop(columns=["ignore"])

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

    return df


def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute features for the current candle data."""
    df = df.copy()

    df["candle_5m_start"] = df["open_time"].dt.floor("5min")
    candle_open = df["open"].iloc[0]
    candle_end = df["candle_5m_start"].iloc[0] + pd.Timedelta(minutes=5)

    df["log_return_1s"] = np.log(df["close"]).diff()

    vol_window = 60
    df["inst_vol_60s"] = (
        df["log_return_1s"].rolling(window=vol_window, min_periods=10).std()
    )
    df["inst_vol_60s_bps"] = df["inst_vol_60s"] * 10_000

    df["trend_mean_60s"] = (
        df["log_return_1s"].rolling(window=vol_window, min_periods=10).mean()
    )
    df["trend_zscore"] = df["trend_mean_60s"] / (df["inst_vol_60s"] + 1e-12)

    df["vwap"] = df["quote_asset_volume"] / df["volume"].replace(0, np.nan)
    df["buy_volume_ratio"] = (
        df["taker_buy_base_asset_volume"] / df["volume"].replace(0, np.nan)
    )

    seconds_to_close = (candle_end - df["open_time"]).dt.total_seconds()
    df["seconds_to_5m_close"] = seconds_to_close.clip(lower=0)
    df["fraction_of_candle_elapsed"] = 1.0 - (df["seconds_to_5m_close"] / 300.0)

    df["return_from_candle_open"] = (df["close"] / candle_open) - 1.0

    return df


def train_model_on_history(training_csv_path: Path) -> tuple:
    """Train a logistic regression model on historical data."""
    df = pd.read_csv(training_csv_path)

    required = [
        "open_time",
        "open",
        "close",
        "volume",
        "number_of_trades",
        "inst_vol_60s",
        "inst_vol_60s_bps",
        "log_return_1s",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Training CSV missing columns: {missing}")

    df["open_time"] = pd.to_datetime(df["open_time"], utc=True)
    df["candle_5m_start"] = df["open_time"].dt.floor("5min")

    candle = (
        df.groupby("candle_5m_start", as_index=False)
        .agg(candle_open=("open", "first"), candle_close=("close", "last"))
        .copy()
    )
    candle["target_up"] = (candle["candle_close"] > candle["candle_open"]).astype(int)
    candle = candle[candle["candle_close"] != candle["candle_open"]].copy()

    df = df.merge(candle[["candle_5m_start", "target_up"]], on="candle_5m_start", how="inner")

    candle_end = df["candle_5m_start"] + pd.Timedelta(minutes=5)
    seconds_to_close = (candle_end - df["open_time"]).dt.total_seconds()
    df["seconds_to_5m_close"] = seconds_to_close.clip(lower=0)
    df["fraction_of_candle_elapsed"] = 1.0 - (df["seconds_to_5m_close"] / 300.0)

    df["return_from_candle_open"] = (
        df["close"] / df.groupby("candle_5m_start")["open"].transform("first")
    ) - 1.0
    if "buy_volume_ratio" in df.columns:
        df["buy_volume_ratio"] = df["buy_volume_ratio"].clip(lower=0, upper=1)

    feature_candidates = [
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
    features = [col for col in feature_candidates if col in df.columns]

    X = df[features].copy()
    y = df["target_up"].astype(int).values

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                features,
            )
        ],
        remainder="drop",
    )

    model = LogisticRegression(max_iter=2000, solver="lbfgs")
    pipe = Pipeline(steps=[("prep", preprocessor), ("model", model)])
    pipe.fit(X, y)

    return pipe, features


def predict_live(
    symbol: str,
    training_csv_path: Path,
    interval_sec: int = 5,
    loop: bool = False,
) -> None:
    """Predict up probability for the current 5-minute candle."""
    training_csv_path = Path(training_csv_path)
    if not training_csv_path.exists():
        raise FileNotFoundError(f"Training CSV not found: {training_csv_path}")

    print(f"Training model on historical data from {training_csv_path}...")
    model, features = train_model_on_history(training_csv_path)
    print(f"Model trained with {len(features)} features\n")

    iteration = 0
    while True:
        iteration += 1
        try:
            now = datetime.now(timezone.utc)
            candle_start = now.replace(second=0, microsecond=0) - timedelta(minutes=now.minute % 5)
            time_in_candle = (now - candle_start).total_seconds()

            print(f"\n[{now.isoformat()}] Fetching current 5m candle data...")
            df_current = fetch_current_5m_data(symbol)
            df_current = compute_features(df_current)

            if len(df_current) < 10:
                print(f"  Only {len(df_current)} rows; not enough for prediction yet")
                if not loop:
                    sys.exit(0)
                time.sleep(interval_sec)
                continue

            latest_row = df_current.iloc[-1:][features].copy()
            latest_row = latest_row.fillna(latest_row.median())

            prob_up = model.predict_proba(latest_row)[0, 1]
            pred_up = int(prob_up >= 0.5)

            seconds_left = 300 - time_in_candle
            pct_elapsed = (time_in_candle / 300.0) * 100

            confidence_bar = "█" * int(prob_up * 30)
            confidence_bar = confidence_bar.ljust(30)

            print(f"  Candle elapsed: {time_in_candle:.0f}s / 300s ({pct_elapsed:.1f}%)")
            print(f"  Seconds remaining: {seconds_left:.0f}s")
            print(f"  P(closes UP): {prob_up:.4f}")
            print(f"  P(closes DOWN): {1 - prob_up:.4f}")
            print(f"  [{confidence_bar}]")
            print(f"  Predicted: {'UP' if pred_up else 'DOWN'}")

            if not loop:
                break

            time.sleep(interval_sec)

        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            print(f"  Error: {e}")
            if not loop:
                sys.exit(1)
            time.sleep(interval_sec)


def main() -> None:
    args = parse_args()
    predict_live(
        symbol=args.symbol,
        training_csv_path=Path(args.training_csv),
        interval_sec=args.interval,
        loop=args.loop,
    )


if __name__ == "__main__":
    main()
