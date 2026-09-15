"""Fetch Binance BTC/USDT 1-second data for backtesting windows.

This script is similar to research/fetch_data.py, but supports multi-day
ranges needed for walk-forward backtests (for example 8-14 days).
"""

from __future__ import annotations

import argparse
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import requests


BINANCE_KLINES_URL = "https://api.binance.com/api/v3/klines"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Fetch enriched 1-second Binance klines for a multi-day lookback "
            "to support backtesting."
        )
    )
    parser.add_argument("--symbol", default="BTCUSDT", help="Trading pair, default BTCUSDT")
    parser.add_argument(
        "--days",
        type=float,
        default=8.0,
        help="Lookback size in days (default: 8). Use >= 7.25 for 7d backtest windows.",
    )
    parser.add_argument(
        "--output",
        default="research/window_generalization_tests/btcusdt_1s_backtest_data.csv",
        help="Output CSV path",
    )
    parser.add_argument(
        "--sleep-seconds",
        type=float,
        default=0.03,
        help="Sleep between API calls to reduce request pressure",
    )
    return parser.parse_args()


def fetch_klines_1s(symbol: str, start_ms: int, end_ms: int, sleep_seconds: float) -> List[List]:
    """Fetch 1-second klines from Binance with pagination."""
    all_rows: List[List] = []
    current_start = start_ms
    session = requests.Session()
    request_count = 0

    while current_start < end_ms:
        params = {
            "symbol": symbol,
            "interval": "1s",
            "startTime": current_start,
            "endTime": end_ms,
            "limit": 1000,
        }

        response = session.get(BINANCE_KLINES_URL, params=params, timeout=30)
        response.raise_for_status()
        batch = response.json()
        request_count += 1

        if not batch:
            break

        all_rows.extend(batch)
        last_open_time = int(batch[-1][0])
        current_start = last_open_time + 1000

        if request_count % 100 == 0:
            fetched_seconds = len(all_rows)
            print(f"Requests: {request_count:,} | Rows fetched: {fetched_seconds:,}")

        time.sleep(sleep_seconds)

    return all_rows


def build_dataframe(raw_rows: List[List]) -> pd.DataFrame:
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

    df = pd.DataFrame(raw_rows, columns=columns)
    df = df.drop(columns=["ignore"]).copy()

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

    # Match the feature engineering used in the current modeling pipeline.
    df["log_return_1s"] = np.log(df["close"]).diff()

    vol_window = 60
    df["inst_vol_60s"] = df["log_return_1s"].rolling(window=vol_window, min_periods=10).std()
    df["inst_vol_60s_bps"] = df["inst_vol_60s"] * 10_000

    df["trend_mean_60s"] = df["log_return_1s"].rolling(window=vol_window, min_periods=10).mean()
    df["trend_zscore"] = df["trend_mean_60s"] / (df["inst_vol_60s"] + 1e-12)

    df["vwap"] = df["quote_asset_volume"] / df["volume"].replace(0, np.nan)
    df["buy_volume_ratio"] = df["taker_buy_base_asset_volume"] / df["volume"].replace(0, np.nan)

    vol_low = df["inst_vol_60s"].quantile(0.30)
    vol_high = df["inst_vol_60s"].quantile(0.70)

    conditions = [
        df["inst_vol_60s"] <= vol_low,
        (df["inst_vol_60s"] >= vol_high) & (df["trend_zscore"] >= 1.5),
        (df["inst_vol_60s"] >= vol_high) & (df["trend_zscore"] <= -1.5),
        (df["inst_vol_60s"] >= vol_high) & (df["trend_zscore"].abs() < 0.5),
    ]
    labels = ["calm", "volatile_trend_up", "volatile_trend_down", "high_vol_chop"]
    df["market_regime"] = np.select(conditions, labels, default="normal")

    return df


def main() -> None:
    args = parse_args()
    if args.days <= 0:
        raise ValueError("--days must be positive")

    end_time = datetime.now(timezone.utc)
    start_time = end_time - timedelta(days=args.days)

    start_ms = int(start_time.timestamp() * 1000)
    end_ms = int(end_time.timestamp() * 1000)

    print(f"Fetching {args.symbol} 1-second klines")
    print(f"Range start: {start_time.isoformat()}")
    print(f"Range end:   {end_time.isoformat()}")
    print(f"Requested days: {args.days}")

    raw_rows = fetch_klines_1s(
        symbol=args.symbol,
        start_ms=start_ms,
        end_ms=end_ms,
        sleep_seconds=args.sleep_seconds,
    )

    if not raw_rows:
        raise RuntimeError("No data returned from Binance. Retry in a few seconds.")

    df = build_dataframe(raw_rows)
    df = df.drop_duplicates(subset=["open_time"]).sort_values("open_time").reset_index(drop=True)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    print(f"Saved rows: {len(df):,}")
    print(f"Output: {output_path.resolve()}")
    print(
        "Done. CSV contains OHLCV + instantaneous volatility + regime + "
        "supporting microstructure features used by the model."
    )


if __name__ == "__main__":
    main()
