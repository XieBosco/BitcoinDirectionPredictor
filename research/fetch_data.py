"""Fetch Binance BTC/USDT 1-second market data for the last 24 hours.

The script saves a CSV with core OHLCV market fields plus:
- log_return_1s: 1-second log return
- inst_vol_60s: rolling 60-second standard deviation of log returns
- inst_vol_60s_bps: same volatility in basis points
- market_regime: a simple label describing local market state
"""

from __future__ import annotations

import argparse
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import requests


BINANCE_KLINES_URL = "https://api.binance.com/api/v3/klines"


def fetch_klines_1s(symbol: str, start_ms: int, end_ms: int) -> List[List]:
	"""Fetch 1-second klines from Binance using pagination."""
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

		response = session.get(BINANCE_KLINES_URL, params=params, timeout=20)
		response.raise_for_status()
		batch = response.json()

		if not batch:
			break

		all_rows.extend(batch)
		last_open_time = int(batch[-1][0])
		# Move forward by 1 second to avoid overlap.
		current_start = last_open_time + 1000
		time.sleep(0.03)

	return all_rows


def build_dataframe(raw_rows: List[List]) -> pd.DataFrame:
	"""Convert raw Binance kline rows to typed DataFrame."""
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

	# Per-second price features.
	df["log_return_1s"] = np.log(df["close"]).diff()

	# Instantaneous volatility proxy: rolling std of 1s returns over 60s.
	vol_window = 60
	df["inst_vol_60s"] = (
		df["log_return_1s"].rolling(window=vol_window, min_periods=10).std()
	)
	df["inst_vol_60s_bps"] = df["inst_vol_60s"] * 10_000

	# Trend proxy from local mean return and normalized strength.
	df["trend_mean_60s"] = (
		df["log_return_1s"].rolling(window=vol_window, min_periods=10).mean()
	)
	df["trend_zscore"] = df["trend_mean_60s"] / (df["inst_vol_60s"] + 1e-12)

	# Useful market activity metrics.
	df["vwap"] = df["quote_asset_volume"] / df["volume"].replace(0, np.nan)
	df["buy_volume_ratio"] = (
		df["taker_buy_base_asset_volume"] / df["volume"].replace(0, np.nan)
	)

	# Regime boundaries from dataset distribution.
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


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(
		description="Fetch Binance 1-second data for last 24h and save enriched CSV."
	)
	parser.add_argument("--symbol", default="BTCUSDT", help="Trading pair, default BTCUSDT")
	parser.add_argument(
		"--output",
		default="btcusdt_1s_last24h.csv",
		help="Output CSV path",
	)
	return parser.parse_args()


def main() -> None:
	args = parse_args()

	end_time = datetime.now(timezone.utc)
	start_time = end_time - timedelta(hours=24)
	start_ms = int(start_time.timestamp() * 1000)
	end_ms = int(end_time.timestamp() * 1000)

	print(
		f"Fetching 1-second klines for {args.symbol} from {start_time.isoformat()} to {end_time.isoformat()}"
	)
	raw_rows = fetch_klines_1s(args.symbol, start_ms, end_ms)

	if not raw_rows:
		raise RuntimeError("No data returned from Binance. Try again in a few seconds.")

	df = build_dataframe(raw_rows)
	df = df.drop_duplicates(subset=["open_time"]).sort_values("open_time")

	output_path = Path(args.output)
	output_path.parent.mkdir(parents=True, exist_ok=True)
	df.to_csv(output_path, index=False)

	print(f"Saved {len(df):,} rows to {output_path.resolve()}")
	print(
		"Columns include price data, volume/trades data, instantaneous volatility, and regime labels."
	)


if __name__ == "__main__":
	main()
