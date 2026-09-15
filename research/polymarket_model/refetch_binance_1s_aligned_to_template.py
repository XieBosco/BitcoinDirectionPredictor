from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import requests

BINANCE_KLINES_URL = "https://api.binance.com/api/v3/klines"


def fetch_klines_1s(symbol: str, start_ms: int, end_ms: int) -> List[List]:
    rows: List[List] = []
    cursor = start_ms
    session = requests.Session()

    while cursor < end_ms:
        params = {
            "symbol": symbol,
            "interval": "1s",
            "startTime": cursor,
            "endTime": end_ms,
            "limit": 1000,
        }

        batch = None
        last_exc: Exception | None = None
        for attempt in range(5):
            try:
                resp = session.get(BINANCE_KLINES_URL, params=params, timeout=30)
                resp.raise_for_status()
                batch = resp.json()
                break
            except Exception as exc:  # pragma: no cover
                last_exc = exc
                time.sleep(0.5 * (attempt + 1))

        if batch is None:
            raise RuntimeError(f"Binance request failed after retries: {last_exc}")

        if not batch:
            break

        rows.extend(batch)
        cursor = int(batch[-1][0]) + 1000
        time.sleep(0.01)

    return rows


def fetch_klines_1s_chunked(symbol: str, start_ms: int, end_ms: int, chunk_hours: int = 6) -> List[List]:
    chunk_ms = int(chunk_hours * 3600 * 1000)
    cursor = start_ms
    out: List[List] = []
    idx = 0

    while cursor < end_ms:
        idx += 1
        chunk_end = min(end_ms, cursor + chunk_ms)
        start_dt = pd.to_datetime(cursor, unit="ms", utc=True)
        end_dt = pd.to_datetime(chunk_end, unit="ms", utc=True)
        print(f"Chunk {idx}: {start_dt} -> {end_dt}")
        chunk_rows = fetch_klines_1s(symbol=symbol, start_ms=cursor, end_ms=chunk_end)
        out.extend(chunk_rows)
        print(f"  chunk rows: {len(chunk_rows):,} | total rows: {len(out):,}")
        cursor = chunk_end + 1000

    return out


def add_intrabar_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["price_velocity_1s"] = df["close"].diff()
    df["micro_price_acceleration"] = df["price_velocity_1s"].diff()

    candle_open = df.groupby("candle_5m_start")["open"].transform("first")
    above_open = (df["close"] > candle_open).astype(float)
    below_open = (df["close"] < candle_open).astype(float)
    ticks_in_candle = df.groupby("candle_5m_start").cumcount() + 1
    df["time_in_profit_pct"] = above_open.groupby(df["candle_5m_start"]).cumsum() / ticks_in_candle
    df["time_below_open_pct"] = below_open.groupby(df["candle_5m_start"]).cumsum() / ticks_in_candle

    running_high = df.groupby("candle_5m_start")["high"].cummax()
    running_low = df.groupby("candle_5m_start")["low"].cummin()
    running_body = (df["close"] - candle_open).abs()
    upper_wick = (running_high - np.maximum(candle_open, df["close"])).clip(lower=0)
    lower_wick = (np.minimum(candle_open, df["close"]) - running_low).clip(lower=0)
    df["dynamic_wick_to_body_ratio"] = (upper_wick + lower_wick) / (running_body + 1e-9)

    candle_cum_quote = df.groupby("candle_5m_start")["quote_asset_volume"].cumsum()
    candle_cum_volume = df.groupby("candle_5m_start")["volume"].cumsum()
    df["micro_vwap"] = candle_cum_quote / candle_cum_volume.replace(0, np.nan)
    df["intra_candle_vwap_deviation"] = (df["close"] / df["micro_vwap"]) - 1.0

    price_delta_10s = df["close"].diff(10)
    velocity_10s = price_delta_10s / 10.0
    volume_mass_10s = df["volume"].rolling(window=10, min_periods=3).sum()
    df["kinetic_energy_10s"] = volume_mass_10s * np.square(velocity_10s)

    tick_dir = np.sign(df["close"].diff()).fillna(0).astype(int)
    is_up = (tick_dir > 0).astype(float)
    is_down = (tick_dir < 0).astype(float)
    is_flat = (tick_dir == 0).astype(float)
    p_up = is_up.rolling(window=10, min_periods=3).mean()
    p_down = is_down.rolling(window=10, min_periods=3).mean()
    p_flat = is_flat.rolling(window=10, min_periods=3).mean()
    probs = np.vstack([p_up.to_numpy(), p_down.to_numpy(), p_flat.to_numpy()]).T
    probs = np.clip(probs, 1e-12, 1.0)
    probs = probs / probs.sum(axis=1, keepdims=True)
    df["tick_entropy_10s"] = -np.sum(probs * (np.log(probs) / np.log(2.0)), axis=1)

    accel_mean_15s = df["micro_price_acceleration"].rolling(window=15, min_periods=5).mean()
    df["micro_acceleration_decay_15s"] = df["micro_price_acceleration"] - accel_mean_15s

    idx_in_candle = df.groupby("candle_5m_start").cumcount().astype(float)
    sec_in_candle = np.minimum(idx_in_candle, 299.0)
    running_high = df.groupby("candle_5m_start")["high"].cummax()
    running_low = df.groupby("candle_5m_start")["low"].cummin()
    hit_new_high = (df["high"] >= running_high).astype(int)
    hit_new_low = (df["low"] <= running_low).astype(int)
    t_high = sec_in_candle.where(hit_new_high.astype(bool)).groupby(df["candle_5m_start"]).ffill().fillna(0.0)
    t_low = sec_in_candle.where(hit_new_low.astype(bool)).groupby(df["candle_5m_start"]).ffill().fillna(0.0)
    df["tte_high_ratio"] = t_high / 300.0
    df["tte_low_ratio"] = t_low / 300.0
    df["tte_range_asymmetry"] = (t_low - t_high) / 300.0

    prev_dir = tick_dir.shift(1)
    continuation = ((tick_dir != 0) & (prev_dir != 0) & (tick_dir == prev_dir)).astype(float)
    reversal = ((tick_dir != 0) & (prev_dir != 0) & (tick_dir != prev_dir)).astype(float)
    cont_20 = continuation.rolling(window=20, min_periods=5).sum()
    rev_20 = reversal.rolling(window=20, min_periods=5).sum()
    df["tick_continuation_reversal_ratio_20s"] = cont_20 / (rev_20 + 1e-6)

    return df


def build_feature_frame(raw_rows: List[List]) -> pd.DataFrame:
    cols = [
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
    df = pd.DataFrame(raw_rows, columns=cols)
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
    df = df.sort_values("open_time").drop_duplicates("open_time").reset_index(drop=True)

    df["log_return_1s"] = np.log(df["close"]).diff()
    vol_window = 60
    df["inst_vol_60s"] = df["log_return_1s"].rolling(window=vol_window, min_periods=10).std()
    df["inst_vol_60s_bps"] = df["inst_vol_60s"] * 10000
    df["trend_mean_60s"] = df["log_return_1s"].rolling(window=vol_window, min_periods=10).mean()
    df["trend_zscore"] = df["trend_mean_60s"] / (df["inst_vol_60s"] + 1e-12)
    df["vwap"] = df["quote_asset_volume"] / df["volume"].replace(0, np.nan)
    df["buy_volume_ratio"] = df["taker_buy_base_asset_volume"] / df["volume"].replace(0, np.nan)

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

    df = add_intrabar_features(df)
    return df


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Refetch Binance 1s klines and align to template timestamps/columns."
    )
    parser.add_argument(
        "--template-csv",
        default=r"notes/resources/binance_1s_for_polymarket_window.csv",
        help="Template CSV whose open_time rows and columns are matched.",
    )
    parser.add_argument(
        "--output-csv",
        default=r"notes/resources/binance_1s_for_polymarket_window.csv",
        help="Output CSV path.",
    )
    parser.add_argument("--symbol", default="BTCUSDT", help="Binance symbol, default BTCUSDT")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    template_path = Path(args.template_csv)
    output_path = Path(args.output_csv)

    template_header = pd.read_csv(template_path, nrows=0)
    template_cols = template_header.columns.tolist()

    template_ts = pd.read_csv(template_path, usecols=["open_time"])
    template_ts["open_time"] = pd.to_datetime(template_ts["open_time"], utc=True)
    template_ts = template_ts.drop_duplicates("open_time").sort_values("open_time").reset_index(drop=True)

    start_ts = template_ts["open_time"].min()
    end_ts = template_ts["open_time"].max()

    start_ms = int(start_ts.timestamp() * 1000)
    end_ms = int((end_ts + pd.Timedelta(seconds=1)).timestamp() * 1000)

    print(f"Template rows: {len(template_ts):,}")
    print(f"Template window: {start_ts} -> {end_ts}")
    print("Fetching Binance 1s data...")

    raw = fetch_klines_1s_chunked(symbol=args.symbol, start_ms=start_ms, end_ms=end_ms, chunk_hours=6)
    if not raw:
        raise RuntimeError("No rows returned from Binance.")

    bn = build_feature_frame(raw)
    print(f"Fetched rows: {len(bn):,}")

    aligned = template_ts.merge(bn, on="open_time", how="left")

    for col in template_cols:
        if col not in aligned.columns:
            aligned[col] = np.nan

    aligned = aligned[template_cols]

    missing_ohlc = aligned[["open", "high", "low", "close"]].isna().any(axis=1).sum()
    print(f"Rows with missing OHLC after alignment: {missing_ohlc:,}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    aligned.to_csv(output_path, index=False)

    print(f"Saved aligned dataset: {output_path.resolve()}")
    print(f"Output rows: {len(aligned):,} | Output columns: {len(aligned.columns):,}")


if __name__ == "__main__":
    main()
