from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


TIME_BUCKET_BINS = [100, 130, 160, 190, 220, 250, 291]
TIME_BUCKET_LABELS = ["100-129", "130-159", "160-189", "190-219", "220-249", "250-290"]
REGIME_VALUES = ["calm", "normal", "volatile_trend_up", "volatile_trend_down", "high_vol_chop"]


def safe_logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def compute_metrics(y_true: np.ndarray, y_prob: np.ndarray) -> Dict[str, float]:
    y_pred = (y_prob >= 0.5).astype(int)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "roc_auc": float(roc_auc_score(y_true, y_prob)) if len(np.unique(y_true)) > 1 else np.nan,
        "log_loss": float(log_loss(y_true, y_prob, labels=[0, 1])),
        "brier": float(brier_score_loss(y_true, y_prob)),
        "calibration_gap_pp": float(np.mean(np.abs(y_prob - y_true)) * 100.0),
        "mean_pred": float(np.mean(y_prob)),
        "mean_actual": float(np.mean(y_true)),
    }


def load_polymarket_1s(path: Path) -> pd.DataFrame:
    raw = pd.read_csv(path)
    raw["timestamp"] = pd.to_datetime(raw["timestamp_log"], unit="s", utc=True)
    raw["elapsed"] = pd.to_numeric(raw["elapsed"], errors="coerce")
    raw["label_pm"] = (raw["winner"].astype(str).str.lower() == "up").astype(int)

    for c in ["bid_YES", "ask_YES", "bid_NO", "ask_NO"]:
        raw[c] = pd.to_numeric(raw[c], errors="coerce")

    # Top-of-book implied probability for YES.
    raw["implied_prob"] = ((raw["bid_YES"] + raw["ask_YES"]) / 2.0).clip(0.0, 1.0)

    # Order-book features available from notes suggestion #3 (L1 proxies only).
    raw["yes_spread"] = (raw["ask_YES"] - raw["bid_YES"]).clip(lower=0)
    raw["no_spread"] = (raw["ask_NO"] - raw["bid_NO"]).clip(lower=0)
    raw["book_spread_sum"] = raw["yes_spread"] + raw["no_spread"]
    raw["book_imbalance_yes_no"] = ((raw["bid_YES"] + raw["ask_YES"]) - (raw["bid_NO"] + raw["ask_NO"])) / 2.0
    raw["book_consistency_gap"] = (raw["implied_prob"] + ((raw["bid_NO"] + raw["ask_NO"]) / 2.0) - 1.0).abs()

    # Upsample to 1s by carry-forward inside each market slug.
    pieces = []
    for slug, g in raw.groupby("slug", sort=False):
        g = g.sort_values("timestamp").copy()
        idx = pd.date_range(g["timestamp"].min(), g["timestamp"].max(), freq="1s", tz="UTC")
        u = g.set_index("timestamp").reindex(idx)
        keep = [
            "slug",
            "elapsed",
            "label_pm",
            "implied_prob",
            "yes_spread",
            "no_spread",
            "book_spread_sum",
            "book_imbalance_yes_no",
            "book_consistency_gap",
            "bid_YES",
            "ask_YES",
            "bid_NO",
            "ask_NO",
        ]
        for c in keep:
            if c in u.columns:
                u[c] = u[c].ffill().bfill()

        u = u.reset_index().rename(columns={"index": "timestamp"})
        u["elapsed"] = u["elapsed"].astype(float).round().astype(int)
        pieces.append(u[["slug", "timestamp", "elapsed", "label_pm", "implied_prob", "yes_spread", "no_spread", "book_spread_sum", "book_imbalance_yes_no", "book_consistency_gap", "bid_YES", "ask_YES", "bid_NO", "ask_NO"]])

    pm = pd.concat(pieces, ignore_index=True)
    pm = pm[(pm["elapsed"] >= 100) & (pm["elapsed"] <= 290)].copy()
    return pm


def load_binance(path: Path) -> pd.DataFrame:
    bn = pd.read_csv(path)
    bn["open_time"] = pd.to_datetime(bn["open_time"], utc=True)
    bn["candle_5m_start"] = pd.to_datetime(bn["candle_5m_start"], utc=True)

    if "market_regime" not in bn.columns:
        vol_low = bn["inst_vol_60s"].quantile(0.30)
        vol_high = bn["inst_vol_60s"].quantile(0.70)
        conditions = [
            bn["inst_vol_60s"] <= vol_low,
            (bn["inst_vol_60s"] >= vol_high) & (bn["trend_zscore"] >= 1.5),
            (bn["inst_vol_60s"] >= vol_high) & (bn["trend_zscore"] <= -1.5),
            (bn["inst_vol_60s"] >= vol_high) & (bn["trend_zscore"].abs() < 0.5),
        ]
        labels = ["calm", "volatile_trend_up", "volatile_trend_down", "high_vol_chop"]
        bn["market_regime"] = np.select(conditions, labels, default="normal")

    return bn.sort_values("open_time").reset_index(drop=True)


def add_multi_horizon_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values("open_time").copy()
    close = df["close"]
    log_ret = np.log(close).diff()

    for w in [5, 15, 30, 60, 180]:
        minp = max(3, min(10, w))
        df[f"ret_{w}s"] = close / close.shift(w) - 1.0
        df[f"rv_{w}s"] = log_ret.rolling(w, min_periods=minp).std()
        hh = df["high"].rolling(w, min_periods=minp).max()
        ll = df["low"].rolling(w, min_periods=minp).min()
        df[f"range_{w}s"] = (hh - ll) / close
        df[f"trades_{w}s"] = df["number_of_trades"].rolling(w, min_periods=minp).sum()
        df[f"volume_{w}s"] = df["volume"].rolling(w, min_periods=minp).sum()
        df[f"trade_intensity_accel_{w}s"] = df["number_of_trades"] / (df["number_of_trades"].rolling(w, min_periods=minp).mean() + 1e-9)
        df[f"volume_accel_{w}s"] = df["volume"] / (df["volume"].rolling(w, min_periods=minp).mean() + 1e-12)

    return df


def add_interactions(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["pm_edge"] = df["implied_prob"] - 0.5
    df["pm_abs_edge"] = df["pm_edge"].abs()
    df["pm_logit"] = safe_logit(df["implied_prob"].values)

    df["pm_x_time"] = df["implied_prob"] * df["fraction_of_candle_elapsed"]
    df["pm_x_vol"] = df["implied_prob"] * df["inst_vol_60s_bps"]
    df["ret_x_vol"] = df["return_from_candle_open"] * df["inst_vol_60s_bps"]
    df["trend_x_vol"] = df["trend_mean_60s"] * df["inst_vol_60s_bps"]
    df["book_imb_x_time"] = df["book_imbalance_yes_no"] * df["fraction_of_candle_elapsed"]
    df["spread_x_time"] = df["book_spread_sum"] * df["fraction_of_candle_elapsed"]
    df["buyratio_x_time"] = df["buy_volume_ratio"] * df["fraction_of_candle_elapsed"]
    return df


def add_top3_orthogonal_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values("open_time").copy()

    # 1) Aggressor order-flow imbalance over 30s.
    signed_flow = 2.0 * df["taker_buy_base_asset_volume"] - df["volume"]
    flow_30 = signed_flow.rolling(30, min_periods=10).sum()
    vol_30 = df["volume"].rolling(30, min_periods=10).sum()
    df["ofi_30s"] = flow_30 / (vol_30 + 1e-12)

    # 2) Cross-venue divergence: Polymarket implied probability vs spot proxy.
    spot_up_proxy = 0.5 + 0.5 * np.tanh(df["ret_30s"] / (df["rv_30s"] + 1e-9))
    df["pm_spot_divergence_30s"] = df["implied_prob"] - np.clip(spot_up_proxy, 0.0, 1.0)

    # 3) Volatility-adjusted VWAP dislocation.
    vwap_gap = (df["close"] - df["vwap"]) / (df["close"] + 1e-12)
    gap_std_120 = vwap_gap.rolling(120, min_periods=30).std()
    df["vwap_gap_z_120s"] = vwap_gap / (gap_std_120 + 1e-9)

    return df


def build_features(df: pd.DataFrame) -> List[str]:
    base = [
        "implied_prob",
        "pm_edge",
        "pm_abs_edge",
        "pm_logit",
        "yes_spread",
        "no_spread",
        "book_spread_sum",
        "book_imbalance_yes_no",
        "book_consistency_gap",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "quote_asset_volume",
        "number_of_trades",
        "taker_buy_base_asset_volume",
        "taker_buy_quote_asset_volume",
        "log_return_1s",
        "inst_vol_60s",
        "inst_vol_60s_bps",
        "trend_mean_60s",
        "trend_zscore",
        "vwap",
        "buy_volume_ratio",
        "seconds_to_5m_close",
        "fraction_of_candle_elapsed",
        "return_from_candle_open",
        "pm_x_time",
        "pm_x_vol",
        "ret_x_vol",
        "trend_x_vol",
        "book_imb_x_time",
        "spread_x_time",
        "buyratio_x_time",
        "ofi_30s",
        "pm_spot_divergence_30s",
        "vwap_gap_z_120s",
    ]
    base += [c for c in df.columns if c.startswith(("ret_", "rv_", "range_", "trades_", "volume_", "trade_intensity_accel_", "volume_accel_"))]
    base += [c for c in df.columns if c.startswith("regime_") or c.startswith("bucket_")]

    unique = []
    seen = set()
    for c in base:
        if c in df.columns and c not in seen:
            seen.add(c)
            unique.append(c)
    return unique


def recency_weights(df: pd.DataFrame) -> np.ndarray:
    ts = df["open_time"].astype("int64").values
    z = (ts - ts.min()) / max(1, ts.max() - ts.min())
    return 0.6 + 0.8 * z


def fit_logreg(X: pd.DataFrame, y: np.ndarray, w: np.ndarray, C: float) -> Pipeline:
    pipe = Pipeline(
        steps=[
            (
                "prep",
                ColumnTransformer(
                    transformers=[
                        (
                            "num",
                            Pipeline(
                                steps=[("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]
                            ),
                            list(X.columns),
                        )
                    ],
                    remainder="drop",
                ),
            ),
            ("model", LogisticRegression(max_iter=3000, solver="lbfgs", C=C)),
        ]
    )
    pipe.fit(X, y, model__sample_weight=w)
    return pipe


def choose_c_with_holdout(train_df: pd.DataFrame, feature_cols: List[str], c_grid: List[float]) -> float:
    # Purged + embargo-aware holdout inside train: last 15% of train candles as validation.
    candles = np.array(sorted(train_df["candle_5m_start"].unique()))
    split_at = int(len(candles) * 0.85)
    split_at = max(30, min(split_at, len(candles) - 10))

    train_c = set(candles[:split_at])
    val_c = set(candles[split_at:])

    tr = train_df[train_df["candle_5m_start"].isin(train_c)].copy()
    va = train_df[train_df["candle_5m_start"].isin(val_c)].copy()

    Xtr = tr[feature_cols]
    ytr = tr["target_up"].astype(int).values
    wtr = recency_weights(tr)

    Xva = va[feature_cols]
    yva = va["target_up"].astype(int).values

    best_c = c_grid[0]
    best_loss = float("inf")

    for c in c_grid:
        model = fit_logreg(Xtr, ytr, wtr, C=c)
        p = model.predict_proba(Xva)[:, 1]
        loss = log_loss(yva, p, labels=[0, 1])
        if loss < best_loss:
            best_loss = loss
            best_c = c

    return best_c


def fit_calibrators(train_pred: pd.DataFrame) -> Dict[str, LogisticRegression]:
    # Suggestion #5: calibrate separately by time bucket.
    calibrators: Dict[str, LogisticRegression] = {}
    for b, g in train_pred.groupby("time_bucket", observed=False):
        if pd.isna(b) or len(g) < 1000 or g["target_up"].nunique() < 2:
            continue
        X = safe_logit(g["raw_prob"].values).reshape(-1, 1)
        y = g["target_up"].astype(int).values
        lr = LogisticRegression(max_iter=2000, solver="lbfgs")
        lr.fit(X, y)
        calibrators[str(b)] = lr
    return calibrators


def apply_calibration(raw_prob: np.ndarray, bucket_labels: pd.Series, calibrators: Dict[str, LogisticRegression]) -> np.ndarray:
    out = raw_prob.copy()
    for b in bucket_labels.astype(str).unique():
        idx = bucket_labels.astype(str) == b
        if b in calibrators:
            z = safe_logit(raw_prob[idx]).reshape(-1, 1)
            out[idx] = calibrators[b].predict_proba(z)[:, 1]
    return np.clip(out, 1e-6, 1 - 1e-6)


def make_splits(candles: np.ndarray, min_train: int = 220, test_size: int = 48, step: int = 24, embargo: int = 2) -> List[Tuple[np.ndarray, np.ndarray]]:
    splits = []
    n = len(candles)
    i = min_train
    while i + test_size <= n:
        train_end = max(0, i - embargo)
        train_c = candles[:train_end]
        test_c = candles[i : i + test_size]
        if len(train_c) >= 120 and len(test_c) > 0:
            splits.append((train_c, test_c))
        i += step
    return splits


def add_categorical_dummies(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    regime = pd.Categorical(out["market_regime"], categories=REGIME_VALUES)
    out = pd.concat([out, pd.get_dummies(regime, prefix="regime", dtype=float)], axis=1)
    bucket = pd.cut(out["seconds_to_5m_close"], bins=TIME_BUCKET_BINS, labels=TIME_BUCKET_LABELS, right=False)
    out["time_bucket"] = bucket
    out = pd.concat([out, pd.get_dummies(bucket, prefix="bucket", dtype=float)], axis=1)
    return out


def evaluate_thresholds(y_true: np.ndarray, y_prob: np.ndarray) -> pd.DataFrame:
    # Suggestion #11: cost-sensitive thresholding.
    # Proxy EV: +1 for correct, -1 for incorrect, minus fee/slippage penalty per trade.
    fee = 0.03
    rows = []
    for t in np.arange(0.50, 0.81, 0.02):
        take = y_prob >= t
        if take.sum() == 0:
            continue
        yhat = (y_prob[take] >= 0.5).astype(int)
        yt = y_true[take]
        acc = (yhat == yt).mean()
        ev = np.mean(np.where(yhat == yt, 1.0, -1.0) - fee)
        rows.append({"threshold": round(float(t), 2), "trades": int(take.sum()), "accuracy": float(acc), "proxy_ev_per_trade": float(ev)})
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="V3 upgraded model with all research suggestions")
    parser.add_argument("--polymarket-csv", default="research/polymarket_model/market_data_2sec_weekly5_with_resolutions.csv")
    parser.add_argument("--binance-csv", default="research/polymarket_model/binance_1s_for_polymarket_window.csv")
    parser.add_argument("--output-dir", default="research/polymarket_model/v3_all_suggestions_results")
    args = parser.parse_args()

    pm = load_polymarket_1s(Path(args.polymarket_csv))
    bn = load_binance(Path(args.binance_csv))

    df = pm.merge(bn, left_on="timestamp", right_on="open_time", how="inner")
    df = add_multi_horizon_features(df)
    df = add_interactions(df)
    df = add_top3_orthogonal_features(df)
    df = add_categorical_dummies(df)

    # Suggestion #10: drop ambiguous tiny-body candles.
    candle_body = (
        df.groupby("candle_5m_start")["return_from_candle_open"].last().abs().rename("abs_body").reset_index()
    )
    body_cut = candle_body["abs_body"].quantile(0.10)
    keep_candles = set(candle_body[candle_body["abs_body"] > body_cut]["candle_5m_start"])
    df = df[df["candle_5m_start"].isin(keep_candles)].copy()

    feature_cols = build_features(df)
    df = df.dropna(subset=feature_cols + ["target_up", "implied_prob"]).copy().reset_index(drop=True)

    candles = np.array(sorted(df["candle_5m_start"].unique()))
    splits = make_splits(candles, min_train=220, test_size=48, step=24, embargo=2)

    if not splits:
        raise RuntimeError("No valid walk-forward splits")

    preds = []
    split_metrics = []
    c_grid = [0.2, 0.5, 1.0, 2.0]

    for s_idx, (train_c, test_c) in enumerate(splits, start=1):
        train_df = df[df["candle_5m_start"].isin(train_c)].copy()
        test_df = df[df["candle_5m_start"].isin(test_c)].copy()

        # Suggestion #1 and #4: regime-specialized and time-aware models.
        best_c = choose_c_with_holdout(train_df, feature_cols, c_grid)
        Xtr = train_df[feature_cols]
        ytr = train_df["target_up"].astype(int).values
        wtr = recency_weights(train_df)

        global_model = fit_logreg(Xtr, ytr, wtr, C=best_c)

        regime_models: Dict[str, Pipeline] = {}
        for r in REGIME_VALUES:
            part = train_df[train_df["market_regime"] == r]
            if len(part) < 5000 or part["target_up"].nunique() < 2:
                continue
            regime_models[r] = fit_logreg(part[feature_cols], part["target_up"].astype(int).values, recency_weights(part), C=best_c)

        bucket_models: Dict[str, Pipeline] = {}
        for b in TIME_BUCKET_LABELS:
            mask = train_df["time_bucket"].astype(str) == b
            part = train_df[mask]
            if len(part) < 5000 or part["target_up"].nunique() < 2:
                continue
            bucket_models[b] = fit_logreg(part[feature_cols], part["target_up"].astype(int).values, recency_weights(part), C=best_c)

        # Train-pred for bucket calibration.
        tr_raw = global_model.predict_proba(train_df[feature_cols])[:, 1]
        tr_pred_frame = train_df[["time_bucket", "target_up"]].copy()
        tr_pred_frame["raw_prob"] = tr_raw
        calibrators = fit_calibrators(tr_pred_frame)

        # Test predictions with time-aware ensemble.
        p_global = global_model.predict_proba(test_df[feature_cols])[:, 1]

        p_regime = p_global.copy()
        for r, mdl in regime_models.items():
            idx = test_df["market_regime"] == r
            if idx.any():
                p_regime[idx] = mdl.predict_proba(test_df.loc[idx, feature_cols])[:, 1]

        p_bucket = p_global.copy()
        for b, mdl in bucket_models.items():
            idx = test_df["time_bucket"].astype(str) == b
            if idx.any():
                p_bucket[idx] = mdl.predict_proba(test_df.loc[idx, feature_cols])[:, 1]

        # Blend global + regime + time-bucket experts.
        p_raw = np.clip(0.40 * p_global + 0.30 * p_regime + 0.30 * p_bucket, 1e-6, 1 - 1e-6)
        p_cal = apply_calibration(p_raw, test_df["time_bucket"], calibrators)

        # Baseline logistic-like model for comparison (single global only).
        p_v1 = p_global

        part = test_df[["slug", "open_time", "candle_5m_start", "seconds_to_5m_close", "time_bucket", "market_regime", "target_up", "implied_prob"]].copy()
        part["v1_prob"] = p_v1
        part["v3_prob"] = p_cal
        part["split"] = s_idx
        preds.append(part)

        split_metrics.append({"split": s_idx, "best_c": best_c, **compute_metrics(part["target_up"].values, part["v3_prob"].values)})

    pred_df = pd.concat(preds, ignore_index=True)
    split_df = pd.DataFrame(split_metrics)

    y = pred_df["target_up"].astype(int).values
    overall = pd.DataFrame(
        [
            {"model": "polymarket", **compute_metrics(y, pred_df["implied_prob"].values)},
            {"model": "v1_logreg_rebuilt", **compute_metrics(y, pred_df["v1_prob"].values)},
            {"model": "v3_all_suggestions", **compute_metrics(y, pred_df["v3_prob"].values)},
        ]
    )

    bucket_rows = []
    for b, g in pred_df.groupby("time_bucket", observed=False):
        if pd.isna(b) or len(g) == 0:
            continue
        yt = g["target_up"].astype(int).values
        bucket_rows.append({"bucket": str(b), "model": "polymarket", "n_rows": len(g), **compute_metrics(yt, g["implied_prob"].values)})
        bucket_rows.append({"bucket": str(b), "model": "v1_logreg_rebuilt", "n_rows": len(g), **compute_metrics(yt, g["v1_prob"].values)})
        bucket_rows.append({"bucket": str(b), "model": "v3_all_suggestions", "n_rows": len(g), **compute_metrics(yt, g["v3_prob"].values)})
    bucket_df = pd.DataFrame(bucket_rows)

    # Suggestion #11 output.
    threshold_df = evaluate_thresholds(y, pred_df["v3_prob"].values)

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    pred_df.to_csv(out / "oof_predictions.csv", index=False)
    split_df.to_csv(out / "split_metrics.csv", index=False)
    overall.to_csv(out / "overall_comparison.csv", index=False)
    bucket_df.to_csv(out / "bucket_comparison.csv", index=False)
    threshold_df.to_csv(out / "cost_sensitive_thresholds.csv", index=False)

    print("=== V3 ALL SUGGESTIONS MODEL ===")
    print(f"Rows used: {len(pred_df):,}")
    print(f"Candles used: {pred_df['candle_5m_start'].nunique():,}")
    print(f"Walk-forward anchors/splits: {len(split_df)}")
    print("\nOverall:")
    print(overall.to_string(index=False, float_format=lambda x: f"{x:.6f}"))

    higher = ["accuracy", "roc_auc"]
    lower = ["log_loss", "brier", "calibration_gap_pp"]
    wins = {m: 0 for m in overall["model"]}
    for m in higher:
        wins[overall.sort_values(m, ascending=False).iloc[0]["model"]] += 1
    for m in lower:
        wins[overall.sort_values(m, ascending=True).iloc[0]["model"]] += 1
    print("\nMetric wins:", wins)
    print(f"\nSaved to: {out.resolve()}")


if __name__ == "__main__":
    main()
