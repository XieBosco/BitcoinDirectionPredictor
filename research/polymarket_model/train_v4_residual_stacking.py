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
from sklearn.model_selection import GroupKFold
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


def load_data(pm_path: Path, bn_path: Path) -> pd.DataFrame:
    pm = pd.read_csv(pm_path)
    pm["timestamp"] = pd.to_datetime(pm["timestamp_log"], unit="s", utc=True)
    pm["elapsed"] = pd.to_numeric(pm["elapsed"], errors="coerce")
    pm["implied_prob"] = (
        pd.to_numeric(pm["bid_YES"], errors="coerce") + pd.to_numeric(pm["ask_YES"], errors="coerce")
    ) / 2.0
    pm["implied_prob"] = pm["implied_prob"].clip(0.0, 1.0)

    pm["yes_spread"] = (pd.to_numeric(pm["ask_YES"], errors="coerce") - pd.to_numeric(pm["bid_YES"], errors="coerce")).clip(lower=0)
    pm["no_spread"] = (pd.to_numeric(pm["ask_NO"], errors="coerce") - pd.to_numeric(pm["bid_NO"], errors="coerce")).clip(lower=0)
    pm["book_spread_sum"] = pm["yes_spread"] + pm["no_spread"]
    pm["book_imbalance_yes_no"] = ((pd.to_numeric(pm["bid_YES"], errors="coerce") + pd.to_numeric(pm["ask_YES"], errors="coerce")) - (pd.to_numeric(pm["bid_NO"], errors="coerce") + pd.to_numeric(pm["ask_NO"], errors="coerce"))) / 2.0

    pm = pm[["slug", "timestamp", "elapsed", "implied_prob", "yes_spread", "no_spread", "book_spread_sum", "book_imbalance_yes_no"]].copy()

    bn = pd.read_csv(bn_path)
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

    df = pm.merge(bn, left_on="timestamp", right_on="open_time", how="inner")
    df = df[(df["elapsed"] >= 100) & (df["elapsed"] <= 290)].copy()

    regime = pd.Categorical(df["market_regime"], categories=REGIME_VALUES)
    bucket = pd.cut(df["seconds_to_5m_close"], bins=TIME_BUCKET_BINS, labels=TIME_BUCKET_LABELS, right=False)
    df = pd.concat([df, pd.get_dummies(regime, prefix="regime", dtype=float), pd.get_dummies(bucket, prefix="bucket", dtype=float)], axis=1)

    # Multi-horizon features
    close = df["close"]
    log_ret = np.log(close).diff()
    for w in [5, 15, 30, 60, 180]:
        minp = max(3, min(10, w))
        df[f"ret_{w}s"] = close / close.shift(w) - 1.0
        df[f"rv_{w}s"] = log_ret.rolling(w, min_periods=minp).std()
        df[f"trades_{w}s"] = df["number_of_trades"].rolling(w, min_periods=minp).sum()
        df[f"volume_accel_{w}s"] = df["volume"] / (df["volume"].rolling(w, min_periods=minp).mean() + 1e-12)

    df["pm_logit"] = safe_logit(df["implied_prob"].values)
    df["pm_edge"] = df["implied_prob"] - 0.5
    df["pm_x_time"] = df["implied_prob"] * df["fraction_of_candle_elapsed"]
    df["pm_x_vol"] = df["implied_prob"] * df["inst_vol_60s_bps"]

    return df.sort_values(["candle_5m_start", "open_time"]).reset_index(drop=True)


def make_base_features(df: pd.DataFrame) -> List[str]:
    cols = [
        "close",
        "volume",
        "number_of_trades",
        "log_return_1s",
        "inst_vol_60s",
        "inst_vol_60s_bps",
        "trend_mean_60s",
        "trend_zscore",
        "vwap",
        "buy_volume_ratio",
        "return_from_candle_open",
        "seconds_to_5m_close",
        "fraction_of_candle_elapsed",
    ]
    cols += [c for c in df.columns if c.startswith(("ret_", "rv_", "trades_", "volume_accel_"))]
    cols = [c for c in cols if c in df.columns]
    seen = set()
    out = []
    for c in cols:
        if c not in seen:
            seen.add(c)
            out.append(c)
    return out


def make_meta_features(df: pd.DataFrame) -> List[str]:
    cols = [
        "implied_prob",
        "pm_logit",
        "pm_edge",
        "yes_spread",
        "no_spread",
        "book_spread_sum",
        "book_imbalance_yes_no",
        "pm_x_time",
        "pm_x_vol",
        "base_lr_prob",
        "base_lr_logit",
        "prob_gap_pm_minus_lr",
        "prob_prod_pm_lr",
    ]
    cols += [c for c in df.columns if c.startswith("regime_") or c.startswith("bucket_")]
    cols = [c for c in cols if c in df.columns]
    return cols


def make_pipeline(feature_cols: List[str], C: float = 1.0) -> Pipeline:
    return Pipeline(
        steps=[
            (
                "prep",
                ColumnTransformer(
                    transformers=[
                        ("num", Pipeline(steps=[("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]), feature_cols)
                    ],
                    remainder="drop",
                ),
            ),
            ("model", LogisticRegression(max_iter=3000, solver="lbfgs", C=C)),
        ]
    )


def recency_weights(df: pd.DataFrame) -> np.ndarray:
    ts = df["open_time"].astype("int64").values
    z = (ts - ts.min()) / max(1, ts.max() - ts.min())
    return 0.7 + 0.6 * z


def make_walkforward_splits(candles: np.ndarray, min_train: int = 220, test_size: int = 48, step: int = 24, embargo: int = 2):
    out = []
    i = min_train
    n = len(candles)
    while i + test_size <= n:
        train_end = max(0, i - embargo)
        train = candles[:train_end]
        test = candles[i : i + test_size]
        if len(train) >= 120 and len(test) > 0:
            out.append((train, test))
        i += step
    return out


def inner_oof_base_lr(train_df: pd.DataFrame, base_cols: List[str]) -> np.ndarray:
    groups = train_df["candle_5m_start"].astype(str).values
    y = train_df["target_up"].astype(int).values
    X = train_df[base_cols]

    ug = np.unique(groups)
    n_splits = min(4, len(ug))
    if n_splits < 2:
        raise RuntimeError("Not enough groups for inner OOF")

    oof = np.full(len(train_df), np.nan)
    gkf = GroupKFold(n_splits=n_splits)

    for tr_idx, va_idx in gkf.split(X, y, groups=groups):
        part_tr = train_df.iloc[tr_idx]
        model = make_pipeline(base_cols, C=0.7)
        model.fit(X.iloc[tr_idx], y[tr_idx], model__sample_weight=recency_weights(part_tr))
        oof[va_idx] = model.predict_proba(X.iloc[va_idx])[:, 1]

    # Any residual NaN (edge cases) fallback to pm prob
    fallback = train_df["implied_prob"].values
    oof = np.where(np.isnan(oof), fallback, oof)
    return np.clip(oof, 1e-6, 1 - 1e-6)


def run_v4(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    # Label cleaning (same spirit as v3): drop tiny-body candles.
    body = df.groupby("candle_5m_start")["return_from_candle_open"].last().abs()
    cut = body.quantile(0.10)
    keep = set(body[body > cut].index)
    df = df[df["candle_5m_start"].isin(keep)].copy().reset_index(drop=True)

    base_cols = make_base_features(df)
    df = df.dropna(subset=base_cols + ["target_up", "implied_prob"]).copy().reset_index(drop=True)

    candles = np.array(sorted(df["candle_5m_start"].unique()))
    splits = make_walkforward_splits(candles)
    if not splits:
        raise RuntimeError("No walk-forward splits")

    pred_parts = []
    split_rows = []

    for s_idx, (train_c, test_c) in enumerate(splits, start=1):
        train_df = df[df["candle_5m_start"].isin(train_c)].copy().reset_index(drop=True)
        test_df = df[df["candle_5m_start"].isin(test_c)].copy().reset_index(drop=True)

        # Base LR model (rebuilt logistic).
        base_model = make_pipeline(base_cols, C=0.7)
        base_model.fit(train_df[base_cols], train_df["target_up"].astype(int).values, model__sample_weight=recency_weights(train_df))
        p_base_test = np.clip(base_model.predict_proba(test_df[base_cols])[:, 1], 1e-6, 1 - 1e-6)

        # Meta model training uses leakage-safe inner OOF base probs.
        p_base_train_oof = inner_oof_base_lr(train_df, base_cols)

        meta_train = train_df.copy()
        meta_train["base_lr_prob"] = p_base_train_oof
        meta_train["base_lr_logit"] = safe_logit(meta_train["base_lr_prob"].values)
        meta_train["prob_gap_pm_minus_lr"] = meta_train["implied_prob"] - meta_train["base_lr_prob"]
        meta_train["prob_prod_pm_lr"] = meta_train["implied_prob"] * meta_train["base_lr_prob"]

        meta_cols = make_meta_features(meta_train)
        meta_model = make_pipeline(meta_cols, C=0.9)
        meta_model.fit(meta_train[meta_cols], meta_train["target_up"].astype(int).values, model__sample_weight=recency_weights(meta_train))

        # Build meta test frame and predict V4.
        meta_test = test_df.copy()
        meta_test["base_lr_prob"] = p_base_test
        meta_test["base_lr_logit"] = safe_logit(meta_test["base_lr_prob"].values)
        meta_test["prob_gap_pm_minus_lr"] = meta_test["implied_prob"] - meta_test["base_lr_prob"]
        meta_test["prob_prod_pm_lr"] = meta_test["implied_prob"] * meta_test["base_lr_prob"]

        p_v4 = np.clip(meta_model.predict_proba(meta_test[meta_cols])[:, 1], 1e-6, 1 - 1e-6)

        # Conservative shrink toward Polymarket to reduce over-correction risk.
        p_v4 = np.clip(0.65 * p_v4 + 0.35 * meta_test["implied_prob"].values, 1e-6, 1 - 1e-6)

        out = meta_test[["slug", "open_time", "candle_5m_start", "seconds_to_5m_close", "target_up", "implied_prob"]].copy()
        out["v1_prob"] = p_base_test
        out["v4_prob"] = p_v4
        out["split"] = s_idx
        pred_parts.append(out)

        split_rows.append({"split": s_idx, **compute_metrics(out["target_up"].values, out["v4_prob"].values)})

    pred_df = pd.concat(pred_parts, ignore_index=True)
    split_df = pd.DataFrame(split_rows)

    y = pred_df["target_up"].astype(int).values
    overall = pd.DataFrame(
        [
            {"model": "polymarket", **compute_metrics(y, pred_df["implied_prob"].values)},
            {"model": "v1_logreg_rebuilt", **compute_metrics(y, pred_df["v1_prob"].values)},
            {"model": "v4_residual_stacking", **compute_metrics(y, pred_df["v4_prob"].values)},
        ]
    )

    return pred_df, split_df, overall


def main() -> None:
    parser = argparse.ArgumentParser(description="V4 residual-stacking model")
    parser.add_argument("--polymarket-csv", default="research/polymarket_model/market_data_2sec_weekly5_with_resolutions.csv")
    parser.add_argument("--binance-csv", default="research/polymarket_model/binance_1s_for_polymarket_window.csv")
    parser.add_argument("--output-dir", default="research/polymarket_model/v4_residual_results")
    args = parser.parse_args()

    df = load_data(Path(args.polymarket_csv), Path(args.binance_csv))
    pred_df, split_df, overall = run_v4(df)

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    pred_df.to_csv(out / "oof_predictions.csv", index=False)
    split_df.to_csv(out / "split_metrics.csv", index=False)
    overall.to_csv(out / "overall_comparison.csv", index=False)

    print("=== V4 RESIDUAL STACKING ===")
    print(f"Rows used: {len(pred_df):,}")
    print(f"Candles used: {pred_df['candle_5m_start'].nunique():,}")
    print(f"Splits: {len(split_df)}")
    print("\nOverall comparison:")
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
