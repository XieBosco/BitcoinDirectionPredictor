"""
Unit tests and sanity checks for the Bitcoin 5-minute direction predictor.
"""

from pathlib import Path
import sys
import numpy as np
import pandas as pd
import pytest

# Ensure the parent and directory itself are in sys.path
_current_dir = Path(__file__).resolve().parent
if str(_current_dir) not in sys.path:
    sys.path.insert(0, str(_current_dir))

try:
    from predict_btc_direction import (
        LogisticRegressionWithCI,
        compute_metrics,
        compute_ece,
        compute_bootstrap_cis,
    )
except ModuleNotFoundError:
    from bitcoin_direction_predictor.predict_btc_direction import (
        LogisticRegressionWithCI,
        compute_metrics,
        compute_ece,
        compute_bootstrap_cis,
    )


def test_logistic_regression_with_ci():
    rng = np.random.default_rng(42)
    n = 1000
    X = rng.normal(size=(n, 2))
    logits = 1.5 * X[:, 0] - 0.8 * X[:, 1]
    p_true = 1.0 / (1.0 + np.exp(-logits))
    y = rng.binomial(1, p_true)

    model = LogisticRegressionWithCI(C=1.0, scale_features=True)
    model.fit(X, y)

    # Test predictions with 95% CI on 2D input
    p_pred, p_lower, p_upper, se_logit = model.predict_proba_with_ci(X[:20], confidence_level=0.95)

    assert len(p_pred) == 20
    assert np.all((p_pred >= 0.0) & (p_pred <= 1.0))
    assert np.all((p_lower >= 0.0) & (p_lower <= 1.0))
    assert np.all((p_upper >= 0.0) & (p_upper <= 1.0))
    assert np.all(p_lower <= p_pred)
    assert np.all(p_pred <= p_upper)
    assert np.all(se_logit > 0)

    # Test direction classification
    res_df = model.predict_direction(X[:20], confidence_level=0.95)
    for _, row in res_df.iterrows():
        if row["ci_lower"] > 0.5:
            assert row["direction_call"] == "UP"
            assert row["is_confident"] is True
        elif row["ci_upper"] < 0.5:
            assert row["direction_call"] == "DOWN"
            assert row["is_confident"] is True
        else:
            assert row["direction_call"] == "NEUTRAL"
            assert row["is_confident"] is False


def test_single_sample_and_series_inference():
    """Verify that single-observation 1D array and pd.Series inputs do not crash."""
    rng = np.random.default_rng(42)
    X = rng.normal(size=(200, 4))
    y = rng.binomial(1, 0.5, size=200)

    model = LogisticRegressionWithCI(C=1.0, scale_features=True)
    model.fit(X, y)

    # 1. 1D numpy array (shape: (4,))
    single_vec = np.array([0.5, -0.2, 1.1, -0.4])
    p_pred, p_lower, p_upper, se = model.predict_proba_with_ci(single_vec)
    assert p_pred.shape == (1,)
    assert 0.0 <= p_pred[0] <= 1.0
    assert 0.0 <= p_lower[0] <= p_pred[0] <= p_upper[0] <= 1.0

    dir_df = model.predict_direction(single_vec)
    assert len(dir_df) == 1
    assert dir_df["direction_call"].iloc[0] in ["UP", "DOWN", "NEUTRAL"]

    # 2. pd.Series
    series_input = pd.Series(single_vec, index=[f"x{i}" for i in range(4)])
    p_pred_s, p_lower_s, p_upper_s, se_s = model.predict_proba_with_ci(series_input)
    assert p_pred_s.shape == (1,)
    assert np.isclose(p_pred[0], p_pred_s[0])


def test_cluster_robust_covariance():
    """Verify that cluster sandwich covariance executes and produces positive variances."""
    rng = np.random.default_rng(42)
    n_clusters = 50
    rows_per_cluster = 10
    n = n_clusters * rows_per_cluster
    cluster_ids = np.repeat(np.arange(n_clusters), rows_per_cluster)

    X = rng.normal(size=(n, 3))
    y = rng.binomial(1, 0.5, size=n)

    model = LogisticRegressionWithCI(C=1.0)
    model.fit(X, y, cluster_ids=cluster_ids)

    assert model.cov_params_ is not None
    assert model.cov_params_.shape == (4, 4)  # intercept + 3 features
    # Diagonal variances must be strictly positive
    variances = np.diag(model.cov_params_)
    assert np.all(variances > 0.0)


def test_compute_metrics():
    y_true = np.array([1, 1, 0, 0, 1, 0, 1, 0])
    y_prob = np.array([0.9, 0.8, 0.2, 0.1, 0.7, 0.3, 0.6, 0.4])

    metrics = compute_metrics(y_true, y_prob)
    assert metrics["accuracy"] == 1.0
    assert metrics["precision"] == 1.0
    assert metrics["recall"] == 1.0
    assert metrics["f1"] == 1.0
    assert metrics["roc_auc"] == 1.0
    assert metrics["brier"] < 0.1
    assert 0.0 <= metrics["calibration_gap_pp"] <= 100.0
    assert 0.0 <= metrics["ece_10bin_pp"] <= 100.0


def test_cluster_bootstrap():
    """Verify that cluster bootstrapping resamples blocks properly."""
    rng = np.random.default_rng(42)
    n_clusters = 20
    rows_per_cluster = 5
    n = n_clusters * rows_per_cluster
    cluster_ids = np.repeat(np.arange(n_clusters), rows_per_cluster)

    y_true = rng.binomial(1, 0.6, size=n)
    y_prob = rng.uniform(0.2, 0.8, size=n)

    cis = compute_bootstrap_cis(
        y_true, y_prob, cluster_ids=cluster_ids, n_bootstraps=50, confidence_level=0.95
    )

    for metric in ["accuracy", "roc_auc", "log_loss", "brier", "calibration_gap_pp"]:
        assert metric in cis
        low, high = cis[metric]
        assert not np.isnan(low)
        assert not np.isnan(high)
        assert low <= high


def test_pipeline_outputs_exist():
    # Dynamically locate the directory containing this test file
    base_dir = Path(__file__).resolve().parent

    assert (base_dir / "predict_btc_direction.py").exists()
    assert (base_dir / "test_predictions_with_ci.csv").exists()
    assert (base_dir / "evaluation_metrics.csv").exists()
    assert (base_dir / "time_bucket_metrics.csv").exists()
    assert (base_dir / "MODEL_PERFORMANCE_REPORT.md").exists()

    # Verify CSV files are non-empty and well-formed
    preds = pd.read_csv(base_dir / "test_predictions_with_ci.csv")
    assert len(preds) > 0
    assert "prob_up_pred" in preds.columns
    assert "ci_lower" in preds.columns
    assert "ci_upper" in preds.columns
    assert "direction_call" in preds.columns
    assert "is_confident" in preds.columns


if __name__ == "__main__":
    test_logistic_regression_with_ci()
    test_single_sample_and_series_inference()
    test_cluster_robust_covariance()
    test_compute_metrics()
    test_cluster_bootstrap()
    test_pipeline_outputs_exist()
    print("All unit tests and sanity checks passed successfully!")
