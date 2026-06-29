import numpy as np
import torch

from timeseries_forge.evaluation.anomaly_metrics import (
    point_adjusted_f1,
    precision_recall_f1,
    roc_auc,
)
from timeseries_forge.evaluation.forecast_metrics import (
    evaluate_forecast,
    pinball_loss,
    point_metrics,
    quantile_coverage,
)
from timeseries_forge.evaluation.walk_forward import walk_forward_splits


def test_point_metrics_zero_error_when_perfect():
    y = torch.randn(10, 5, 2)
    metrics = point_metrics(y, y)
    assert metrics["mae"] == 0.0
    assert metrics["rmse"] == 0.0


def test_pinball_loss_is_nonnegative():
    y_true = torch.randn(8, 4, 2)
    y_pred = torch.randn(8, 4, 2, 3)
    loss = pinball_loss(y_true, y_pred, quantiles=(0.1, 0.5, 0.9))
    assert loss >= 0


def test_quantile_coverage_when_median_equals_truth():
    # if the predicted median always exactly equals the true value, then
    # "true <= predicted median" holds for every sample, so q0.5 coverage == 1.0
    n = 200
    y_true = torch.randn(n, 1, 1)
    low = y_true - 1.0
    high = y_true + 1.0
    y_pred = torch.stack([low, y_true, high], dim=-1)  # (n, 1, 1, 3)
    coverage = quantile_coverage(y_true, y_pred, quantiles=(0.1, 0.5, 0.9))
    assert coverage["coverage_q0.5"] == 1.0


def test_evaluate_forecast_returns_expected_keys():
    y_true = torch.randn(20, 6, 2)
    y_pred = torch.randn(20, 6, 2, 3)
    report = evaluate_forecast(y_true, y_pred, quantiles=(0.1, 0.5, 0.9))
    for key in ["mae", "rmse", "mape", "smape", "pinball_loss", "mean_interval_width"]:
        assert key in report


def test_precision_recall_f1_perfect_prediction():
    y_true = np.array([0, 1, 1, 0, 1])
    y_pred = np.array([0, 1, 1, 0, 1])
    result = precision_recall_f1(y_true, y_pred)
    assert result["precision"] == 1.0
    assert result["recall"] == 1.0
    assert result["f1"] == 1.0


def test_precision_recall_f1_no_predictions():
    y_true = np.array([0, 1, 1, 0])
    y_pred = np.array([0, 0, 0, 0])
    result = precision_recall_f1(y_true, y_pred)
    assert result["precision"] == 0.0
    assert result["recall"] == 0.0
    assert result["f1"] == 0.0


def test_point_adjusted_f1_credits_partial_segment_detection():
    # true anomaly segment spans indices 2-6; model only flags index 3
    y_true = np.array([0, 0, 1, 1, 1, 1, 1, 0, 0])
    y_pred = np.array([0, 0, 0, 1, 0, 0, 0, 0, 0])

    raw = precision_recall_f1(y_true, y_pred)
    adjusted = point_adjusted_f1(y_true, y_pred)

    # raw recall should be low (only 1/5 points flagged)
    assert raw["recall"] < 0.5
    # point-adjusted recall should be perfect since the segment was touched
    assert adjusted["recall"] == 1.0


def test_roc_auc_perfect_separation():
    y_true = np.array([0, 0, 0, 1, 1, 1])
    scores = np.array([0.1, 0.2, 0.3, 0.8, 0.9, 0.95])
    auc = roc_auc(y_true, scores)
    assert auc == 1.0


def test_roc_auc_random_scores_near_half():
    rng = np.random.default_rng(0)
    y_true = rng.integers(0, 2, size=2000)
    scores = rng.random(2000)  # uncorrelated with labels
    auc = roc_auc(y_true, scores)
    assert 0.45 <= auc <= 0.55


def test_walk_forward_splits_are_chronological_and_nonoverlapping():
    folds = walk_forward_splits(n=1000, n_folds=4, expanding=True)
    assert len(folds) == 4
    for fold in folds:
        assert fold.train_slice.stop <= fold.val_slice.start
        assert fold.val_slice.start < fold.val_slice.stop <= 1000

    # expanding window: training set should grow across folds
    train_sizes = [f.train_slice.stop - f.train_slice.start for f in folds]
    assert train_sizes == sorted(train_sizes)


def test_walk_forward_rolling_window_keeps_train_size_bounded():
    folds = walk_forward_splits(n=1000, n_folds=4, expanding=False)
    train_sizes = [f.train_slice.stop - f.train_slice.start for f in folds]
    # rolling window sizes should not grow without bound
    assert max(train_sizes) - min(train_sizes) <= max(train_sizes) * 0.5 + 5
