"""
Forecast evaluation metrics.

Includes both standard point-forecast error metrics (computed on the
median/0.5 quantile) and probabilistic calibration metrics, since a
forecasting model's point accuracy says nothing about whether its
uncertainty bands are trustworthy -- a model can have a great MAE and
still be badly overconfident or underconfident, which matters a lot
when downstream decisions (alerting thresholds, safety stock, etc.)
are based on the quantile bands rather than the median.
"""

from __future__ import annotations

import numpy as np
import torch


def pinball_loss(
    y_true: torch.Tensor, y_pred: torch.Tensor, quantiles: tuple[float, ...]
) -> float:
    """
    Args:
        y_true: (n, horizon, targets)
        y_pred: (n, horizon, targets, num_quantiles)
        quantiles: matching last dim of y_pred
    """
    q = torch.tensor(quantiles, device=y_pred.device, dtype=y_pred.dtype)
    errors = y_true.unsqueeze(-1) - y_pred
    loss = torch.maximum(q * errors, (q - 1) * errors)
    return float(loss.mean())


def point_metrics(
    y_true: torch.Tensor, y_pred_median: torch.Tensor
) -> dict[str, float]:
    """MAE, RMSE, and MAPE on the median forecast.

    MAPE is computed with a small epsilon guard and should be read
    cautiously near-zero targets, where it's known to blow up; for
    series with values near zero, prefer MAE/RMSE/sMAPE.
    """
    err = y_true - y_pred_median
    mae = err.abs().mean()
    rmse = (err.pow(2).mean()).sqrt()
    denom = y_true.abs().clamp_min(1e-6)
    mape = (err.abs() / denom).mean() * 100

    smape_denom = (y_true.abs() + y_pred_median.abs()).clamp_min(1e-6)
    smape = (2 * err.abs() / smape_denom).mean() * 100

    return {
        "mae": float(mae),
        "rmse": float(rmse),
        "mape": float(mape),
        "smape": float(smape),
    }


def quantile_coverage(
    y_true: torch.Tensor, y_pred: torch.Tensor, quantiles: tuple[float, ...]
) -> dict[str, float]:
    """Empirical coverage of each predicted quantile.

    For a well-calibrated model, the fraction of true values falling
    below the predicted q-quantile should equal q (e.g. ~90% of true
    values should fall below the predicted 0.9 quantile). Large
    deviations indicate miscalibration -- e.g. consistently < q means
    the model's upper quantiles are too low (overconfident upside).
    """
    coverage = {}
    y_true_exp = y_true.unsqueeze(-1)
    for i, q in enumerate(quantiles):
        below = (y_true_exp <= y_pred[..., i : i + 1]).float().mean()
        coverage[f"coverage_q{q}"] = float(below)
    return coverage


def interval_width(y_pred: torch.Tensor, low_idx: int, high_idx: int) -> float:
    """Mean width of the prediction interval between two quantile indices.

    Useful alongside coverage: two models can have identical coverage
    but very different (less useful) interval widths. Report both.
    """
    return float((y_pred[..., high_idx] - y_pred[..., low_idx]).mean())


def evaluate_forecast(
    y_true: torch.Tensor,
    y_pred: torch.Tensor,
    quantiles: tuple[float, ...],
    median_idx: int | None = None,
) -> dict[str, float]:
    """Full forecast evaluation report combining point + probabilistic metrics."""
    if median_idx is None:
        median_idx = quantiles.index(0.5) if 0.5 in quantiles else len(quantiles) // 2

    results = point_metrics(y_true, y_pred[..., median_idx])
    results["pinball_loss"] = pinball_loss(y_true, y_pred, quantiles)
    results.update(quantile_coverage(y_true, y_pred, quantiles))

    if len(quantiles) >= 2:
        results["mean_interval_width"] = interval_width(y_pred, 0, len(quantiles) - 1)

    return results
