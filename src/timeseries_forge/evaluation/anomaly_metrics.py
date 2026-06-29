"""
Anomaly detection evaluation metrics.

Implements both standard point-wise metrics and the widely-used
"point-adjusted" F1 protocol from the time-series anomaly detection
literature (Xu et al., 2018), which counts an entire contiguous
ground-truth anomalous segment as detected if the model flags *any*
point within it -- a fairer protocol than strict point-wise F1, since
in practice an operator only needs one alert to investigate a whole
incident window, not a flag on every single timestep within it.
Reporting both protocols side by side avoids the well-known pitfall of
point-adjusted F1 looking artificially high on its own.
"""

from __future__ import annotations

import numpy as np


def _binarize(scores: np.ndarray, threshold: float) -> np.ndarray:
    return (scores >= threshold).astype(int)


def precision_recall_f1(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {"precision": precision, "recall": recall, "f1": f1, "tp": tp, "fp": fp, "fn": fn}


def point_adjusted_f1(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Point-adjusted precision/recall/F1.

    If any point in a contiguous true-anomaly segment is predicted
    positive, the entire segment counts as a true positive (all its
    points are relabeled positive for scoring purposes). Predicted
    positives outside any true segment remain false positives as-is.
    """
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    adjusted_pred = y_pred.copy()

    # find contiguous segments of y_true == 1
    diffs = np.diff(np.concatenate([[0], y_true, [0]]))
    starts = np.where(diffs == 1)[0]
    ends = np.where(diffs == -1)[0]

    for s, e in zip(starts, ends):
        if y_pred[s:e].any():
            adjusted_pred[s:e] = 1

    return precision_recall_f1(y_true, adjusted_pred)


def roc_auc(y_true: np.ndarray, scores: np.ndarray) -> float:
    """ROC-AUC computed via the rank-sum (Mann-Whitney U) identity, no sklearn dependency.

    AUC = P(score(positive) > score(negative)), estimated exactly via
    rank statistics rather than numerical integration, so there's no
    threshold-grid resolution error.
    """
    y_true = np.asarray(y_true)
    scores = np.asarray(scores)
    pos = scores[y_true == 1]
    neg = scores[y_true == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")

    ranks = np.argsort(np.argsort(np.concatenate([pos, neg])))
    pos_ranks = ranks[: len(pos)]
    auc = (pos_ranks.sum() - len(pos) * (len(pos) - 1) / 2) / (len(pos) * len(neg))
    return float(auc)


def best_threshold_f1(
    y_true: np.ndarray, scores: np.ndarray, n_thresholds: int = 200, point_adjusted: bool = True
) -> dict[str, float]:
    """Sweeps thresholds over the score distribution and returns the best F1.

    In production you would not have ground-truth labels to pick a
    threshold this way; this is intended for offline evaluation against
    a labeled validation set (e.g. our synthetic data's injected
    anomalies) to report a model's *best achievable* operating point,
    which is then fixed and used as-is at deployment time.
    """
    thresholds = np.quantile(scores, np.linspace(0.5, 0.999, n_thresholds))
    best = {"f1": -1.0, "threshold": 0.0}

    for thresh in thresholds:
        pred = _binarize(scores, thresh)
        metric_fn = point_adjusted_f1 if point_adjusted else precision_recall_f1
        result = metric_fn(y_true, pred)
        if result["f1"] > best["f1"]:
            best = {**result, "threshold": float(thresh)}

    return best


def evaluate_anomaly_detection(
    y_true: np.ndarray, scores: np.ndarray, threshold: float | None = None
) -> dict[str, float]:
    """Full anomaly detection evaluation report.

    If `threshold` is not provided, sweeps for the best point-adjusted
    F1 threshold (offline/labeled evaluation mode). Reports both raw
    and point-adjusted metrics plus ROC-AUC (threshold-independent).
    """
    report: dict[str, float] = {"roc_auc": roc_auc(y_true, scores)}

    if threshold is None:
        best = best_threshold_f1(y_true, scores, point_adjusted=True)
        threshold = best["threshold"]
        report["selected_threshold"] = threshold

    pred = _binarize(scores, threshold)
    raw = precision_recall_f1(y_true, pred)
    adjusted = point_adjusted_f1(y_true, pred)

    report.update({f"raw_{k}": v for k, v in raw.items()})
    report.update({f"point_adjusted_{k}": v for k, v in adjusted.items()})
    return report
