"""
Walk-forward (rolling-origin) cross-validation.

Standard k-fold CV shuffles data and is invalid for time series: it
lets the model train on future data to predict the past, producing
wildly optimistic validation scores. Walk-forward CV instead expands
(or rolls) the training window forward in time across several folds,
always validating on a block strictly after the training block --
the only methodologically sound way to estimate out-of-sample
performance for a forecasting model.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class WalkForwardFold:
    train_slice: slice
    val_slice: slice
    fold_index: int


def walk_forward_splits(
    n: int,
    n_folds: int = 5,
    min_train_size: int | None = None,
    val_size: int | None = None,
    expanding: bool = True,
) -> list[WalkForwardFold]:
    """Generate walk-forward CV folds over a series of length `n`.

    Args:
        n: total number of timesteps available
        n_folds: number of validation folds
        min_train_size: size of the first training block (defaults to
            leaving enough room for n_folds equally sized val blocks)
        val_size: size of each validation block (defaults to splitting
            the remaining data evenly across folds)
        expanding: if True, each fold's training set grows to include
            all prior data (expanding window). If False, training set
            is a fixed-size sliding window (rolling window) -- useful
            when older data is believed to be non-representative of
            current dynamics (concept drift).

    Returns:
        list of WalkForwardFold, each with non-overlapping, strictly
        chronologically-ordered train/val slices (val always after train).
    """
    if val_size is None:
        val_size = max(1, n // (n_folds + 2))
    if min_train_size is None:
        min_train_size = n - n_folds * val_size
        min_train_size = max(min_train_size, val_size)

    folds = []
    train_start = 0
    train_end = min_train_size

    for fold_idx in range(n_folds):
        val_start = train_end
        val_end = min(val_start + val_size, n)
        if val_start >= n:
            break

        folds.append(
            WalkForwardFold(
                train_slice=slice(0 if expanding else train_start, train_end),
                val_slice=slice(val_start, val_end),
                fold_index=fold_idx,
            )
        )

        train_end = val_end
        if not expanding:
            train_start = max(0, train_end - min_train_size)

    return folds
