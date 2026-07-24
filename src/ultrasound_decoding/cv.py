from __future__ import annotations

import numpy as np


def grouped_cv_splits(groups: np.ndarray, max_folds: int = 10) -> list[tuple[np.ndarray, np.ndarray]]:
    """Create leave-one-cycle-out when possible, otherwise up to 10 group folds."""
    unique_groups = np.unique(groups)
    if len(unique_groups) < 2:
        raise ValueError("Need at least two cycle groups for grouped CV")

    n_folds = len(unique_groups) if len(unique_groups) <= max_folds else max_folds
    folds = [fold for fold in np.array_split(unique_groups, n_folds) if len(fold)]
    splits = []
    for fold_groups in folds:
        test_mask = np.isin(groups, fold_groups)
        train_idx = np.flatnonzero(~test_mask)
        test_idx = np.flatnonzero(test_mask)
        splits.append((train_idx, test_idx))
    return splits
