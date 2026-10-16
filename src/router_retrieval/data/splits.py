"""k-fold cross-validation split generation over queries."""

from typing import List, Tuple, Sequence

import numpy as np


def create_kfold_splits(
    query_ids: Sequence[str],
    n_splits: int = 5,
    seed: int = 42,
    shuffle: bool = True,
) -> List[Tuple[List[str], List[str]]]:
    """
    Generate reproducible k-fold cross-validation splits over queries.

    Args:
        query_ids: Ordered sequence of query IDs.
        n_splits: Number of folds (default 5).
        seed: Random seed for deterministic shuffling.
        shuffle: Whether to shuffle before splitting.

    Returns:
        List of (train_query_ids, val_query_ids) tuples for each fold.
    """
    ids = sorted(query_ids)
    if n_splits < 2 or n_splits > len(ids):
        raise ValueError(f"n_splits must be in [2, {len(ids)}], got {n_splits}")
    if shuffle:
        rng = np.random.default_rng(seed)
        ids = [ids[i] for i in rng.permutation(len(ids))]
    folds = np.array_split(np.arange(len(ids)), n_splits)
    splits = []
    for fold in folds:
        held_out = set(fold.tolist())
        train = [ids[i] for i in range(len(ids)) if i not in held_out]
        val = [ids[i] for i in fold]
        splits.append((train, val))
    return splits
