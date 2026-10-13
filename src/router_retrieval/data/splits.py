"""Train/dev/test query split resolution (see configs/datasets.yaml)."""

from typing import Dict, List


def carve_dev_from_train(
    train_query_ids: List[str],
    dev_fraction: float = 0.2,
    seed: int = 42,
) -> Dict[str, List[str]]:
    """
    Hold out a seeded random fraction of train queries as dev, for datasets
    that ship no dev qrels (SciFact).

    Args:
        train_query_ids: Query IDs from the official train qrels.
        dev_fraction: Fraction of train queries moved to dev.
        seed: Random seed.

    Returns:
        Dict with keys 'train' and 'dev' mapping to disjoint, sorted query ID lists.
    """
    raise NotImplementedError


def resolve_splits(dataset_name: str, datasets_config: Dict, seed: int = 42) -> Dict[str, List[str]]:
    """
    Return the query IDs for each role ('train', 'dev', 'test') of a dataset,
    using official qrels where available and carve_dev_from_train otherwise.
    Roles with no data (e.g. trec-covid train) map to an empty list.

    Args:
        dataset_name: Key in configs/datasets.yaml.
        datasets_config: Parsed configs/datasets.yaml.
        seed: Random seed for any carved split.

    Returns:
        Dict of role to list of query IDs; the three lists are disjoint.
    """
    raise NotImplementedError
