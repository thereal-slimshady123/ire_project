"""Train/dev/test query split resolution (see configs/datasets.yaml)."""

from pathlib import Path
import random
from typing import Dict, List, Optional
from router_retrieval.data.schema import QRel
from router_retrieval.data.beir_loader import load_qrels


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
    unique_ids = sorted(list(set(train_query_ids)))
    if not unique_ids:
        return {"train": [], "dev": []}

    rng = random.Random(seed)
    shuffled = list(unique_ids)
    rng.shuffle(shuffled)

    n_dev = int(round(len(shuffled) * dev_fraction))
    # Ensure at least 1 dev query if dev_fraction > 0 and len > 1
    if dev_fraction > 0 and n_dev == 0 and len(shuffled) > 1:
        n_dev = 1

    dev_ids = sorted(shuffled[:n_dev])
    train_ids = sorted(shuffled[n_dev:])

    return {"train": train_ids, "dev": dev_ids}


def resolve_splits(
    dataset_name: str,
    datasets_config: Dict,
    seed: int = 42,
    dataset_dir: Optional[Path] = None,
) -> Dict[str, List[str]]:
    """
    Return the query IDs for each role ('train', 'dev', 'test') of a dataset,
    using official qrels where available and carve_dev_from_train otherwise.
    Roles with no data (e.g. trec-covid train) map to an empty list.

    Args:
        dataset_name: Key in configs/datasets.yaml.
        datasets_config: Parsed configs/datasets.yaml.
        seed: Random seed for any carved split.
        dataset_dir: Directory where raw dataset resides (default: data/raw/{dataset_name}).

    Returns:
        Dict of role to list of query IDs; the three lists are disjoint.
    """
    dataset_cfg = datasets_config.get("datasets", {}).get(dataset_name, {})
    if dataset_dir is None:
        dataset_dir = Path("data/raw") / dataset_name
    dataset_dir = Path(dataset_dir)

    splits_spec = dataset_cfg.get("splits", {})
    train_split_name = splits_spec.get("train")
    dev_split_name = splits_spec.get("dev")
    test_split_name = splits_spec.get("test")
    dev_fraction = float(dataset_cfg.get("dev_from_train_fraction", 0.0))

    # Resolve test queries
    test_ids: List[str] = []
    if test_split_name:
        test_file = dataset_dir / "qrels" / f"{test_split_name}.tsv"
        if test_file.is_file():
            test_qrels = load_qrels(dataset_dir, test_split_name)
            test_ids = sorted(list({qr.query_id for qr in test_qrels}))

    # Resolve train and dev queries
    train_ids: List[str] = []
    dev_ids: List[str] = []

    if train_split_name:
        train_file = dataset_dir / "qrels" / f"{train_split_name}.tsv"
        if train_file.is_file():
            raw_train_qrels = load_qrels(dataset_dir, train_split_name)
            raw_train_ids = sorted(list({qr.query_id for qr in raw_train_qrels}))

            if dev_split_name:
                # Official dev split exists
                train_ids = raw_train_ids
                dev_file = dataset_dir / "qrels" / f"{dev_split_name}.tsv"
                if dev_file.is_file():
                    dev_qrels = load_qrels(dataset_dir, dev_split_name)
                    dev_ids = sorted(list({qr.query_id for qr in dev_qrels}))
            elif dev_fraction > 0.0:
                # No official dev split; carve from train with specified seed
                carved = carve_dev_from_train(raw_train_ids, dev_fraction=dev_fraction, seed=seed)
                train_ids = carved["train"]
                dev_ids = carved["dev"]
            else:
                train_ids = raw_train_ids

    return {
        "train": train_ids,
        "dev": dev_ids,
        "test": test_ids,
    }


def load_split_qrels(
    dataset_dir: Path,
    dataset_name: str,
    split: str,
    datasets_config: Dict,
    seed: int = 42,
) -> List[QRel]:
    """
    Load qrels for a specific split role ('train', 'dev', 'test'), correctly handling
    carved dev splits where dev qrels come from the train qrels file.

    Args:
        dataset_dir: Path to dataset raw folder.
        dataset_name: Name of dataset.
        split: 'train', 'dev', or 'test'.
        datasets_config: Configuration dictionary.
        seed: Random seed for carved splits.

    Returns:
        List of QRel objects belonging to the specified split role.
    """
    dataset_dir = Path(dataset_dir)
    resolved = resolve_splits(dataset_name, datasets_config, seed=seed, dataset_dir=dataset_dir)
    target_qids = set(resolved.get(split, []))
    if not target_qids:
        return []

    dataset_cfg = datasets_config.get("datasets", {}).get(dataset_name, {})
    splits_spec = dataset_cfg.get("splits", {})

    if split == "dev" and splits_spec.get("dev") is None and splits_spec.get("train"):
        # Read from train qrels file and filter by carved dev query IDs
        all_qrels = load_qrels(dataset_dir, splits_spec["train"])
        return [qr for qr in all_qrels if qr.query_id in target_qids]
    elif split == "train" and splits_spec.get("dev") is None and dataset_cfg.get("dev_from_train_fraction", 0.0) > 0.0:
        # Read from train qrels file and filter out carved dev query IDs
        all_qrels = load_qrels(dataset_dir, splits_spec["train"])
        return [qr for qr in all_qrels if qr.query_id in target_qids]
    else:
        split_name = splits_spec.get(split)
        if not split_name:
            return []
        all_qrels = load_qrels(dataset_dir, split_name)
        return [qr for qr in all_qrels if qr.query_id in target_qids]

    splits_spec = dataset_cfg.get("splits", {})
    train_split_name = splits_spec.get("train")
    dev_split_name = splits_spec.get("dev")
    test_split_name = splits_spec.get("test")
    dev_fraction = float(dataset_cfg.get("dev_from_train_fraction", 0.0))

    # Resolve test queries
    test_ids: List[str] = []
    if test_split_name:
        test_file = dataset_dir / "qrels" / f"{test_split_name}.tsv"
        if test_file.is_file():
            test_qrels = load_qrels(dataset_dir, test_split_name)
            test_ids = sorted(list({qr.query_id for qr in test_qrels}))

    # Resolve train and dev queries
    train_ids: List[str] = []
    dev_ids: List[str] = []

    if train_split_name:
        train_file = dataset_dir / "qrels" / f"{train_split_name}.tsv"
        if train_file.is_file():
            raw_train_qrels = load_qrels(dataset_dir, train_split_name)
            raw_train_ids = sorted(list({qr.query_id for qr in raw_train_qrels}))

            if dev_split_name:
                # Official dev split exists
                train_ids = raw_train_ids
                dev_file = dataset_dir / "qrels" / f"{dev_split_name}.tsv"
                if dev_file.is_file():
                    dev_qrels = load_qrels(dataset_dir, dev_split_name)
                    dev_ids = sorted(list({qr.query_id for qr in dev_qrels}))
            elif dev_fraction > 0.0:
                # No official dev split; carve from train with specified seed
                carved = carve_dev_from_train(raw_train_ids, dev_fraction=dev_fraction, seed=seed)
                train_ids = carved["train"]
                dev_ids = carved["dev"]
            else:
                train_ids = raw_train_ids

    return {
        "train": train_ids,
        "dev": dev_ids,
        "test": test_ids,
    }


def load_split_qrels(
    dataset_dir: Path,
    dataset_name: str,
    split: str,
    datasets_config: Dict,
    seed: int = 42,
) -> List[QRel]:
    """
    Load qrels for a specific split role ('train', 'dev', 'test'), correctly handling
    carved dev splits where dev qrels come from the train qrels file.

    Args:
        dataset_dir: Path to dataset raw folder.
        dataset_name: Name of dataset.
        split: 'train', 'dev', or 'test'.
        datasets_config: Configuration dictionary.
        seed: Random seed for carved splits.

    Returns:
        List of QRel objects belonging to the specified split role.
    """
    dataset_dir = Path(dataset_dir)
    resolved = resolve_splits(dataset_name, datasets_config, seed=seed, dataset_dir=dataset_dir)
    target_qids = set(resolved.get(split, []))
    if not target_qids:
        return []

    dataset_cfg = datasets_config.get("datasets", {}).get(dataset_name, {})
    splits_spec = dataset_cfg.get("splits", {})

    if split == "dev" and splits_spec.get("dev") is None and splits_spec.get("train"):
        # Read from train qrels file and filter by carved dev query IDs
        all_qrels = load_qrels(dataset_dir, splits_spec["train"])
        return [qr for qr in all_qrels if qr.query_id in target_qids]
    elif split == "train" and splits_spec.get("dev") is None and dataset_cfg.get("dev_from_train_fraction", 0.0) > 0.0:
        # Read from train qrels file and filter out carved dev query IDs
        all_qrels = load_qrels(dataset_dir, splits_spec["train"])
        return [qr for qr in all_qrels if qr.query_id in target_qids]
    else:
        split_name = splits_spec.get(split)
        if not split_name:
            return []
        all_qrels = load_qrels(dataset_dir, split_name)
        return [qr for qr in all_qrels if qr.query_id in target_qids]

