"""Load BEIR corpus/queries/qrels into the common schema."""

from pathlib import Path
from typing import Dict, List, Tuple
from router_retrieval.data.schema import Document, Query, QRel


def load_corpus(dataset_dir: Path) -> Dict[str, Document]:
    """
    Load corpus.jsonl ({"_id", "title", "text"}) into Documents.

    Args:
        dataset_dir: Path to data/raw/{dataset}/.

    Returns:
        Dict of doc_id to Document.
    """
    raise NotImplementedError


def load_queries(dataset_dir: Path) -> Dict[str, Query]:
    """
    Load queries.jsonl ({"_id", "text"}) into Queries.

    Args:
        dataset_dir: Path to data/raw/{dataset}/.

    Returns:
        Dict of query_id to Query (all splits; filter by the qrels of a split).
    """
    raise NotImplementedError


def load_qrels(dataset_dir: Path, split: str) -> List[QRel]:
    """
    Load qrels/{split}.tsv (query-id, corpus-id, score). Rows with score 0 are kept;
    pytrec_eval treats them as non-relevant.

    Args:
        dataset_dir: Path to data/raw/{dataset}/.
        split: BEIR qrels split name ('train', 'dev' or 'test').

    Returns:
        List of QRel objects.
    """
    raise NotImplementedError


def load_dataset(dataset_dir: Path, split: str) -> Tuple[Dict[str, Document], Dict[str, Query], List[QRel]]:
    """
    Load corpus, the queries that appear in the split's qrels, and those qrels.

    Returns:
        Tuple of (corpus, queries restricted to the split, qrels of the split).
    """
    raise NotImplementedError
