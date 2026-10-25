"""Load BEIR corpus/queries/qrels into the common schema."""

import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple
from router_retrieval.data.schema import Document, Query, QRel


def load_corpus(dataset_dir: Path) -> Dict[str, Document]:
    """
    Load corpus.jsonl ({"_id", "title", "text"}) into Documents.

    Args:
        dataset_dir: Path to data/raw/{dataset}/.

    Returns:
        Dict of doc_id to Document.
    """
    corpus_file = Path(dataset_dir) / "corpus.jsonl"
    if not corpus_file.is_file():
        raise FileNotFoundError(f"Corpus file not found: {corpus_file}")

    corpus: Dict[str, Document] = {}
    with open(corpus_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            doc_id = str(data["_id"])
            title = str(data.get("title") or "")
            text = str(data.get("text") or "")
            # Preserve extra metadata excluding core fields
            metadata = {k: v for k, v in data.items() if k not in ("_id", "title", "text")}
            corpus[doc_id] = Document(
                doc_id=doc_id,
                text=text,
                title=title,
                metadata=metadata,
            )
    return corpus


def load_queries(dataset_dir: Path) -> Dict[str, Query]:
    """
    Load queries.jsonl ({"_id", "text"}) into Queries.

    Args:
        dataset_dir: Path to data/raw/{dataset}/.

    Returns:
        Dict of query_id to Query (all splits; filter by the qrels of a split).
    """
    queries_file = Path(dataset_dir) / "queries.jsonl"
    if not queries_file.is_file():
        raise FileNotFoundError(f"Queries file not found: {queries_file}")

    queries: Dict[str, Query] = {}
    with open(queries_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            query_id = str(data["_id"])
            text = str(data.get("text") or "")
            metadata = {k: v for k, v in data.items() if k not in ("_id", "text")}
            queries[query_id] = Query(
                query_id=query_id,
                text=text,
                metadata=metadata,
            )
    return queries


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
    qrels_file = Path(dataset_dir) / "qrels" / f"{split}.tsv"
    if not qrels_file.is_file():
        raise FileNotFoundError(f"Qrels file not found: {qrels_file}")

    qrels: List[QRel] = []
    with open(qrels_file, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) < 3:
                parts = line.split()
            if len(parts) < 3:
                continue

            # Skip header line if present
            if i == 0 and ("query-id" in parts[0].lower() or "query_id" in parts[0].lower()):
                continue

            query_id = str(parts[0].strip())
            doc_id = str(parts[1].strip())
            try:
                score = int(float(parts[2].strip()))
            except ValueError:
                continue

            qrels.append(QRel(query_id=query_id, doc_id=doc_id, score=score))
    return qrels


def load_dataset(
    dataset_dir: Path,
    split: str,
    query_ids: Optional[Sequence[str]] = None,
) -> Tuple[Dict[str, Document], Dict[str, Query], List[QRel]]:
    """
    Load corpus, queries restricted to the split (or specified query_ids), and those qrels.

    Args:
        dataset_dir: Path to data/raw/{dataset}/.
        split: BEIR split name ('train', 'dev', 'test').
        query_ids: Optional explicit filter of query IDs.

    Returns:
        Tuple of (corpus, queries restricted to the split, qrels of the split).
    """
    dataset_dir = Path(dataset_dir)
    corpus = load_corpus(dataset_dir)
    all_queries = load_queries(dataset_dir)

    qrels = load_qrels(dataset_dir, split)
    if query_ids is not None:
        query_id_set = set(query_ids)
        qrels = [qr for qr in qrels if qr.query_id in query_id_set]

    # Restrict queries to those in the split's qrels
    active_query_ids = {qr.query_id for qr in qrels}
    filtered_queries = {qid: all_queries[qid] for qid in active_query_ids if qid in all_queries}

    return corpus, filtered_queries, qrels
