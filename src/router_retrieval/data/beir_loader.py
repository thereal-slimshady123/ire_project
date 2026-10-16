"""BEIR dataset loader into common schema."""

import csv
import json
from pathlib import Path
from typing import Dict, List, Tuple
from router_retrieval.data.schema import Document, Query, QRel


def load_beir_dataset(
    dataset_dir: Path,
    split: str = "test",
) -> Tuple[Dict[str, Document], Dict[str, Query], List[QRel]]:
    """
    Load BEIR corpus, queries, and qrels into common schema dataclasses.

    Only queries that have at least one judgment in the requested split are returned
    (BEIR ships queries.jsonl with all splits mixed together).

    Args:
        dataset_dir: Path to directory containing corpus.jsonl, queries.jsonl, and qrels/.
        split: Split name (e.g., 'test', 'dev', 'train').

    Returns:
        Tuple of:
            - corpus: Mapping from doc_id to Document
            - queries: Mapping from query_id to Query
            - qrels: List of QRel relevance judgments
    """
    dataset_dir = Path(dataset_dir)
    corpus = load_corpus(dataset_dir / "corpus.jsonl")
    qrels = load_qrels(dataset_dir / "qrels" / f"{split}.tsv")
    judged = {q.query_id for q in qrels}
    queries = {qid: q for qid, q in load_queries(dataset_dir / "queries.jsonl").items() if qid in judged}
    return corpus, queries, qrels


def load_corpus(corpus_path: Path) -> Dict[str, Document]:
    """Load corpus.jsonl into Document objects."""
    corpus: Dict[str, Document] = {}
    with open(corpus_path, encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            corpus[str(row["_id"])] = Document(
                doc_id=str(row["_id"]),
                text=row.get("text", "") or "",
                title=row.get("title") or None,
                metadata=row.get("metadata", {}) or {},
            )
    return corpus


def load_queries(queries_path: Path) -> Dict[str, Query]:
    """Load queries.jsonl into Query objects."""
    queries: Dict[str, Query] = {}
    with open(queries_path, encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            queries[str(row["_id"])] = Query(
                query_id=str(row["_id"]),
                text=row["text"],
                metadata=row.get("metadata", {}) or {},
            )
    return queries


def load_qrels(qrels_path: Path) -> List[QRel]:
    """Load BEIR TSV qrels (query-id, corpus-id, score with header row) into QRel objects."""
    qrels: List[QRel] = []
    with open(qrels_path, encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        next(reader)  # header
        for row in reader:
            if len(row) < 3:
                continue
            qrels.append(QRel(query_id=str(row[0]), doc_id=str(row[1]), relevance=int(row[2])))
    return qrels


def qrels_to_dict(qrels: List[QRel]) -> Dict[str, Dict[str, int]]:
    """Convert QRel list to nested {query_id: {doc_id: relevance}} mapping."""
    out: Dict[str, Dict[str, int]] = {}
    for q in qrels:
        out.setdefault(q.query_id, {})[q.doc_id] = q.relevance
    return out
