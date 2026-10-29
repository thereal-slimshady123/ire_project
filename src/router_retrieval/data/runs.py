"""Candidate-run I/O: TREC run files and the Contract 1 Parquet layout (query_id, doc_id, score, rank)."""

from pathlib import Path
from typing import Dict, List

import pandas as pd

from router_retrieval.data.schema import Candidate

Run = Dict[str, List[Candidate]]


def read_trec_run(path: Path) -> Run:
    """Read a TREC-format run (`qid Q0 doc rank score tag`); candidates are returned in rank order."""
    run: Run = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            qid, _, doc_id, rank, score, _ = line.split()
            run.setdefault(qid, []).append(Candidate(qid, doc_id, float(score), int(rank)))
    for cands in run.values():
        cands.sort(key=lambda c: c.rank)
    return run


def write_trec_run(run: Run, path: Path, tag: str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for qid, cands in run.items():
            for c in cands:
                f.write(f"{qid} Q0 {c.doc_id} {c.rank} {c.score:.8f} {tag}\n")


def read_run_parquet(path: Path) -> Run:
    """Read a Contract 1 run: columns query_id (str), doc_id (str), score, rank (1-based)."""
    df = pd.read_parquet(path)
    df["query_id"], df["doc_id"] = df["query_id"].astype(str), df["doc_id"].astype(str)
    run: Run = {}
    for qid, doc_id, score, rank in zip(df["query_id"], df["doc_id"], df["score"], df["rank"]):
        run.setdefault(qid, []).append(Candidate(qid, doc_id, float(score), int(rank)))
    for cands in run.values():
        cands.sort(key=lambda c: c.rank)
    return run


def run_to_frame(run: Run) -> pd.DataFrame:
    """Contract 1 frame (query_id/doc_id str, score float32, rank int32) from an in-memory run."""
    rows = [(c.query_id, c.doc_id, c.score, c.rank) for cands in run.values() for c in cands]
    df = pd.DataFrame(rows, columns=["query_id", "doc_id", "score", "rank"])
    return df.astype({"query_id": str, "doc_id": str, "score": "float32", "rank": "int32"})
