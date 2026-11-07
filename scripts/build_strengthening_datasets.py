"""Download and build the fresh held-out sets for results/preregistration/phase2_strengthening_20261107.md.

  forums     mteb/cqadupstack-{stats,wordpress,tex}: full corpora, official test qrels (as for the Phase 2 validation set)
  non-forum  mteb/{scidocs,nq,hotpotqa}: all test queries with a positive judgment; corpus = every judged document +
             distractors up to 50,000 docs (as the Phase 1 quora protocol), drawn by a seed-42 reservoir sample in one
             streaming pass so multi-GB corpora never sit in memory

Conversion uses the unchanged beir_like / write_beir of scripts/build_niche_datasets.py.

Usage:
  python scripts/build_strengthening_datasets.py
"""

import importlib.util
import random
from pathlib import Path

from huggingface_hub import hf_hub_download

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("bnd", ROOT / "scripts" / "build_niche_datasets.py")
bnd = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bnd)

FORUMS = ["cqadupstack-stats", "cqadupstack-wordpress", "cqadupstack-tex"]
NON_FORUM = ["scidocs", "nq", "hotpotqa"]
FILES = ["corpus.jsonl", "queries.jsonl", "qrels/test.tsv"]
MAX_DOCS, SEED = 50000, 42


def main():
    for name in FORUMS + NON_FORUM:
        repo = f"mteb__{name}"
        for f in FILES:
            hf_hub_download(f"mteb/{name}", f, repo_type="dataset", local_dir=bnd.DL / repo)
        if name in FORUMS:
            corpus, queries, qrels = bnd.beir_like(repo)
            bnd.write_beir(ROOT / "data" / "raw" / name, corpus, queries, qrels, max_docs=10**9, seed=SEED)
            continue
        src = bnd.DL / repo
        queries = {str(r["_id"]): r["text"] for r in bnd.read_jsonl(src / "queries.jsonl")}
        qrels = bnd.read_tsv_qrels(src / "qrels" / "test.tsv")
        keep = {q for q, rel in qrels.items() if q in queries and any(r > 0 for r in rel.values())}
        qrels = {q: rel for q, rel in qrels.items() if q in keep}
        queries = {q: t for q, t in queries.items() if q in keep}
        judged = {d for rel in qrels.values() for d in rel}
        corpus = stream_corpus(src / "corpus.jsonl", judged, MAX_DOCS - len(judged), SEED)
        bnd.write_beir(ROOT / "data" / "raw" / name, corpus, queries, qrels, max_docs=MAX_DOCS, seed=SEED)


def stream_corpus(path: Path, judged: set, n_distractors: int, seed: int) -> dict:
    """One pass over corpus.jsonl: every judged doc + a seeded reservoir sample of n_distractors other docs."""
    import json

    rng, keep, reservoir, seen = random.Random(seed), {}, [], 0
    with open(path, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            d, doc = str(r["_id"]), (r.get("title", ""), r.get("text", ""))
            if d in judged:
                keep[d] = doc
                continue
            seen += 1
            if len(reservoir) < n_distractors:
                reservoir.append((d, doc))
            else:
                j = rng.randrange(seen)
                if j < n_distractors:
                    reservoir[j] = (d, doc)
    keep.update(reservoir)
    return keep


if __name__ == "__main__":
    main()
