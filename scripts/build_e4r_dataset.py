"""Build Climate-FEVER for results/preregistration/e4r_climate_fever_20261107.md (BEIR format under data/raw/).

mteb/climate-fever: all test queries with a positive judgment; corpus = every judged document + seed-42 reservoir
distractors up to 50,000 docs, streamed in one pass (same protocol as build_strengthening_datasets.py).

Usage:
  python scripts/build_e4r_dataset.py
"""

import importlib.util
from pathlib import Path

from huggingface_hub import hf_hub_download

ROOT = Path(__file__).resolve().parents[1]


def _load(name, file):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / file)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


bnd = _load("bnd", "build_niche_datasets.py")
bsd = _load("bsd", "build_strengthening_datasets.py")


def main():
    repo = "mteb__climate-fever"
    for f in bsd.FILES:
        hf_hub_download("mteb/climate-fever", f, repo_type="dataset", local_dir=bnd.DL / repo)
    src = bnd.DL / repo
    queries = {str(r["_id"]): r["text"] for r in bnd.read_jsonl(src / "queries.jsonl")}
    qrels = bnd.read_tsv_qrels(src / "qrels" / "test.tsv")
    keep = {q for q, rel in qrels.items() if q in queries and any(v > 0 for v in rel.values())}
    qrels = {q: rel for q, rel in qrels.items() if q in keep}
    queries = {q: t for q, t in queries.items() if q in keep}
    judged = {d for rel in qrels.values() for d in rel}
    corpus = bsd.stream_corpus(src / "corpus.jsonl", judged, bsd.MAX_DOCS - len(judged), bsd.SEED)
    bnd.write_beir(ROOT / "data" / "raw" / "climate-fever", corpus, queries, qrels, max_docs=bsd.MAX_DOCS, seed=bsd.SEED)


if __name__ == "__main__":
    main()
