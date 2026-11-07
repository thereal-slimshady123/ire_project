"""Download and build the fresh held-out set for the amended E4 (results/preregistration/e4_matched_rate_20261107.md).

Downloads mteb/cqadupstack-{physics,programmers,gis} (corpus.jsonl, queries.jsonl, qrels/test.tsv) into
data/downloads/ and converts them with the unchanged beir_like/write_beir from build_niche_datasets.py
(full corpora, official test qrels).

Usage:
  python scripts/build_e4_fresh_datasets.py
"""

import importlib.util
from pathlib import Path

from huggingface_hub import hf_hub_download

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("bnd", ROOT / "scripts" / "build_niche_datasets.py")
bnd = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bnd)

FRESH = ["cqadupstack-physics", "cqadupstack-programmers", "cqadupstack-gis"]
FILES = ["corpus.jsonl", "queries.jsonl", "qrels/test.tsv"]


def main():
    for name in FRESH:
        repo = f"mteb__{name}"
        for f in FILES:
            hf_hub_download(f"mteb/{name}", f, repo_type="dataset", local_dir=bnd.DL / repo)
        corpus, queries, qrels = bnd.beir_like(repo)
        bnd.write_beir(ROOT / "data" / "raw" / name, corpus, queries, qrels, max_docs=10**9, seed=42)


if __name__ == "__main__":
    main()
