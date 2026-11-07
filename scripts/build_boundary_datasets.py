"""Build the held-out sets for results/preregistration/e8_boundary_20261107.md (BEIR format under data/raw/).

  musique          dgslibisey/MuSiQue, musique_ans_v1.0_dev.jsonl: every question; corpus = union of all questions'
                   paragraphs (deduplicated by title + text); relevant = is_supporting paragraphs.
  2wikimultihopqa  xanhho/2WikiMultihopQA, dev.parquet: 3,000 questions sampled with seed 42; corpus = union of the
                   sampled questions' context paragraphs (title + joined sentences, deduplicated); relevant = context
                   paragraphs whose title appears in supporting_facts.
  dbpedia          mteb/dbpedia: all test queries with a positive judgment; corpus = every judged doc + seed-42
                   reservoir distractors up to 50,000 (as build_strengthening_datasets.py).

Usage:
  python scripts/build_boundary_datasets.py
"""

import ast
import hashlib
import importlib.util
import json
import random
from pathlib import Path

import pandas as pd
from huggingface_hub import hf_hub_download

ROOT = Path(__file__).resolve().parents[1]


def _load(name, file):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / file)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


bnd = _load("bnd", "build_niche_datasets.py")
bsd = _load("bsd", "build_strengthening_datasets.py")
OUT, SEED, N_2WIKI = ROOT / "data" / "raw", 42, 3000


def doc_id(title: str, text: str) -> str:
    return "d" + hashlib.md5(f"{title}\n{text}".encode("utf-8")).hexdigest()[:20]


def parse(s):
    if not isinstance(s, str):
        return s
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        return ast.literal_eval(s)


def musique():
    path = hf_hub_download("dgslibisey/MuSiQue", "musique_ans_v1.0_dev.jsonl", repo_type="dataset",
                           local_dir=bnd.DL / "musique")
    corpus, queries, qrels = {}, {}, {}
    for r in bnd.read_jsonl(Path(path)):
        q = str(r["id"])
        queries[q] = r["question"]
        for p in r["paragraphs"]:
            d = doc_id(p["title"], p["paragraph_text"])
            corpus[d] = (p["title"], p["paragraph_text"])
            if p["is_supporting"]:
                qrels.setdefault(q, {})[d] = 1
    bnd.write_beir(OUT / "musique", corpus, queries, qrels, max_docs=10**9, seed=SEED)


def twowiki():
    path = hf_hub_download("xanhho/2WikiMultihopQA", "dev.parquet", repo_type="dataset", local_dir=bnd.DL / "2wikimultihopqa")
    df = pd.read_parquet(path)
    keep = set(random.Random(SEED).sample(sorted(df["_id"].astype(str)), N_2WIKI))
    corpus, queries, qrels = {}, {}, {}
    for r in df.to_dict("records"):
        q = str(r["_id"])
        if q not in keep:
            continue
        queries[q] = r["question"]
        support = {t for t, _ in parse(r["supporting_facts"])}
        for title, sents in parse(r["context"]):
            text = " ".join(s.strip() for s in sents)
            d = doc_id(title, text)
            corpus[d] = (title, text)
            if title in support:
                qrels.setdefault(q, {})[d] = 1
    bnd.write_beir(OUT / "2wikimultihopqa", corpus, queries, qrels, max_docs=10**9, seed=SEED)


def dbpedia():
    repo = "mteb__dbpedia"
    for f in bsd.FILES:
        hf_hub_download("mteb/dbpedia", f, repo_type="dataset", local_dir=bnd.DL / repo)
    src = bnd.DL / repo
    queries = {str(r["_id"]): r["text"] for r in bnd.read_jsonl(src / "queries.jsonl")}
    qrels = bnd.read_tsv_qrels(src / "qrels" / "test.tsv")
    keep = {q for q, rel in qrels.items() if q in queries and any(v > 0 for v in rel.values())}
    qrels = {q: rel for q, rel in qrels.items() if q in keep}
    queries = {q: t for q, t in queries.items() if q in keep}
    judged = {d for rel in qrels.values() for d in rel}
    corpus = bsd.stream_corpus(src / "corpus.jsonl", judged, bsd.MAX_DOCS - len(judged), SEED)
    bnd.write_beir(OUT / "dbpedia", corpus, queries, qrels, max_docs=bsd.MAX_DOCS, seed=SEED)


if __name__ == "__main__":
    musique()
    twowiki()
    dbpedia()
