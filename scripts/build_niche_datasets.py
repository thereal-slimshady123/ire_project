"""Convert real, human-written niche query sets into BEIR format under data/raw/<name>.

Sources (downloaded to data/downloads/ first; no LLM-generated queries):
  wands                   Wayfair product search (real shopper queries, graded Exact/Partial/Irrelevant)
  cosqa                   CoSQA real Bing code-search queries -> Python functions (CoIR release)
  cqadupstack-android     StackExchange Android question titles -> duplicate questions
  cqadupstack-webmasters  StackExchange Webmasters question titles -> duplicate questions
  cqadupstack-{mathematica,english,gaming}  Phase 2 validation set (same format; held out)
  arguana                 ArguAna: argument -> best counter-argument (idebate.org)
  bright-<domain>         BRIGHT: real StackExchange posts -> reasoning-relevant web passages

Corpora larger than --max_docs keep every judged document plus random distractors.

Usage:
  python scripts/build_niche_datasets.py [--datasets wands cosqa ...] [--max_docs 25000]
"""

import argparse
import json
import random
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DL = ROOT / "data" / "downloads"
ALL = ["wands", "cosqa", "cqadupstack-android", "cqadupstack-webmasters", "arguana",
       "bright-biology", "bright-economics", "bright-psychology"]
VALIDATION = ["cqadupstack-mathematica", "cqadupstack-english", "cqadupstack-gaming"]  # Phase 2 held-out set


def write_beir(dst: Path, corpus: dict, queries: dict, qrels: dict, max_docs: int, seed: int) -> None:
    """corpus {id: (title, text)}, queries {id: text}, qrels {qid: {did: rel}} -> BEIR files."""
    qrels = {q: {d: r for d, r in rel.items() if d in corpus} for q, rel in qrels.items() if q in queries}
    qrels = {q: rel for q, rel in qrels.items() if any(r > 0 for r in rel.values())}
    judged = {d for rel in qrels.values() for d, r in rel.items() if r > 0}
    keep = set(corpus)
    if len(corpus) > max_docs:
        rest = sorted(set(corpus) - judged)
        random.Random(seed).shuffle(rest)
        keep = judged | set(rest[: max(0, max_docs - len(judged))])
        qrels = {q: {d: r for d, r in rel.items() if d in keep} for q, rel in qrels.items()}
    (dst / "qrels").mkdir(parents=True, exist_ok=True)
    with open(dst / "corpus.jsonl", "w", encoding="utf-8") as f:
        for d in sorted(keep):
            title, text = corpus[d]
            f.write(json.dumps({"_id": d, "title": title or "", "text": text or ""}) + "\n")
    with open(dst / "queries.jsonl", "w", encoding="utf-8") as f:
        for q in sorted(qrels):
            f.write(json.dumps({"_id": q, "text": queries[q]}) + "\n")
    with open(dst / "qrels" / "test.tsv", "w", encoding="utf-8") as f:
        f.write("query-id\tcorpus-id\tscore\n")
        for q in sorted(qrels):
            for d, r in sorted(qrels[q].items()):
                f.write(f"{q}\t{d}\t{r}\n")
    print(f"[{dst.name}] {len(qrels)} queries, {len(keep)} docs ({len(judged)} positive), "
          f"{sum(len(r) for r in qrels.values())} qrels")


def read_jsonl(path: Path) -> list:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def read_tsv_qrels(path: Path) -> dict:
    out = {}
    df = pd.read_csv(path, sep="\t", dtype=str)
    for q, d, s in df.itertuples(index=False):
        out.setdefault(q, {})[d] = int(s)
    return out


def beir_like(repo: str, drop_query_docs: bool = False):
    src = DL / repo
    corpus = {str(r["_id"]): (r.get("title", ""), r.get("text", "")) for r in read_jsonl(src / "corpus.jsonl")}
    queries = {str(r["_id"]): r["text"] for r in read_jsonl(src / "queries.jsonl")}
    qrels = read_tsv_qrels(src / "qrels" / "test.tsv")
    if drop_query_docs:  # ArguAna: the query argument itself sits in the corpus; BEIR's evaluator skips it
        relevant = {d for rel in qrels.values() for d in rel}
        for q in qrels:
            if q in corpus and q not in relevant:
                del corpus[q]
    return corpus, queries, qrels


def wands():
    src = DL / "wands"
    prod = pd.read_csv(src / "product.csv", sep="\t", dtype=str).fillna("")
    corpus = {r.product_id: (r.product_name, f"{r.product_class}. {r.product_description}")
              for r in prod.itertuples(index=False)}
    q = pd.read_csv(src / "query.csv", sep="\t", dtype=str)
    queries = dict(zip(q["query_id"], q["query"]))
    lab = pd.read_csv(src / "label.csv", sep="\t", dtype=str)
    grade = {"Exact": 2, "Partial": 1, "Irrelevant": 0}
    qrels = {}
    for r in lab.itertuples(index=False):
        qrels.setdefault(r.query_id, {})[r.product_id] = grade[r.label]
    return corpus, queries, qrels


def cosqa():
    src = DL / "CoIR-Retrieval__cosqa"
    c = pd.read_parquet(src / "corpus" / "corpus-00000-of-00001.parquet")
    corpus = {i: ("", t) for i, t in zip(c["_id"], c["text"])}
    qq = pd.read_parquet(src / "queries" / "queries-00000-of-00001.parquet")
    queries = {i: t.strip('"') for i, t in zip(qq["_id"], qq["text"])}
    t = pd.read_parquet(src / "data" / "test-00000-of-00001.parquet")
    qrels = {}
    for q, d, s in t.itertuples(index=False):
        qrels.setdefault(str(q), {})[str(d)] = int(s)
    return corpus, queries, qrels


def bright(domain: str):
    src = DL / "xlangai__BRIGHT"
    docs = pd.read_parquet(src / "documents" / f"{domain}-00000-of-00001.parquet")
    ex = pd.read_parquet(src / "examples" / f"{domain}-00000-of-00001.parquet")
    excluded = {d for ids in ex["excluded_ids"] for d in ids if d != "N/A"}
    corpus = {r.id: ("", r.content) for r in docs.itertuples(index=False) if r.id not in excluded}
    queries = {str(r.id): r.query for r in ex.itertuples(index=False)}
    qrels = {str(r.id): {d: 1 for d in r.gold_ids} for r in ex.itertuples(index=False)}
    return corpus, queries, qrels


BUILDERS = {
    "wands": wands,
    "cosqa": cosqa,
    "cqadupstack-android": lambda: beir_like("mteb__cqadupstack-android"),
    "cqadupstack-webmasters": lambda: beir_like("mteb__cqadupstack-webmasters"),
    "arguana": lambda: beir_like("mteb__arguana", drop_query_docs=True),
    **{v: (lambda v=v: beir_like(f"mteb__{v}")) for v in VALIDATION},
    **{f"bright-{d}": (lambda d=d: bright(d)) for d in ["biology", "economics", "psychology"]},
}


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--datasets", nargs="+", default=ALL, choices=ALL + VALIDATION)
    p.add_argument("--out_dir", type=Path, default=ROOT / "data" / "raw")
    p.add_argument("--max_docs", type=int, default=25000)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    for name in args.datasets:
        corpus, queries, qrels = BUILDERS[name]()
        write_beir(args.out_dir / name, corpus, queries, qrels, args.max_docs, args.seed)


if __name__ == "__main__":
    main()
