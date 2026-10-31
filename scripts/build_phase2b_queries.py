"""Phase 2b: build the skewed query set with hand-written zero-overlap paraphrase queries.

Paraphrase-style queries were written by an LLM (Claude) from the passages in results/phase2b/passages.json
(80 eligible passages per dataset, seed 42, none relevant to a natural test query), one question per passage
that the passage answers, without seeing any retrieval result. A question is kept only if it shares no
analyzed term (BM25 tokenizer: stopwords removed, stemmed) with its full passage, a rule on wording only.
The same number of keyword-style queries is drawn (seed 42) from the Phase 2 extractive set. Output:
results/phase2b/synthetic/{dataset}.parquet.

Usage:
  python scripts/build_phase2b_queries.py
"""

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from router_retrieval.data.beir_loader import load_beir_dataset  # noqa: E402
from router_retrieval.retrieval.bm25 import BM25Retriever, document_text  # noqa: E402
from router_retrieval.synthetic import (  # noqa: E402
    SyntheticQuery, assert_no_leakage, build_route_table, excluded_doc_ids, load_synthetic, save_synthetic,
)
from router_retrieval.synthetic.generator import content_overlap  # noqa: E402
from router_retrieval.utils.config import load_config  # noqa: E402
from router_retrieval.utils.logging_utils import get_logger  # noqa: E402


def _load_script(name: str, file: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / file)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


gen_script = _load_script("gen_script", "generate_synthetic_queries.py")
log = get_logger("phase2b")
DATASETS = ["scifact", "nfcorpus"]
MAX_OVERLAP = 0


def parse_args():
    p = argparse.ArgumentParser(description="Phase 2b: zero-overlap paraphrase + keyword query set")
    p.add_argument("--datasets", nargs="+", default=DATASETS)
    p.add_argument("--data_dir", type=Path, default=Path("data/raw"))
    p.add_argument("--phase2b_dir", type=Path, default=Path("results/phase2b"))
    p.add_argument("--phase2_synthetic_dir", type=Path, default=Path("results/phase2/synthetic"))
    p.add_argument("--results_dir", type=Path, default=Path("results"))
    p.add_argument("--config_dir", type=Path, default=Path("configs"))
    p.add_argument("--dense_model", type=str, default="intfloat/e5-small-v2")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()
    ret_cfg = load_config(args.config_dir / "retrievers.yaml")
    hyb_cfg = load_config(args.config_dir / "hybrid.yaml")
    alphas = [float(a) for a in hyb_cfg["alpha_grid"]]
    passages = json.loads((args.phase2b_dir / "passages.json").read_text(encoding="utf-8"))
    out_dir = args.phase2b_dir / "synthetic"
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = {"max_overlap": MAX_OVERLAP, "datasets": {}}
    for ds in args.datasets:
        questions = json.loads((args.phase2b_dir / f"questions_{ds}.json").read_text(encoding="utf-8"))
        if len(questions) != len(passages[ds]):
            raise RuntimeError(f"[{ds}] {len(questions)} questions for {len(passages[ds])} passages")
        corpus, _, qrels = load_beir_dataset(args.data_dir / ds, "test")
        kept, dropped = [], []
        for i, (p, q) in enumerate(zip(passages[ds], questions)):
            overlap = content_overlap(q, document_text(corpus[p["doc_id"]]))
            (kept if overlap <= MAX_OVERLAP else dropped).append(
                (SyntheticQuery(f"para-{ds}-{i}", q, p["doc_id"], "semantic", ds), overlap))
        assert_no_leakage([q.source_doc for q, _ in kept], excluded_doc_ids(qrels))
        log.info(f"[{ds}] {len(kept)}/{len(questions)} paraphrase queries pass the zero-overlap rule")

        bm25 = BM25Retriever(k1=ret_cfg["bm25"]["k1"], b=ret_cfg["bm25"]["b"])
        bm25.index(corpus)
        dense = gen_script.load_dense(args, ds, corpus, ret_cfg)
        para = build_route_table([q for q, _ in kept], bm25, dense, alphas, corpus,
                                 ret_cfg["dense"].get("top_k", 100), hyb_cfg["normalization"]["method"])
        lexical = load_synthetic(args.phase2_synthetic_dir / f"{ds}.parquet")
        lexical = lexical[lexical["style"] == "lexical"].sample(n=len(para), random_state=args.seed)
        table = pd.concat([para, lexical[para.columns]])
        save_synthetic(table, out_dir / f"{ds}.parquet")
        manifest["datasets"][ds] = {"paraphrase_written": len(questions), "paraphrase_kept": len(kept),
                                    "dropped_overlap": [{"query": q.text, "overlap": o} for q, o in dropped],
                                    "keyword": len(lexical)}
        for st, g in table.groupby("style"):
            log.info(f"[{ds}/{st}] n={len(g)} BM25={g.ndcg_bm25.mean():.3f} dense={g.ndcg_dense.mean():.3f} "
                     f"hybrid(.5)={g['ndcg_hyb_0.5'].mean():.3f}")
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
