"""Generate, retrieve and label a large synthetic training set for the advantage router.

Per dataset (default NFCorpus, SciFact, FiQA; ~30,000 queries in total):
  1. exclude every passage relevant to ANY natural test query (leakage control, improvements.md 3.3);
  2. sample source passages and write one `lexical` (keyword / identifier) and/or one `semantic`
     (paraphrastic question) query per passage, 50/50 by default;
  3. drop empty, duplicate and natural-query-identical queries;
  4. retrieve over the FULL corpus with BM25 and dense, fuse for every alpha on the grid, score nDCG@10
     against the source passage, extract the 18 router features;
  5. write `results/synthetic/{dataset}.parquet` (+ `manifest.json`), same schema as the natural route
     tables, consumed by `scripts/run_experiment2b_advantage.py --synthetic_dir results/synthetic`.

Synthetic queries are only ever used on the training side of a fold; evaluation stays on natural queries.

Needs data/raw, plus torch + transformers (semantic style) and sentence-transformers + faiss (dense).

Usage:
  python scripts/download_data.py
  python scripts/generate_synthetic_queries.py --n_total 30000
  python scripts/generate_synthetic_queries.py --n_total 600 --datasets scifact --lexical_only   # smoke test
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from router_retrieval.data.beir_loader import load_beir_dataset  # noqa: E402
from router_retrieval.features.feature_builder import FeatureBuilder  # noqa: E402
from router_retrieval.retrieval.bm25 import BM25Retriever  # noqa: E402
from router_retrieval.synthetic import (  # noqa: E402
    ExtractiveLexicalGenerator, MixedGenerator, Seq2SeqGenerator, SyntheticQuery, allocate_budget,
    assert_no_leakage, build_route_table, clean_queries, eligible_doc_ids, excluded_doc_ids, plan_queries,
    save_synthetic, style_capacity,
)
from router_retrieval.synthetic.generator import STYLES, generate_low_overlap  # noqa: E402
from router_retrieval.retrieval.bm25 import document_text  # noqa: E402
from router_retrieval.utils.config import load_config  # noqa: E402
from router_retrieval.utils.logging_utils import get_logger  # noqa: E402
from router_retrieval.utils.seeding import seed_everything  # noqa: E402

log = get_logger("synthetic")
E5_PREFIXES = {"query_prefix": "query: ", "passage_prefix": "passage: "}


def parse_args():
    p = argparse.ArgumentParser(description="Generate and label synthetic routing-training queries")
    p.add_argument("--datasets", nargs="+", default=["nfcorpus", "scifact", "fiqa"])
    p.add_argument("--n_total", type=int, default=30000, help="Total queries across datasets (equal split, capped)")
    p.add_argument("--lexical_fraction", type=float, default=0.5)
    p.add_argument("--lexical_only", action="store_true", help="Model-free smoke test: only the lexical style")
    p.add_argument("--generator_model", type=str, default="google/flan-t5-base")
    p.add_argument("--lexical_generator", choices=["extractive", "seq2seq"], default="extractive")
    p.add_argument("--lexical_pool_factor", type=float, default=1.0,
                   help="Sample lexical terms from the top pool_factor*n rarest tokens (>1 = harder; problems.md item 3)")
    p.add_argument("--gen_batch_size", type=int, default=32)
    p.add_argument("--semantic_max_overlap", type=int, default=-1,
                   help="Keep a semantic query only if it shares <= this many analyzed terms with its passage "
                        "(resampling failures); -1 = no filter")
    p.add_argument("--semantic_attempts", type=int, default=8, help="Generation attempts per passage under the filter")
    p.add_argument("--dense_model", type=str, default="intfloat/e5-small-v2")
    p.add_argument("--max_passage_words", type=int, default=300)
    p.add_argument("--min_passage_words", type=int, default=25)
    p.add_argument("--data_dir", type=Path, default=Path("data/raw"))
    p.add_argument("--results_dir", type=Path, default=Path("results"))
    p.add_argument("--output_dir", type=Path, default=Path("results/synthetic"))
    p.add_argument("--config_dir", type=Path, default=Path("configs"))
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def load_dense(args, ds, corpus, ret_cfg):
    """Dense retriever with the cached corpus index (results/indexes/{ds}/{model}) when present."""
    from router_retrieval.retrieval.dense import DenseRetriever

    d = ret_cfg["dense"]
    prefixes = E5_PREFIXES if "e5" in args.dense_model.lower() else {}
    kwargs = dict(batch_size=d["batch_size"], device=d["device"], faiss_index_type=d["faiss_index_type"],
                  max_length=d.get("max_length"), **prefixes)
    index_dir = args.results_dir / "indexes" / ds / args.dense_model.replace("/", "__")
    if (index_dir / "embeddings.npy").exists():
        return DenseRetriever.load(index_dir, args.dense_model, **kwargs)
    dense = DenseRetriever(model_name=args.dense_model, **kwargs)
    log.info(f"[{ds}] encoding {len(corpus)} passages with {args.dense_model} on {dense.device}")
    dense.index(corpus)
    dense.save(index_dir)
    return dense


def main():
    args = parse_args()
    seed_everything(args.seed)
    ret_cfg = load_config(args.config_dir / "retrievers.yaml")
    hyb_cfg = load_config(args.config_dir / "hybrid.yaml")
    alphas = [float(a) for a in hyb_cfg["alpha_grid"]]
    lex_frac = 1.0 if args.lexical_only else args.lexical_fraction

    data, eligible = {}, {}
    for ds in args.datasets:
        corpus, queries, qrels = load_beir_dataset(args.data_dir / ds, "test")
        excluded = excluded_doc_ids(qrels)
        data[ds] = (corpus, queries, excluded)
        eligible[ds] = eligible_doc_ids(corpus, excluded, args.min_passage_words, args.max_passage_words)
        log.info(f"[{ds}] {len(corpus)} passages, {len(excluded)} excluded (relevant to test queries), "
                 f"{len(eligible[ds])} eligible")
    budget = allocate_budget({ds: style_capacity(len(e), lex_frac) for ds, e in eligible.items()}, args.n_total)
    log.info(f"Per-dataset budget: {budget} (requested {args.n_total})")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {"seed": args.seed, "lexical_fraction": lex_frac, "lexical_pool_factor": args.lexical_pool_factor, "semantic_max_overlap": args.semantic_max_overlap, "semantic_attempts": args.semantic_attempts, "generator_model": args.generator_model,
                "dense_model": args.dense_model, "alphas": alphas, "datasets": {}}
    for ds in args.datasets:
        corpus, natural_queries, excluded = data[ds]
        n = budget[ds]
        if n == 0:
            continue
        bm25 = BM25Retriever(k1=ret_cfg["bm25"]["k1"], b=ret_cfg["bm25"]["b"])
        bm25.index(corpus)
        lexical = ExtractiveLexicalGenerator(bm25.term_idf, seed=args.seed, pool_factor=args.lexical_pool_factor)
        seq2seq = None if (args.lexical_only and args.lexical_generator == "extractive") else Seq2SeqGenerator(
            args.generator_model, batch_size=args.gen_batch_size, seed=args.seed)
        gen = MixedGenerator(lexical if args.lexical_generator == "extractive" else seq2seq, seq2seq)

        plan = plan_queries(eligible[ds], n, lex_frac, args.seed)
        assert_no_leakage([d for d, _ in plan], excluded)
        generated = []
        for style in STYLES:
            docs = [d for d, s in plan if s == style]
            if not docs:
                continue
            log.info(f"[{ds}] generating {len(docs)} {style} queries")
            passages = [document_text(corpus[d]) for d in docs]
            if style == "semantic" and args.semantic_max_overlap >= 0:
                texts = generate_low_overlap(gen, passages, style, args.semantic_max_overlap, args.semantic_attempts)
                log.info(f"[{ds}] semantic overlap filter: {sum(bool(t) for t in texts)}/{len(texts)} passages passed")
            else:
                texts = gen.generate(passages, style)
            generated.extend(zip(docs, [style] * len(docs), texts))
        keep = clean_queries([t for _, _, t in generated], (q.text for q in natural_queries.values()))
        queries = [SyntheticQuery(f"syn-{ds}-{i}", t.strip(), d, s, ds)
                   for i, ((d, s, t), ok) in enumerate(zip(generated, keep)) if ok]
        log.info(f"[{ds}] kept {len(queries)}/{len(generated)} queries after cleaning")
        assert_no_leakage([q.source_doc for q in queries], excluded)

        dense = load_dense(args, ds, corpus, ret_cfg)
        k = ret_cfg["dense"].get("top_k", 100)
        table = build_route_table(queries, bm25, dense, alphas, corpus, k, hyb_cfg["normalization"]["method"])
        save_synthetic(table, args.output_dir / f"{ds}.parquet")
        stats = {s: {"n": int((table["style"] == s).sum()), "ndcg_bm25": float(table.loc[table["style"] == s, "ndcg_bm25"].mean()),
                     "ndcg_dense": float(table.loc[table["style"] == s, "ndcg_dense"].mean()),
                     "ndcg_hyb_0.5": float(table.loc[table["style"] == s, "ndcg_hyb_0.5"].mean())}
                 for s in table["style"].unique()}
        manifest["datasets"][ds] = {"n_queries": len(table), "n_excluded_passages": len(excluded),
                                    "n_eligible_passages": len(eligible[ds]), "by_style": stats}
        for s, v in stats.items():
            log.info(f"[{ds}/{s}] n={v['n']} BM25={v['ndcg_bm25']:.3f} dense={v['ndcg_dense']:.3f} hybrid(.5)={v['ndcg_hyb_0.5']:.3f}")
        (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    log.info(f"Wrote synthetic route tables to {args.output_dir}")


if __name__ == "__main__":
    main()
