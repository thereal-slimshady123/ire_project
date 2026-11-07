"""Mine a "Real-World Niche Skewed Benchmark" from unmodified human queries (no LLM rewriting).

Every query, qrel and feature comes from the existing per-query tables (Experiment 2 caches: BM25, e5-small
dense, hybrid alpha grid, 18 features), so the output has the same 36-column schema as the synthetic set.

Track 1 (selection without ndcg_*; note: idf_weighted_overlap_ratio does look at the relevant docs D+):
  semantic: query_length >= 5, idf_weighted_overlap_ratio <= 0.35, jaccard_overlap_top10 <= 0.15
  lexical : (max_idf >= P75(max_idf) or entity_density >= 0.25 or rare_term_density >= 0.25),
            idf_weighted_overlap_ratio >= 0.70, bm25_top1_top2_ratio >= P75(ratio), jaccard_overlap_top10 <= 0.20
Track 2 (oracle disagreement):
  semantic: ndcg_dense - ndcg_bm25 >= 0.35 and ndcg_dense >= 0.50
  lexical : ndcg_bm25 - ndcg_dense >= 0.35 and ndcg_bm25 >= 0.50

Both tracks are balanced 50/50 per dataset; datasets with < 30 queries in a bucket are balanced as one pool.

Usage:
  python scripts/build_natural_niche.py [--synthetic_dir <dir with scifact.parquet, nfcorpus.parquet>]
"""

import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from router_retrieval.data.beir_loader import load_corpus, load_qrels, load_queries, qrels_to_dict  # noqa: E402
from router_retrieval.retrieval.bm25 import document_text, tokenize  # noqa: E402

DATASETS = ["cosqa", "wands", "cqadupstack-android", "cqadupstack-webmasters", "nfcorpus", "scifact", "fiqa",
            "arguana"]
CACHES = [ROOT / "results/niche/experiment2/cache", ROOT / "results/experiment2/cache"]
FEATURES = ["query_length", "max_idf", "mean_idf", "rare_term_density", "entity_density", "oov_fraction",
            "jaccard_overlap_top10", "jaccard_overlap_top20", "rank_correlation_spearman", "bm25_top1_score",
            "bm25_top1_top2_gap", "bm25_top1_top2_ratio", "bm25_score_entropy", "dense_top1_score",
            "dense_top1_top2_gap", "dense_score_entropy", "bm25_top10_query_term_coverage",
            "dense_top10_query_term_coverage"]
ALPHAS = [f"{a / 10:.1f}" for a in range(11)]
COLUMNS = (["query_id", "dataset", "ndcg_bm25", "ndcg_dense", *[f"ndcg_hyb_{a}" for a in ALPHAS], *FEATURES,
            "style", "source_doc", "query_text"])
PREFIX = {"semantic": "nat-para", "lexical": "nat-lex"}
MIN_BUCKET = 30


def load_cache(ds: str) -> pd.DataFrame:
    for d in CACHES:
        if (d / f"{ds}.csv").exists():
            df = pd.read_csv(d / f"{ds}.csv", index_col=0, dtype={"query_id": str})
            df.index = df.index.astype(str)
            return df
    raise FileNotFoundError(f"no Experiment 2 cache for {ds}")


def overlap_table(ds: str) -> pd.DataFrame:
    """source_doc, query_text, raw_token_overlap and idf_weighted_overlap_ratio for each test query."""
    path = ROOT / "data/raw" / ds
    corpus = load_corpus(path / "corpus.jsonl")
    qrels = qrels_to_dict(load_qrels(path / "qrels" / "test.tsv"))
    queries = load_queries(path / "queries.jsonl")
    top = {}
    for q, rel in qrels.items():
        best = max(rel.values())
        if best > 0 and q in queries:
            top[q] = sorted(d for d, r in rel.items() if r == best and d in corpus)
    needed = {d for docs in top.values() for d in docs}
    df_count, doc_terms = Counter(), {}
    for did, doc in corpus.items():
        terms = set(tokenize(document_text(doc)))
        df_count.update(terms)
        if did in needed:
            doc_terms[did] = terms
    n = len(corpus)

    def idf(t):
        df = df_count.get(t, 0)
        return math.log(1.0 + (n - df + 0.5) / (df + 0.5))

    rows = {}
    for q, docs in top.items():
        if not docs:
            continue
        q_terms = set(tokenize(queries[q].text))
        d_terms = set().union(*(doc_terms[d] for d in docs))
        shared = q_terms & d_terms
        denom = sum(idf(t) for t in q_terms)
        rows[q] = {"source_doc": docs[0], "query_text": queries[q].text, "raw_token_overlap": len(shared),
                   "idf_weighted_overlap_ratio": sum(idf(t) for t in shared) / denom if denom else 0.0,
                   "n_target_docs": len(docs)}
    return pd.DataFrame.from_dict(rows, orient="index")


def track1(df: pd.DataFrame) -> pd.Series:
    p75_idf, p75_ratio = df["max_idf"].quantile(0.75), df["bm25_top1_top2_ratio"].quantile(0.75)
    sem = ((df["query_length"] >= 5) & (df["idf_weighted_overlap_ratio"] <= 0.35)
           & (df["jaccard_overlap_top10"] <= 0.15))
    anchor = (df["max_idf"] >= p75_idf) | (df["entity_density"] >= 0.25) | (df["rare_term_density"] >= 0.25)
    lex = (anchor & (df["idf_weighted_overlap_ratio"] >= 0.70) & (df["bm25_top1_top2_ratio"] >= p75_ratio)
           & (df["jaccard_overlap_top10"] <= 0.20))
    style = pd.Series(None, index=df.index, dtype=object)
    style[sem], style[lex] = "semantic", "lexical"
    return style, {"P75_max_idf": float(p75_idf), "P75_bm25_top1_top2_ratio": float(p75_ratio)}


def track2(df: pd.DataFrame) -> pd.Series:
    gap = df["ndcg_dense"] - df["ndcg_bm25"]
    style = pd.Series(None, index=df.index, dtype=object)
    style[(gap >= 0.35) & (df["ndcg_dense"] >= 0.50)] = "semantic"
    style[(-gap >= 0.35) & (df["ndcg_bm25"] >= 0.50)] = "lexical"
    return style


def balance(pool: pd.DataFrame, seed: int) -> pd.DataFrame:
    """Downsample the larger style bucket to the size of the smaller one."""
    k = pool["style"].value_counts().reindex(["semantic", "lexical"], fill_value=0).min()
    return pd.concat([g.sample(n=k, random_state=seed) for _, g in pool.groupby("style")]) if k else pool.iloc[:0]


def balance_track(found: pd.DataFrame, seed: int):
    counts = found.groupby(["dataset", "style"]).size().unstack(fill_value=0).reindex(
        columns=["semantic", "lexical"], fill_value=0)
    own = [d for d in counts.index if counts.loc[d].min() >= MIN_BUCKET]
    pooled = [d for d in counts.index if d not in own]
    parts = [balance(found[found["dataset"] == d], seed) for d in own]
    parts.append(balance(found[found["dataset"].isin(pooled)], seed))
    return pd.concat(parts), own, pooled


def to_schema(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["query_id"] = [f"{PREFIX[s]}-{d}-{q}" for s, d, q in zip(out["style"], out["dataset"], out.index)]
    out = out[COLUMNS].reset_index(drop=True)
    for c in COLUMNS:
        out[c] = out[c].astype(str) if c in ("query_id", "dataset", "style", "source_doc", "query_text") \
            else out[c].astype("float64")
    return out


def pct(s: pd.Series) -> dict:
    return {f"p{p}": float(s.quantile(p / 100)) for p in (10, 50, 90)}


def comparison(real: pd.DataFrame, synth: pd.DataFrame) -> dict:
    """P10/P50/P90 per feature and style, plus KS distance real vs synthetic (largest = most shifted)."""
    out = {}
    for style in ["semantic", "lexical"]:
        r, s = real[real["style"] == style], synth[synth["style"] == style]
        rows = {}
        for f in FEATURES:
            ks = stats.ks_2samp(r[f], s[f]) if len(r) and len(s) else None
            rows[f] = {"real": pct(r[f]), "synthetic": pct(s[f]),
                       "ks_stat": float(ks.statistic) if ks else None, "ks_p": float(ks.pvalue) if ks else None}
        out[style] = dict(sorted(rows.items(), key=lambda kv: -(kv[1]["ks_stat"] or 0)))
    return out


def md_table(cmp: dict, label: str) -> list:
    lines = [f"### {label}", ""]
    for style, rows in cmp.items():
        lines += [f"**{style}** (sorted by KS distance; top rows shift most from synthetic)", "",
                  "| Feature | Real P10 / P50 / P90 | Synthetic P10 / P50 / P90 | KS |",
                  "| :--- | :---: | :---: | :---: |"]
        for f, v in rows.items():
            r, s = v["real"], v["synthetic"]
            ks = f"{v['ks_stat']:.2f}" if v["ks_stat"] is not None else "—"
            lines.append(f"| {f} | {r['p10']:.3g} / {r['p50']:.3g} / {r['p90']:.3g} | "
                         f"{s['p10']:.3g} / {s['p50']:.3g} / {s['p90']:.3g} | {ks} |")
        lines.append("")
    return lines


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--datasets", nargs="+", default=DATASETS)
    p.add_argument("--synthetic_dir", type=Path, required=True)
    p.add_argument("--out_dir", type=Path, default=ROOT / "data/processed/natural-niche")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    tables, thresholds = [], {}
    for ds in args.datasets:
        df = load_cache(ds).join(overlap_table(ds), how="inner")
        df["dataset"] = ds
        df["t1"], thresholds[ds] = track1(df)
        df["t2"] = track2(df)
        tables.append(df)
        print(f"[{ds}] {len(df)} queries | T1 sem={int((df.t1 == 'semantic').sum())} "
              f"lex={int((df.t1 == 'lexical').sum())} | T2 sem={int((df.t2 == 'semantic').sum())} "
              f"lex={int((df.t2 == 'lexical').sum())}")
    full = pd.concat(tables)

    synth = pd.concat([pd.read_parquet(args.synthetic_dir / f"{d}.parquet") for d in ("scifact", "nfcorpus")])
    manifest = {"source": "unmodified human queries + official qrels; features from Experiment 2 caches",
                "datasets": {}, "thresholds": thresholds, "balancing": {}, "feature_comparison": {}}
    report = ["# Real-world niche benchmark: feature comparison vs the 288-query synthetic set", ""]
    for track in ("t1", "t2"):
        name = {"t1": "natural", "t2": "contrastive"}[track]
        found = full[full[track].notna()].assign(style=lambda d, t=track: d[t])
        kept, own, pooled = balance_track(found, args.seed)
        manifest["balancing"][name] = {"balanced_per_dataset": own, "balanced_as_pool": pooled,
                                       "total_kept": len(kept),
                                       "kept_by_style": kept["style"].value_counts().to_dict()}
        for ds in args.datasets:
            sub = kept[kept["dataset"] == ds]
            to_schema(sub).to_parquet(args.out_dir / f"{ds}_{name}_niche.parquet", index=False)
            entry = manifest["datasets"].setdefault(ds, {"queries_scanned": int((full["dataset"] == ds).sum())})
            f_ds, k_ds = found[found["dataset"] == ds], sub
            entry[name] = {
                style: {"found": int((f_ds["style"] == style).sum()), "kept": int((k_ds["style"] == style).sum()),
                        "mean_raw_token_overlap": float(f_ds.loc[f_ds["style"] == style, "raw_token_overlap"].mean())
                        if (f_ds["style"] == style).any() else None,
                        "share_raw_overlap_gt0": float((f_ds.loc[f_ds["style"] == style, "raw_token_overlap"] > 0).mean())
                        if (f_ds["style"] == style).any() else None,
                        "mean_idf_weighted_overlap_ratio": float(
                            f_ds.loc[f_ds["style"] == style, "idf_weighted_overlap_ratio"].mean())
                        if (f_ds["style"] == style).any() else None,
                        "mean_ndcg_bm25": float(k_ds.loc[k_ds["style"] == style, "ndcg_bm25"].mean())
                        if (k_ds["style"] == style).any() else None,
                        "mean_ndcg_dense": float(k_ds.loc[k_ds["style"] == style, "ndcg_dense"].mean())
                        if (k_ds["style"] == style).any() else None}
                for style in ("semantic", "lexical")}
        cmp = comparison(kept, synth)
        manifest["feature_comparison"][name] = cmp
        report += md_table(cmp, f"Track {track[1]} ({name}, n = {len(kept)}) vs synthetic (n = {len(synth)})")
    full.drop(columns=["t1", "t2"]).to_csv(args.out_dir / "all_queries_with_overlap.csv")
    with open(args.out_dir / "manifest_natural.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    (args.out_dir / "feature_comparison.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    for name, b in manifest["balancing"].items():
        print(f"{name}: kept {b['total_kept']} {b['kept_by_style']} | own={b['balanced_per_dataset']} "
              f"pooled={b['balanced_as_pool']}")


if __name__ == "__main__":
    main()
