"""Per-document learned fusion vs fixed-alpha hybrid (the design implication of E4).

Fixed-alpha hybrid applies one BM25/dense weight to every document of every query; per-query routing picks one list per
query. If the winning retriever is a property of the (query, document) pair, the decision should be made per document.

Candidates: union of the BM25 and dense top-100 (the same pool the hybrid fuses). Per (query, document) features:
  bm25 / dense min-max scores as used by the hybrid (0 if not retrieved), raw scores, ranks (101 if not retrieved),
  retrieved-by flags, rank difference, and the query's 18 router features (exp2 cache).
Model: LightGBM LambdaRank with router.yaml's GBM settings (100 trees, lr 0.05, 31 leaves), fixed a priori; labels are
the graded qrels. 5-fold CV by query within each dataset (seed 42, Exp 2 folds). Baseline: hybrid at alpha* chosen on
the training folds (Exp 2's rule). Metric nDCG@10; paired bootstrap 10,000, seed 42, pooled over datasets.
Secondary: leave-one-dataset-out transfer (train on the other datasets, test on all of the target).

Usage:
  python scripts/run_learned_fusion.py --datasets <ds ...> --exp2_dir <dir ...> --output_dir <dir> [--dense_model intfloat/e5-small-v2]
"""

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from router_retrieval.data.splits import create_kfold_splits  # noqa: E402
from router_retrieval.evaluation.ranking_metrics import ndcg_at_k  # noqa: E402
from router_retrieval.evaluation.significance import paired_bootstrap_test  # noqa: E402
from router_retrieval.retrieval.hybrid import BM25_TRUNCATED_FLOOR, normalize_for_fusion  # noqa: E402
from router_retrieval.utils.config import load_config  # noqa: E402


def _load(name, file):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / file)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


exp1 = _load("exp1", "run_experiment1_hybrid_degradation.py")
exp2 = _load("exp2", "run_experiment2_heldout_routing.py")
lrel = _load("lrel", "analyze_label_reliability.py")

K, SEED, N_BOOT = 100, 42, 10000
DOC_FEATURES = ["b_norm", "d_norm", "b_raw", "d_raw", "b_rank", "d_rank", "in_b", "in_d", "rank_diff"]
FEATURE_SETS = {  # "doc": corpus-independent per-document signals only (a learned, nonlinear alpha)
    "all": DOC_FEATURES + list(exp2.FEATURES),
    "doc": ["b_norm", "d_norm", "b_rank", "d_rank", "in_b", "in_d", "rank_diff"],
    "doc+query": ["b_norm", "d_norm", "b_rank", "d_rank", "in_b", "in_d", "rank_diff"] + list(exp2.FEATURES),
}
FEATS = FEATURE_SETS["all"]


def candidate_table(ds: str, cache: pd.DataFrame, dense_slug: str) -> pd.DataFrame:
    run_dir = ROOT / "results" / "runs" / ds
    bm25 = exp1.read_trec_run(run_dir / "bm25_k1-1.2_b-0.75.trec")
    dense = exp1.read_trec_run(run_dir / f"dense_{dense_slug}.trec")
    qrels = lrel.read_qrels(lrel.qrels_path(ds))
    rows = []
    for q in cache.index:
        b, d = bm25.get(q, []), dense.get(q, [])
        bn = normalize_for_fusion(b, K, "min_max", BM25_TRUNCATED_FLOOR)
        dn = normalize_for_fusion(d, K, "min_max")
        braw = {c.doc_id: c.score for c in b}
        draw = {c.doc_id: c.score for c in d}
        brank = {c.doc_id: c.rank for c in b}
        drank = {c.doc_id: c.rank for c in d}
        bmin = min(braw.values()) if braw else 0.0
        dmin = min(draw.values()) if draw else 0.0
        rel = qrels.get(q, {})
        for doc in sorted(set(braw) | set(draw)):
            rows.append((q, doc, bn.get(doc, 0.0), dn.get(doc, 0.0), braw.get(doc, bmin), draw.get(doc, dmin),
                         brank.get(doc, K + 1), drank.get(doc, K + 1), doc in braw, doc in draw,
                         brank.get(doc, K + 1) - drank.get(doc, K + 1), max(rel.get(doc, 0), 0)))
    t = pd.DataFrame(rows, columns=["query_id", "doc_id", *DOC_FEATURES, "label"])
    t[["in_b", "in_d"]] = t[["in_b", "in_d"]].astype(int)
    t = t.join(cache[exp2.FEATURES], on="query_id")
    t.insert(0, "dataset", ds)
    return t


def fit(train: pd.DataFrame, router_cfg: dict):
    import lightgbm as lgb

    g = router_cfg["models"]["gbm"]
    train = train.sort_values(["dataset", "query_id"], kind="stable")
    groups = train.groupby(["dataset", "query_id"], sort=False).size().to_numpy()
    m = lgb.LGBMRanker(objective="lambdarank", n_estimators=g["n_estimators"], learning_rate=g["learning_rate"],
                       num_leaves=g["num_leaves"], random_state=SEED, deterministic=True, force_row_wise=True, verbose=-1)
    return m.fit(train[FEATS], train["label"], group=groups)


def score_queries(model, test: pd.DataFrame, qrels: dict) -> pd.Series:
    test = test.assign(s=model.predict(test[FEATS]))
    out = {}
    for q, g in test.groupby("query_id", sort=False):
        docs = g.sort_values(["s", "doc_id"], ascending=[False, True])["doc_id"].tolist()
        out[q] = ndcg_at_k(docs, qrels.get(q, {}), 10)
    return pd.Series(out)


def apply_frozen(args, caches, cands, alphas, router_cfg, exp_cfg, slug):
    """Train one model on all source datasets; apply unchanged to new datasets (no target labels used).

    Primary baseline: hybrid at the source alpha (best pooled mean on the source datasets; same information).
    Secondary baseline: hybrid at alpha* tuned on the target's own training folds (Exp 2's rule; uses target labels).
    """
    model = fit(pd.concat(cands.values()), router_cfg)
    src_alpha = max(alphas, key=lambda x: pd.concat([c[f"ndcg_hyb_{x}"] for c in caches.values()]).mean())
    rows = []
    for ds in args.apply_datasets:
        src = next(d for d in args.apply_exp2_dir if (d / "cache" / f"{ds}.csv").exists())
        c = pd.read_csv(src / "cache" / f"{ds}.csv", index_col=0, dtype={"query_id": str})
        c.index = c.index.astype(str)
        s = score_queries(model, candidate_table(ds, c, slug), lrel.read_qrels(lrel.qrels_path(ds))).reindex(c.index).fillna(0.0)
        tuned = pd.Series(index=c.index, dtype=float)
        for tr, te in create_kfold_splits(list(c.index), exp_cfg["k_folds"], exp_cfg["seed"]):
            a = max(alphas, key=lambda x: c.loc[tr, f"ndcg_hyb_{x}"].mean())
            tuned.loc[te] = c.loc[te, f"ndcg_hyb_{a}"].values
        rows.append(pd.DataFrame({"dataset": ds, "learned_fusion": s.values, "hybrid_src_alpha": c[f"ndcg_hyb_{src_alpha}"].values,
                                  "hybrid_tuned": tuned.values}, index=[f"{ds}:{q}" for q in c.index]))
        print(f"[apply -> {ds}] vs src alpha {(s - c[f'ndcg_hyb_{src_alpha}']).mean():+.4f}, vs tuned {(s - tuned).mean():+.4f}")
    per_q = pd.concat(rows)
    b = lambda df, base: paired_bootstrap_test(df["learned_fusion"].values, df[base].values, N_BOOT, SEED)  # noqa: E731
    block = lambda df: {"n": int(len(df)), **{k: float(df[k].mean()) for k in ["learned_fusion", "hybrid_src_alpha", "hybrid_tuned"]},  # noqa: E731
                        "vs_src_alpha": b(df, "hybrid_src_alpha"), "vs_tuned": b(df, "hybrid_tuned")}
    res = {"features": args.features, "train_datasets": args.datasets, "source_alpha": src_alpha, "pooled": block(per_q),
           "per_dataset": {ds: block(g) for ds, g in per_q.groupby("dataset", sort=False)}}
    p = res["pooled"]["vs_src_alpha"]
    res["verdict"] = "ACCEPTED" if p["ci_95_low"] > 0 else "REJECTED" if p["ci_95_high"] < 0 else "INCONCLUSIVE"
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    per_q.to_csv(out / "per_query_apply.csv")
    (out / "summary.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    ci = lambda t: f"{t['delta_mean']:+.4f} [{t['ci_95_low']:+.4f}, {t['ci_95_high']:+.4f}]"  # noqa: E731
    L = ["# Per-document learned fusion: frozen model applied to new datasets", "",
         f"One LambdaRank model ({args.features} features) trained on {len(args.datasets)} source datasets; applied unchanged. "
         f"Source α = {src_alpha}.", "", f"**Verdict (vs hybrid at source α): {res['verdict']}**", "",
         "| Dataset | n | Learned fusion | Hybrid (source α) | Hybrid (α* tuned on target) | Learned − hybrid (source α) | Learned − hybrid (tuned α*) |",
         "| :--- | :---: | :---: | :---: | :---: | :---: | :---: |"]
    for ds, r in [*res["per_dataset"].items(), ("**pooled**", res["pooled"])]:
        L.append(f"| {ds} | {r['n']} | {r['learned_fusion']:.4f} | {r['hybrid_src_alpha']:.4f} | {r['hybrid_tuned']:.4f} "
                 f"| {ci(r['vs_src_alpha'])} | {ci(r['vs_tuned'])} |")
    (out / "tables.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--datasets", nargs="+", required=True)
    ap.add_argument("--exp2_dir", type=Path, nargs="+", required=True)
    ap.add_argument("--dense_model", default="intfloat/e5-small-v2")
    ap.add_argument("--output_dir", type=Path, required=True)
    ap.add_argument("--no_transfer", action="store_true")
    ap.add_argument("--no_within", action="store_true", help="skip within-dataset CV (transfer only)")
    ap.add_argument("--features", choices=list(FEATURE_SETS), default="all")
    ap.add_argument("--apply_datasets", nargs="+",
                    help="frozen mode: train ONE model on all --datasets, apply it unchanged to these datasets")
    ap.add_argument("--apply_exp2_dir", type=Path, nargs="+")
    args = ap.parse_args()
    global FEATS
    FEATS = FEATURE_SETS[args.features]
    cfg = lambda n: load_config(ROOT / "configs" / n)  # noqa: E731
    router_cfg, exp_cfg = cfg("router.yaml"), cfg("experiment.yaml")
    alphas = [float(a) for a in cfg("hybrid.yaml")["alpha_grid"]]
    slug = args.dense_model.replace("/", "__")

    caches, cands, qrels = {}, {}, {}
    for ds in args.datasets:
        src = next(d for d in args.exp2_dir if (d / "cache" / f"{ds}.csv").exists())
        c = pd.read_csv(src / "cache" / f"{ds}.csv", index_col=0, dtype={"query_id": str})
        c.index = c.index.astype(str)
        caches[ds], qrels[ds] = c, lrel.read_qrels(lrel.qrels_path(ds))
        cands[ds] = candidate_table(ds, c, slug)
        print(f"[{ds}] {len(c)} queries, {len(cands[ds])} candidates, "
              f"{(cands[ds].label > 0).groupby(cands[ds].query_id).any().mean():.1%} queries with a relevant candidate")

    if args.apply_datasets:
        return apply_frozen(args, caches, cands, alphas, router_cfg, exp_cfg, slug)

    rows = []
    for ds in ([] if args.no_within else args.datasets):
        c, t = caches[ds], cands[ds]
        for fold, (tr, te) in enumerate(create_kfold_splits(list(c.index), exp_cfg["k_folds"], exp_cfg["seed"])):
            a = max(alphas, key=lambda x: c.loc[tr, f"ndcg_hyb_{x}"].mean())
            m = fit(t[t.query_id.isin(tr)], router_cfg)
            s = score_queries(m, t[t.query_id.isin(te)], qrels[ds]).reindex(te).fillna(0.0)
            rows.append(pd.DataFrame({"dataset": ds, "fold": fold, "alpha_star": a, "hybrid": c.loc[te, f"ndcg_hyb_{a}"].values,
                                      "learned_fusion": s.values, "bm25": c.loc[te, "ndcg_bm25"].values,
                                      "dense": c.loc[te, "ndcg_dense"].values}, index=[f"{ds}:{q}" for q in te]))
    per_q = pd.concat(rows) if rows else pd.DataFrame(columns=["dataset", "hybrid", "learned_fusion"])
    b = lambda df: paired_bootstrap_test(df["learned_fusion"].values, df["hybrid"].values, N_BOOT, SEED)  # noqa: E731
    res = {} if per_q.empty else {"pooled": {"n": len(per_q), "hybrid": float(per_q.hybrid.mean()), "learned_fusion": float(per_q.learned_fusion.mean()),
                      "lf_minus_hybrid": b(per_q)},
           "per_dataset": {ds: {"n": int(len(g)), "hybrid": float(g.hybrid.mean()), "learned_fusion": float(g.learned_fusion.mean()),
                                "lf_minus_hybrid": b(g)} for ds, g in per_q.groupby("dataset", sort=False)}}
    res["features"] = args.features
    if not args.no_transfer and len(args.datasets) > 1:
        res["transfer"], t_rows = {}, []
        for ds in args.datasets:
            others = pd.concat([cands[o] for o in args.datasets if o != ds])
            src_alpha = max(alphas, key=lambda x: pd.concat([caches[o][f"ndcg_hyb_{x}"] for o in args.datasets if o != ds]).mean())
            s = score_queries(fit(others, router_cfg), cands[ds], qrels[ds]).reindex(caches[ds].index).fillna(0.0)
            h_src = caches[ds][f"ndcg_hyb_{src_alpha}"]
            res["transfer"][ds] = {"lf_minus_hybrid_src_alpha": paired_bootstrap_test(s.values, h_src.values, N_BOOT, SEED),
                                   "source_alpha": src_alpha, "n": int(len(s))}
            t_rows.append(pd.DataFrame({"dataset": ds, "learned_fusion": s.values, "hybrid_src_alpha": h_src.values},
                                       index=[f"{ds}:{q}" for q in s.index]))
            print(f"[transfer -> {ds}] {(s - h_src).mean():+.4f}")
        tq = pd.concat(t_rows)
        tq.to_csv(args.output_dir / "per_query_transfer.csv") if args.output_dir.mkdir(parents=True, exist_ok=True) is None else None
        res["transfer_pooled"] = paired_bootstrap_test(tq.learned_fusion.values, tq.hybrid_src_alpha.values, N_BOOT, SEED)
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    if not per_q.empty:
        per_q.to_csv(out / "per_query_heldout.csv")
    (out / "summary.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    ci = lambda t: f"{t['delta_mean']:+.4f} [{t['ci_95_low']:+.4f}, {t['ci_95_high']:+.4f}]"  # noqa: E731
    L = ["# Per-document learned fusion vs fixed-α hybrid", "", f"Features: {args.features}. Dense model {args.dense_model}. Held-out nDCG@10, 5-fold CV by query "
         "within each dataset; hybrid α* tuned on the training folds.", "",
         "| Dataset | n | Hybrid (α*) | Learned fusion | Learned − hybrid [95% CI] | Transfer: learned − hybrid (source α) |",
         "| :--- | :---: | :---: | :---: | :---: | :---: |"]
    for ds in [*args.datasets, "**pooled**"]:
        r = res["pooled"] if ds == "**pooled**" and "pooled" in res else res.get("per_dataset", {}).get(ds)
        tr = res.get("transfer", {}).get(ds, {}).get("lf_minus_hybrid_src_alpha") or (res.get("transfer_pooled") if ds == "**pooled**" else None)
        L.append(f"| {ds} | {r['n'] if r else res.get('transfer', {}).get(ds, {}).get('n', '—')} | {r['hybrid']:.4f} | " if r else f"| {ds} | — | — | ")
        L[-1] += (f"{r['learned_fusion']:.4f} | {ci(r['lf_minus_hybrid'])} " if r else "— | — ") + f"| {ci(tr) if tr else '—'} |"
    (out / "tables.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
