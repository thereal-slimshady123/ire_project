"""Phase 2 validation of explanatory hypotheses E1-E2 (+ secondary E3), exactly as pre-registered in
results/preregistration/phase2_explanatory_validation_20261107.md.

Only points the pinned functions (diagnose_router_lock.py, analyze_niche_existence.py, Experiment 2d) at the
held-out validation datasets; no definition is changed here.

Usage:
  python scripts/run_phase2_validation.py --synthetic_dir <dir with scifact.parquet, nfcorpus.parquet>
"""

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from router_retrieval.evaluation.significance import paired_bootstrap_test  # noqa: E402


def _load(name, file):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / file)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


diag = _load("diag", "diagnose_router_lock.py")
nich = _load("nich", "analyze_niche_existence.py")

DATASETS = ["cqadupstack-mathematica", "cqadupstack-english", "cqadupstack-gaming"]
VAL = ROOT / "results/phase2_validation"
PREREG = "results/preregistration/phase2_explanatory_validation_20261107.md"
DELTA, N_BOOT, SEED = 0.005, 10000, 42
FEATURES = diag.FEATURES


def boot(a, b):
    return paired_bootstrap_test(np.asarray(a, float), np.asarray(b, float), N_BOOT, SEED)


def ci90(d):
    rng = np.random.default_rng(SEED)
    d = np.asarray(d, float)
    m = d[rng.integers(0, len(d), (N_BOOT, len(d)))].mean(axis=1)
    return np.percentile(m, [5, 95])


def fmt(t):
    return f"{t['delta_mean']:+.4f} [{t['ci_95_low']:+.4f}, {t['ci_95_high']:+.4f}]"


def winner_auc(df, hyb="ndcg_hyb_0.6"):
    """Best AUC' = max(AUC, 1-AUC) over the 18 features, dense-winner vs BM25-winner; bootstrap 95% CI."""
    h = df[hyb]
    dense = (df["ndcg_dense"] > h) & (df["ndcg_dense"] > df["ndcg_bm25"])
    bm25 = (df["ndcg_bm25"] > h) & (df["ndcg_bm25"] > df["ndcg_dense"])
    W = pd.concat([df[dense].assign(y=1), df[bm25].assign(y=0)])
    aucs = {f: roc_auc_score(W["y"], W[f]) for f in FEATURES}
    best = max(aucs, key=lambda f: max(aucs[f], 1 - aucs[f]))
    rng = np.random.default_rng(SEED)
    pos, neg = W[W.y == 1][best].to_numpy(), W[W.y == 0][best].to_numpy()
    bs = []
    for _ in range(2000):
        p, n = pos[rng.integers(0, len(pos), len(pos))], neg[rng.integers(0, len(neg), len(neg))]
        a = roc_auc_score(np.r_[np.ones(len(p)), np.zeros(len(n))], np.r_[p, n])
        bs.append(max(a, 1 - a))
    a = max(aucs[best], 1 - aucs[best])
    lo, hi = np.percentile(bs, [2.5, 97.5])
    return best, a, lo, hi, int(dense.sum()), int(bm25.sum())


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--synthetic_dir", type=Path, required=True)
    args = p.parse_args()

    cache = pd.concat([pd.read_csv(VAL / "experiment2/cache" / f"{d}.csv", index_col=0, dtype={"query_id": str})
                       for d in DATASETS])
    cache.index = cache["dataset"] + ":" + cache.index.astype(str)
    pub = pd.read_csv(VAL / "experiment2d/per_query_heldout.csv", index_col=0)
    pub.index = pub.index.astype(str)
    s2d = json.loads((VAL / "experiment2d/summary.json").read_text(encoding="utf-8"))

    # ---------------- E1 ----------------
    synth = pd.concat([pd.read_parquet(args.synthetic_dir / f"{d}.parquet") for d in ("scifact", "nfcorpus")])
    synth = synth.set_index("query_id")
    R_syn = pd.DataFrame({"bm25": synth["ndcg_bm25"], "dense": synth["ndcg_dense"],
                          "hybrid": synth["ndcg_hyb_0.6"]}, index=synth.index)
    pred_fn, gate = diag.fit_2d(synth[FEATURES], R_syn)
    routes = pred_fn(cache[FEATURES])
    R_val = pd.DataFrame({"bm25": cache["ndcg_bm25"], "dense": cache["ndcg_dense"],
                          "hybrid": cache["ndcg_hyb_0.6"]}, index=cache.index)
    e1_scores = R_val.to_numpy()[np.arange(len(R_val)), [R_val.columns.get_loc(r) for r in routes]]
    e1 = boot(e1_scores, R_val["hybrid"])
    e1_verdict = ("ACCEPTED" if e1["ci_95_high"] < -DELTA else "REJECTED" if e1["ci_95_low"] > -DELTA
                  else "INCONCLUSIVE")
    e1_per = {d: boot(e1_scores[cache["dataset"] == d], R_val["hybrid"][cache["dataset"] == d]) for d in DATASETS}
    auc_syn = winner_auc(synth)

    # ---------------- E2 ----------------
    df = pub.join(cache[["ndcg_bm25", "ndcg_dense"]], how="inner")
    texts = {}
    for d in DATASETS:
        with open(ROOT / "data/raw" / d / "queries.jsonl", encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                texts[f"{d}:{r['_id']}"] = r["text"]
    df["text"] = pd.Series(texts).reindex(df.index)
    df = nich.add_groups(df)
    pairs = []
    for col in ["dataset", "g_length", "g_form", "g_ident"]:
        for g, d in df.dropna(subset=[col]).groupby(col, observed=True):
            if len(d) < 300:
                continue
            for ret in ("bm25", "dense"):
                t = boot(d[f"ndcg_{ret}"], d["hybrid_ndcg_cut_10"])
                pairs.append({"grouping": col, "group": g, "n": len(d), "retriever": ret, **t})
    pairs = pd.DataFrame(pairs)
    pairs["p_holm"] = nich.holm(pairs["p_value"].to_numpy())
    pairs["niche"] = (pairs["p_holm"] < 0.05) & (pairs["delta_mean"] >= DELTA)
    e2a_niche = bool(pairs["niche"].any())
    e2b = boot(df["cost_aware_ndcg_cut_10"], df["hybrid_ndcg_cut_10"])
    e2b_ok = e2b["ci_95_high"] < DELTA
    best_f, auc_v, auc_lo, auc_hi, n_d, n_b = winner_auc(cache)
    e2c_ok = auc_v < 0.65 and auc_hi < 0.70
    if (not e2a_niche) and e2b_ok and e2c_ok:
        e2_verdict = "ACCEPTED"
    elif e2a_niche or auc_v >= 0.70:
        e2_verdict = "REJECTED"
    else:
        e2_verdict = "INCONCLUSIVE"
    tost = ci90(df["cost_aware_ndcg_cut_10"] - df["hybrid_ndcg_cut_10"])
    head = (df["oracle_ndcg_cut_10"] - df["hybrid_ndcg_cut_10"])
    gain10 = (df[["ndcg_bm25", "ndcg_dense"]].max(axis=1) - df["hybrid_ndcg_cut_10"] >= 0.10).mean()
    folds = [f for ds in s2d["fold_info"].values() for f in ds]
    fb = sum(f["cost_aware"]["epsilon"] == "always_hybrid" for f in folds)
    hyb_share = (df["route_cost_aware"] == "hybrid").mean()

    # ---------------- E3 (secondary, descriptive) ----------------
    e3 = VAL / "experiment3/summary.json"
    e3s = json.loads(e3.read_text(encoding="utf-8")) if e3.exists() else None

    md = ["# Phase 2 validation: explanatory hypotheses E1–E2", "",
          f"Pre-registered in `{PREREG}` (hash in `SHA256SUMS.txt`). Validation data: CQADupStack "
          f"{', '.join(d.split('-')[1] for d in DATASETS)} (n = {len(cache)}). nDCG@10, paired bootstrap "
          f"({N_BOOT} resamples), δ = {DELTA}.", "",
          f"## E1 — Distribution shift: **{e1_verdict}**", "",
          f"Synthetic-trained Exp 2d router (gate on synthetic data: {gate}) − hybrid (α = 0.6), pooled: "
          f"**{fmt(e1)}**, p = {e1['p_value']:.1e}. Rule: accept if CI upper < −{DELTA}.", "",
          "| Dataset | n | Synthetic-trained router − hybrid | Route share BM25 / dense / hybrid |",
          "| :--- | :---: | :---: | :---: |"]
    for d in DATASETS:
        r = routes[(cache["dataset"] == d).to_numpy()]
        md.append(f"| {d} | {len(r)} | {fmt(e1_per[d])} | {(r == 'bm25').mean():.1%} / {(r == 'dense').mean():.1%} / "
                  f"{(r == 'hybrid').mean():.1%} |")
    md += ["", f"Secondary: winner-prediction AUC' — synthetic **{auc_syn[1]:.3f}** [{auc_syn[2]:.3f}, {auc_syn[3]:.3f}] "
           f"(`{auc_syn[0]}`) vs validation **{auc_v:.3f}** [{auc_lo:.3f}, {auc_hi:.3f}] (`{best_f}`).", "",
           f"## E2 — No pre-retrieval-identifiable niche: **{e2_verdict}**", "",
           f"- (a) Niche found: **{'yes' if e2a_niche else 'no'}** ({len(pairs)} (group, retriever) pairs with n ≥ 300, "
           "Holm-corrected; table below).",
           f"- (b) Cost-aware router − hybrid: **{fmt(e2b)}** → CI upper < +{DELTA}: **{'yes' if e2b_ok else 'no'}**.",
           f"- (c) Best AUC' (dense-winner n = {n_d} vs BM25-winner n = {n_b}): **{auc_v:.3f}** [{auc_lo:.3f}, "
           f"{auc_hi:.3f}] (`{best_f}`) → < 0.65 and CI upper < 0.70: **{'yes' if e2c_ok else 'no'}**.", "",
           f"Secondary: TOST 90% CI [{tost[0]:+.4f}, {tost[1]:+.4f}] (equivalent within ±{DELTA}: "
           f"{'yes' if tost[0] > -DELTA and tost[1] < DELTA else 'no'}); oracle − hybrid headroom +{head.mean():.4f}; "
           f"queries with a single retriever ≥ 0.10 above hybrid {gain10:.1%}; fallback gate {fb}/{len(folds)} folds; "
           f"router hybrid share {hyb_share:.1%}.", "",
           "| Grouping | Group | n | Retriever | − hybrid [95% CI] | Holm p | Niche |",
           "| :--- | :--- | :---: | :--- | :---: | :---: | :---: |"]
    for r in pairs.itertuples():
        md.append(f"| {r.grouping.replace('g_', '')} | {r.group} | {r.n} | {r.retriever} | {r.delta_mean:+.4f} "
                  f"[{r.ci_95_low:+.4f}, {r.ci_95_high:+.4f}] | {r.p_holm:.3f} | {'yes' if r.niche else 'no'} |")
    md += ["", "## E3 — Skipping dense retrieval (secondary, descriptive)", ""]
    if e3s:
        pl = e3s["pooled"] if "pooled" in e3s else e3s
        md.append(f"See `results/phase2_validation/experiment3/tables.md`. Pre-recorded prediction: skip rate < 20% "
                  f"on forum queries.")
    else:
        md.append("Experiment 3 not run on the validation set yet.")
    out = VAL / "validation_results.md"
    out.write_text("\n".join(md) + "\n", encoding="utf-8")
    sys.stdout.reconfigure(encoding="utf-8")
    print("\n".join(md))


if __name__ == "__main__":
    main()
