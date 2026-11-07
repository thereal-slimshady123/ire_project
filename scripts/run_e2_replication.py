"""E2 (no pre-retrieval-identifiable niche) on any set of datasets / dense model, with the definitions of
scripts/run_phase2_validation.py (groups and Holm test from analyze_niche_existence.py, winner AUC', delta = 0.005).

Inputs are the outputs of run_experiment2_heldout_routing.py (--exp2_dir) and run_experiment2d_cost_aware_router.py
(--exp2d_dir) for the same datasets and dense model.

Usage:
  python scripts/run_e2_replication.py --datasets <ds ...> --exp2_dir <dir> --exp2d_dir <dir> --label <name> --output_dir <dir>
"""

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("p2v", ROOT / "scripts" / "run_phase2_validation.py")
p2v = importlib.util.module_from_spec(spec)
spec.loader.exec_module(p2v)
nich, boot, fmt, DELTA = p2v.nich, p2v.boot, p2v.fmt, p2v.DELTA


def pct_within(cache: pd.DataFrame) -> pd.DataFrame:
    """Features as within-dataset percentile ranks: removes between-dataset scale differences from E2(c)."""
    z = cache.copy()
    for f in p2v.FEATURES:
        z[f] = cache.groupby("dataset")[f].rank(pct=True, method="average")
    return z


def e2(datasets, exp2_dir: Path, exp2d_dir: Path, auc_features: str = "pct") -> dict:
    cache = pd.concat([pd.read_csv(exp2_dir / "cache" / f"{d}.csv", index_col=0, dtype={"query_id": str}) for d in datasets])
    cache.index = cache["dataset"] + ":" + cache.index.astype(str)
    pub = pd.read_csv(exp2d_dir / "per_query_heldout.csv", index_col=0)
    pub.index = pub.index.astype(str)
    pub = pub[pub.index.str.split(":").str[0].isin(datasets)]
    s2d = json.loads((exp2d_dir / "summary.json").read_text(encoding="utf-8"))

    df = pub.join(cache[["ndcg_bm25", "ndcg_dense"]], how="inner")
    texts = {}
    for d in datasets:
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
                pairs.append({"grouping": col, "group": str(g), "n": len(d), "retriever": ret, **t})
    pairs = pd.DataFrame(pairs)
    pairs["p_holm"] = nich.holm(pairs["p_value"].to_numpy())
    pairs["niche"] = (pairs["p_holm"] < 0.05) & (pairs["delta_mean"] >= DELTA)
    niche = bool(pairs["niche"].any())
    b = boot(df["cost_aware_ndcg_cut_10"], df["hybrid_ndcg_cut_10"])
    raw = p2v.winner_auc(cache)
    pct = p2v.winner_auc(pct_within(cache))
    best_f, auc, lo, hi, n_d, n_b = pct if auc_features == "pct" else raw
    verdict = ("ACCEPTED" if (not niche) and b["ci_95_high"] < DELTA and auc < 0.65 and hi < 0.70
               else "REJECTED" if niche or auc >= 0.70 else "INCONCLUSIVE")
    folds = [f for ds in s2d["fold_info"].values() for f in ds]
    return {
        "n": int(len(df)), "verdict": verdict, "niche_found": niche, "pairs": pairs.to_dict(orient="records"),
        "router_minus_hybrid": b, "router_ci_upper_below_delta": bool(b["ci_95_high"] < DELTA),
        "auc_features": auc_features,
        "best_auc": {"feature": best_f, "auc": auc, "ci": [lo, hi], "n_dense_winner": n_d, "n_bm25_winner": n_b},
        "best_auc_raw": {"feature": raw[0], "auc": raw[1], "ci": [raw[2], raw[3]]},
        "best_auc_pct": {"feature": pct[0], "auc": pct[1], "ci": [pct[2], pct[3]]},
        "tost_90": [float(v) for v in p2v.ci90(df["cost_aware_ndcg_cut_10"] - df["hybrid_ndcg_cut_10"])],
        "oracle_headroom": float((df["oracle_ndcg_cut_10"] - df["hybrid_ndcg_cut_10"]).mean()),
        "single_retriever_gain_ge_0.10": float((df[["ndcg_bm25", "ndcg_dense"]].max(axis=1) - df["hybrid_ndcg_cut_10"] >= 0.10).mean()),
        "fallback_folds": [sum(f["cost_aware"]["epsilon"] == "always_hybrid" for f in folds), len(folds)],
        "router_hybrid_share": float((df["route_cost_aware"] == "hybrid").mean()),
        "per_dataset_bm25_dense_minus_hybrid": {d: {r: boot(g[f"ndcg_{r}"], g["hybrid_ndcg_cut_10"]) for r in ("bm25", "dense")}
                                                for d, g in df.groupby("dataset")},
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--datasets", nargs="+", required=True)
    ap.add_argument("--exp2_dir", type=Path, required=True)
    ap.add_argument("--exp2d_dir", type=Path, required=True)
    ap.add_argument("--label", required=True, help="e.g. 'fresh forums, e5-small-v2'")
    ap.add_argument("--output_dir", type=Path, required=True)
    ap.add_argument("--auc_features", choices=["pct", "raw"], default="pct",
                    help="features for the E2(c) verdict: within-dataset percentiles (default) or raw values")
    args = ap.parse_args()
    r = e2(args.datasets, args.exp2_dir, args.exp2d_dir, args.auc_features)
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps({"label": args.label, "datasets": args.datasets, **r}, indent=2,
                                                 default=lambda o: bool(o) if isinstance(o, np.bool_) else float(o)),
                                      encoding="utf-8")
    a = r["best_auc"]
    L = [f"# E2 replication: {args.label}", "", f"Datasets: {', '.join(args.datasets)} (n = {r['n']}). Definitions as in "
         "`results/preregistration/phase2_explanatory_validation_20261107.md`.", "",
         f"**E2: {r['verdict']}**", "",
         f"- (a) Niche found: **{'yes' if r['niche_found'] else 'no'}** ({len(r['pairs'])} (group, retriever) pairs with n ≥ 300, Holm-corrected).",
         f"- (b) Cost-aware router − hybrid: **{fmt(r['router_minus_hybrid'])}** → CI upper < +0.005: **{'yes' if r['router_ci_upper_below_delta'] else 'no'}**.",
         f"- (c) Best AUC′ (dense-winner n = {a['n_dense_winner']} vs BM25-winner n = {a['n_bm25_winner']}): **{a['auc']:.3f}** "
         f"[{a['ci'][0]:.3f}, {a['ci'][1]:.3f}] (`{a['feature']}`; {'within-dataset percentile' if r['auc_features'] == 'pct' else 'raw'} features). "
         f"Other definition: raw {r['best_auc_raw']['auc']:.3f} [{r['best_auc_raw']['ci'][0]:.3f}, {r['best_auc_raw']['ci'][1]:.3f}] "
         f"(`{r['best_auc_raw']['feature']}`), percentile {r['best_auc_pct']['auc']:.3f} [{r['best_auc_pct']['ci'][0]:.3f}, "
         f"{r['best_auc_pct']['ci'][1]:.3f}] (`{r['best_auc_pct']['feature']}`).", "",
         f"Secondary: TOST 90% CI [{r['tost_90'][0]:+.4f}, {r['tost_90'][1]:+.4f}]; oracle − hybrid headroom {r['oracle_headroom']:+.4f}; "
         f"queries with a single retriever ≥ 0.10 above hybrid {r['single_retriever_gain_ge_0.10']:.1%}; fallback gate "
         f"{r['fallback_folds'][0]}/{r['fallback_folds'][1]} folds; router hybrid share {r['router_hybrid_share']:.1%}.", "",
         "| Grouping | Group | n | Retriever | − hybrid [95% CI] | Holm p | Niche |", "| :--- | :--- | :---: | :--- | :---: | :---: | :---: |"]
    for p in r["pairs"]:
        L.append(f"| {p['grouping']} | {p['group']} | {p['n']} | {p['retriever']} | {fmt(p)} | {p['p_holm']:.3g} | {'**yes**' if p['niche'] else 'no'} |")
    (out / "tables.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L[:12]))


if __name__ == "__main__":
    main()
