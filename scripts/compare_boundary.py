"""E8: is the per-query winner more stable when relevant documents are complements (multi-hop) than alternatives?

Uses analyze_label_reliability.py unchanged to compute, per query, the qrels split-half reference margin and label
(5 random splits, seed 42), then pools each group of datasets:
  reference AUC' = AUC' of the half-A dense - BM25 margin for the half-B binary winner (E4's definition),
  bootstrap 2,000 resamples of whole queries within each group (seed 42), and the difference between groups.
Secondary: router - H1 at a 20% skip rate per group (run_e4_matched_rate.py per-query file); E2(c) best AUC' with
within-dataset percentile features per group (run_e2_replication.pct_within + run_phase2_validation.winner_auc).

Usage:
  python scripts/compare_boundary.py --complementary musique 2wikimultihopqa --alternative dbpedia \
      --exp2_dir <dir> --matched_dir <dir> --output_dir <dir>
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


lrel = _load("lrel", "analyze_label_reliability.py")
e2r = _load("e2r", "run_e2_replication.py")
N_BOOT, SEED = 2000, 42


def half_rows(datasets, exp2_dir):
    ys, ss, gs = [], [], []
    for ds in datasets:
        cache = pd.read_csv(exp2_dir / "cache" / f"{ds}.csv", index_col=0, dtype={"query_id": str})
        cache.index = cache.index.astype(str)
        hyb = [c for c in cache if c.startswith("ndcg_hyb_")]
        a = round(float(max(hyb, key=lambda c: cache[c].mean()).split("_")[-1]), 1)
        df = lrel.dataset_scores(ds, list(cache.index), a)
        for name, _, _, bb, _, margin, mask in lrel.variants(df):
            if not name.startswith("qrels-half"):
                continue
            lab = (bb != "none") & mask
            ys.append((bb[lab] == "dense").astype(int))
            ss.append(margin[lab])
            gs.append(np.array([f"{ds}:{q}" for q in df.index[lab]]))
    return np.concatenate(ys), np.concatenate(ss), np.concatenate(gs)


def boot_auc(y, s, g, rng):
    codes, uniq = pd.factorize(g)
    out = []
    for _ in range(N_BOOT):
        w = np.bincount(rng.integers(0, len(uniq), len(uniq)), minlength=len(uniq))[codes]
        k = w > 0
        if len(set(y[k])) < 2:
            out.append(np.nan)
            continue
        a = roc_auc_score(y[k], s[k], sample_weight=w[k])
        out.append(max(a, 1 - a))
    return np.array(out)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--complementary", nargs="+", required=True)
    ap.add_argument("--alternative", nargs="+", required=True)
    ap.add_argument("--exp2_dir", type=Path, required=True)
    ap.add_argument("--matched_dir", type=Path, required=True)
    ap.add_argument("--output_dir", type=Path, required=True)
    args = ap.parse_args()
    rng = np.random.default_rng(SEED)
    res, boots = {}, {}
    for grp, dss in [("complementary", args.complementary), ("alternative", args.alternative)]:
        y, s, g = half_rows(dss, args.exp2_dir)
        boots[grp] = boot_auc(y, s, g, rng)
        a = roc_auc_score(y, s)
        res[grp] = {"datasets": dss, "queries": int(len(set(g))), "rows": int(len(y)), "ref_auc": float(max(a, 1 - a)),
                    "ci": [float(v) for v in np.nanpercentile(boots[grp], [2.5, 97.5])]}
    d = boots["complementary"] - boots["alternative"]
    res["difference"] = {"value": res["complementary"]["ref_auc"] - res["alternative"]["ref_auc"],
                         "ci": [float(v) for v in np.nanpercentile(d, [2.5, 97.5])]}
    c, al, df_ = res["complementary"], res["alternative"], res["difference"]
    a_ok = c["ref_auc"] >= 0.70 and c["ci"][0] > 0.65
    b_ok = al["ref_auc"] < 0.65
    c_ok = df_["ci"][0] > 0
    res["criteria"] = {"a_complementary_high": a_ok, "b_alternative_low": b_ok, "c_difference_positive": c_ok}
    res["verdict"] = ("ACCEPTED" if a_ok and b_ok and c_ok else
                      "REJECTED" if c["ref_auc"] < 0.60 or df_["ci"][1] <= 0 else "INCONCLUSIVE")

    # Secondary
    pq_ = pd.read_csv(args.matched_dir / "per_query_heldout.csv", index_col=0)
    cache_all = pd.concat([pd.read_csv(args.exp2_dir / "cache" / f"{ds}.csv", index_col=0, dtype={"query_id": str})
                           for ds in args.complementary + args.alternative])
    for grp, dss in [("complementary", args.complementary), ("alternative", args.alternative)]:
        sub = pq_[pq_.dataset.isin(dss)]
        res[grp]["router_minus_H1@0.20"] = paired_bootstrap_test(sub["router@0.20"].values, sub["H1_bm25_top_ratio@0.20"].values, 10000, SEED)
        res[grp]["router_minus_hybrid@0.20"] = paired_bootstrap_test(sub["router@0.20"].values, sub["hybrid"].values, 10000, SEED)
        w = e2r.p2v.winner_auc(e2r.pct_within(cache_all[cache_all.dataset.isin(dss)]))
        res[grp]["e2c_auc_pct"] = {"feature": w[0], "auc": float(w[1]), "ci": [float(w[2]), float(w[3])]}

    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps(res, indent=2, default=lambda o: bool(o) if isinstance(o, np.bool_) else float(o)),
                                      encoding="utf-8")
    ci = lambda t: f"{t['delta_mean']:+.4f} [{t['ci_95_low']:+.4f}, {t['ci_95_high']:+.4f}]"  # noqa: E731
    yn = lambda b: "PASS" if b else "FAIL"  # noqa: E731
    L = ["# E8: boundary condition — complementary vs alternative relevant documents", "",
         f"**Verdict: {res['verdict']}** — (a) complementary AUC′ ≥ 0.70 and CI low > 0.65: {yn(a_ok)}; "
         f"(b) alternative AUC′ < 0.65: {yn(b_ok)}; (c) difference CI low > 0: {yn(c_ok)}.", "",
         "| Group | Datasets | Queries | Split-half reference AUC′ [95% CI] | E2(c) best AUC′ (percentile) | Router − H1 @20% | Router − hybrid @20% |",
         "| :--- | :--- | :---: | :---: | :---: | :---: | :---: |"]
    for grp in ("complementary", "alternative"):
        r = res[grp]
        L.append(f"| {grp} | {', '.join(r['datasets'])} | {r['queries']} | {r['ref_auc']:.3f} [{r['ci'][0]:.3f}, {r['ci'][1]:.3f}] "
                 f"| {r['e2c_auc_pct']['auc']:.3f} [{r['e2c_auc_pct']['ci'][0]:.3f}, {r['e2c_auc_pct']['ci'][1]:.3f}] "
                 f"| {ci(r['router_minus_H1@0.20'])} | {ci(r['router_minus_hybrid@0.20'])} |")
    L += ["", f"Difference (complementary − alternative): {df_['value']:+.3f} [{df_['ci'][0]:+.3f}, {df_['ci'][1]:+.3f}]."]
    (out / "tables.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
