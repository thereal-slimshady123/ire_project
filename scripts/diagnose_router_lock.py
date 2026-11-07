"""Diagnose the 94-100% always-hybrid lock of the Experiment 2d router and probe Track 2 separability.

No retrieval is re-run: everything comes from the Experiment 2 caches, data/processed/natural-niche/
(all_queries_with_overlap.csv, *_contrastive_niche.parquet) and the 288-query synthetic set.

Writes data/processed/natural-niche/{router_diagnosis.md, router_decision_log.csv, router_fold_log.csv,
feature_separability.csv, lodo_per_query.csv}.

Usage:
  python scripts/diagnose_router_lock.py --synthetic_dir <dir with scifact.parquet, nfcorpus.parquet>
"""

import argparse
import importlib.util
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier, export_text

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
warnings.filterwarnings("ignore")

from router_retrieval.data.splits import create_kfold_splits  # noqa: E402
from router_retrieval.evaluation.significance import paired_bootstrap_test  # noqa: E402
from router_retrieval.labeling.margin_labeler import MarginLabeler  # noqa: E402
from router_retrieval.utils.config import load_config  # noqa: E402

spec = importlib.util.spec_from_file_location("exp2d", ROOT / "scripts" / "run_experiment2d_cost_aware_router.py")
exp2d = importlib.util.module_from_spec(spec)
spec.loader.exec_module(exp2d)
exp2 = exp2d.exp2

NICHE = ROOT / "data/processed/natural-niche"
FEATURES = list(exp2.FEATURES)
ROUTES = ["bm25", "dense", "hybrid"]
REAL8 = ["cosqa", "wands", "cqadupstack-android", "cqadupstack-webmasters", "nfcorpus", "scifact", "fiqa",
         "arguana"]
BRIGHT = ["bright-biology", "bright-economics", "bright-psychology"]
CACHES = [ROOT / "results/niche/experiment2/cache", ROOT / "results/experiment2/cache"]
ZCOLS = ["bm25_top1_score", "dense_top1_score", "bm25_top1_top2_gap", "dense_top1_top2_gap",
         "bm25_score_entropy", "dense_score_entropy"]
FIXED_ALPHA = "0.6"
N_BOOT = 10000

cfg = {n: load_config(ROOT / "configs" / f"{n}.yaml") for n in ("experiment", "hybrid", "router", "labels")}
SEED, K = cfg["experiment"]["seed"], cfg["experiment"]["k_folds"]
ALPHAS = [float(a) for a in cfg["hybrid"]["alpha_grid"]]
EPS_GRID = [float(e) for e in cfg["labels"]["epsilon_grid"]]
ROUTER_CFG = cfg["router"]


def load_cache(ds):
    for d in CACHES:
        if (d / f"{ds}.csv").exists():
            df = pd.read_csv(d / f"{ds}.csv", index_col=0, dtype={"query_id": str})
            df.index = df.index.astype(str)
            return df
    raise FileNotFoundError(ds)


def R_of(df, alpha):
    return exp2.route_matrix(df, alpha)


# ----------------------------------------------------------------------------------------------
# Task 1: instrumented Experiment 2d (cost-aware GBM: margin-to-hybrid labels, inner-CV eps, fallback)
# ----------------------------------------------------------------------------------------------

def shap_summary(model, X):
    """LightGBM TreeSHAP contributions: (n, classes, features+1); last column is the bias."""
    raw = model.model.predict(X, pred_contrib=True)
    classes = list(model.model.classes_)
    raw = np.asarray(raw)
    if len(classes) == 2:  # binary: one log-odds vector for classes[1]; split symmetrically
        raw = np.stack([-raw / 2, raw / 2], axis=1)
    else:
        raw = raw.reshape(len(X), len(classes), len(FEATURES) + 1)
    return raw, classes


def instrument_dataset(ds, df):
    fold_rows, query_rows, shap_acc = [], [], []
    for i, (tr, te) in enumerate(create_kfold_splits(list(df.index), K, SEED)):
        alpha = max(ALPHAS, key=lambda a: df.loc[tr, f"ndcg_hyb_{a}"].mean())
        R_tr, R_te = R_of(df.loc[tr], alpha), R_of(df.loc[te], alpha)
        X_tr, X_te = df.loc[tr, FEATURES], df.loc[te, FEATURES]
        eps, util = exp2d.choose_epsilon(X_tr, R_tr, EPS_GRID, False, "default", True, ROUTER_CFG, SEED)
        best_eps = max(EPS_GRID, key=lambda e: (util[e], -e))
        inner_h = float(R_tr["hybrid"].mean())
        fallback = eps == exp2d.HYBRID
        # Fit the GBM at the best epsilon regardless, to show what the gate suppressed.
        labels = MarginLabeler(epsilon=best_eps, margin_to="default").label_wide(R_tr)["best_route"]
        m = exp2d.fit_router(X_tr, R_tr, best_eps, False, "default", ROUTER_CFG, SEED)
        if getattr(m, "constant", None) is None:
            proba = m.predict_proba(X_te)
            pred = np.array(m.predict(X_te), dtype=object)
            contrib, classes = shap_summary(m, X_te)
            shap_acc.append((contrib, classes, pred))
        else:
            proba = np.tile([1.0 if r == m.constant else 0.0 for r in ROUTES], (len(te), 1))
            pred = np.array([m.constant] * len(te), dtype=object)
        final = np.array(["hybrid"] * len(te), dtype=object) if fallback else pred
        fold_rows.append({
            "dataset": ds, "fold": i, "alpha": alpha, "best_eps": best_eps, "gate": "FALLBACK" if fallback else "router",
            "inner_router_utility": util[best_eps], "inner_hybrid_utility": inner_h,
            "inner_margin": util[best_eps] - inner_h,
            **{f"inner_util_eps_{e:g}": util[e] for e in EPS_GRID},
            "train_label_share_bm25": float((labels == "bm25").mean()),
            "train_label_share_dense": float((labels == "dense").mean()),
            "train_label_share_hybrid": float((labels == "hybrid").mean()),
            "ungated_pred_share_hybrid": float((pred == "hybrid").mean()),
        })
        for j, q in enumerate(te):
            query_rows.append({"dataset": ds, "query_id": q, "fold": i, "gate": "FALLBACK" if fallback else "router",
                               "eps": best_eps, "p_bm25": proba[j][0], "p_dense": proba[j][1], "p_hybrid": proba[j][2],
                               "ungated_route": pred[j], "final_route": final[j],
                               "ndcg_final": R_te.iloc[j][final[j]], "ndcg_ungated": R_te.iloc[j][pred[j]],
                               "ndcg_hybrid": R_te.iloc[j]["hybrid"], "ndcg_oracle": R_te.iloc[j].max()})
    return fold_rows, query_rows, shap_acc


def shap_table(shap_acc):
    """Mean |SHAP| per feature and mean signed SHAP pushing toward the hybrid class."""
    absum, hyb, n = np.zeros(len(FEATURES)), np.zeros(len(FEATURES)), 0
    for contrib, classes, _ in shap_acc:
        absum += np.abs(contrib[:, :, :-1]).sum(axis=(0, 1))
        if "hybrid" in classes:
            k = classes.index("hybrid")
            others = [c for c in range(len(classes)) if c != k]
            hyb += (contrib[:, k, :-1] - contrib[:, others, :-1].mean(axis=1)).sum(axis=0)
        n += contrib.shape[0]
    bias = np.mean([c[:, cl.index("hybrid"), -1].mean() - np.delete(c[:, :, -1], cl.index("hybrid"), axis=1).mean()
                    for c, cl, _ in shap_acc if "hybrid" in cl]) if shap_acc else float("nan")
    t = pd.DataFrame({"mean_abs_shap": absum / max(n, 1), "mean_push_to_hybrid": hyb / max(n, 1)}, index=FEATURES)
    return t.sort_values("mean_abs_shap", ascending=False), float(bias)


# ----------------------------------------------------------------------------------------------
# Task 2 helpers
# ----------------------------------------------------------------------------------------------

def auc_row(pos, neg, f):
    y = np.r_[np.ones(len(pos)), np.zeros(len(neg))]
    x = np.r_[pos[f].to_numpy(float), neg[f].to_numpy(float)]
    a = roc_auc_score(y, x)
    return a


def zscore_within(df, cols):
    out = df.copy()
    for c in cols:
        g = out.groupby("dataset")[c]
        out[f"z_{c}"] = (out[c] - g.transform("mean")) / g.transform("std").replace(0, 1)
    out["zdiff_gap_bm25_minus_dense"] = out["z_bm25_top1_top2_gap"] - out["z_dense_top1_top2_gap"]
    out["zdiff_entropy_dense_minus_bm25"] = out["z_dense_score_entropy"] - out["z_bm25_score_entropy"]
    out["zdiff_top1_bm25_minus_dense"] = out["z_bm25_top1_score"] - out["z_dense_top1_score"]
    return out


def best_route_labels(R):
    return MarginLabeler(epsilon=0.0, margin_to="default").label_wide(R)["best_route"].to_numpy()


def fit_probe(kind, X, R):
    y = best_route_labels(R)
    w = (R.max(axis=1) - R.min(axis=1) + 0.01).to_numpy()
    if len(set(y)) < 2:
        return ("const", y[0])
    if kind == "tree":
        m = DecisionTreeClassifier(max_depth=2, random_state=SEED).fit(X, y, sample_weight=w)
        return ("sk", m, None)
    sc = StandardScaler().fit(X)
    m = LogisticRegression(penalty="l1", solver="saga", C=0.05, max_iter=5000, random_state=SEED)
    m.fit(sc.transform(X), y, sample_weight=w)
    return ("sk", m, sc)


def predict_probe(model, X):
    if model[0] == "const":
        return np.array([model[1]] * len(X), dtype=object)
    _, m, sc = model
    return np.array(m.predict(sc.transform(X) if sc is not None else X), dtype=object)


def fit_2d(X, R):
    """Experiment 2d's cost-aware recipe on arbitrary training data; returns a predict function."""
    eps, _ = exp2d.choose_epsilon(X, R, EPS_GRID, False, "default", True, ROUTER_CFG, SEED)
    if eps == exp2d.HYBRID:
        return lambda Xt: np.array(["hybrid"] * len(Xt), dtype=object), "fallback"
    m = exp2d.fit_router(X, R, eps, False, "default", ROUTER_CFG, SEED)
    return (lambda Xt: exp2.predict_routes(m, Xt)), f"eps={eps:g}"


def ci(a, b):
    t = paired_bootstrap_test(np.asarray(a), np.asarray(b), N_BOOT, SEED)
    return f"{t['delta_mean']:+.4f} [{t['ci_95_low']:+.4f}, {t['ci_95_high']:+.4f}]"


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--synthetic_dir", type=Path, required=True)
    args = p.parse_args()
    md = ["# Router lock diagnosis and Track 2 separability", "",
          "No retrieval re-run; all inputs are cached features/nDCG. Router = Experiment 2d cost-aware GBM "
          "(`scripts/run_experiment2d_cost_aware_router.py`), reproduced fold-by-fold with instrumentation.", ""]

    # ---------------- Task 1 ----------------
    synth = pd.concat([pd.read_parquet(args.synthetic_dir / f"{d}.parquet") for d in ("scifact", "nfcorpus")])
    synth = synth.set_index("query_id")
    sets = {ds: load_cache(ds) for ds in REAL8 + BRIGHT}
    sets.update({f"synthetic-{d}": synth[synth["dataset"] == d] for d in ("scifact", "nfcorpus")})
    folds, queries, shap_real, shap_syn = [], [], [], []
    for ds, df in sets.items():
        f, q, s = instrument_dataset(ds, df)
        folds += f
        queries += q
        (shap_syn if ds.startswith("synthetic") else shap_real).extend(s)
    folds, queries = pd.DataFrame(folds), pd.DataFrame(queries)
    folds.to_csv(NICHE / "router_fold_log.csv", index=False)
    queries.to_csv(NICHE / "router_decision_log.csv", index=False)

    # check against the published Experiment 2d route shares
    pub = {}
    for path in [ROOT / "results/niche/experiment2d/summary.json", ROOT / "results/experiment2d/summary.json"]:
        s = json.loads(path.read_text(encoding="utf-8"))
        for ds, v in s["per_dataset"].items():
            pub.setdefault(ds, v["route_share"]["cost_aware"]["hybrid"])

    real = queries[~queries["dataset"].str.startswith("synthetic")]
    by = queries.groupby("dataset")
    t1 = pd.DataFrame({
        "n": by.size(),
        "folds_fallback": folds.groupby("dataset")["gate"].apply(lambda g: f"{(g == 'FALLBACK').sum()}/{len(g)}"),
        "share_hybrid_final": by["final_route"].apply(lambda r: (r == "hybrid").mean()),
        "published_2d_hybrid": pd.Series(pub),
        "share_hybrid_if_ungated": by["ungated_route"].apply(lambda r: (r == "hybrid").mean()),
        "mean_inner_margin": folds.groupby("dataset")["inner_margin"].mean(),
        "train_nonhybrid_labels": folds.groupby("dataset").apply(
            lambda g: 1 - g["train_label_share_hybrid"].mean()),
        "ungated_minus_hybrid": by.apply(lambda g: g["ndcg_ungated"].mean() - g["ndcg_hybrid"].mean()),
        "oracle_minus_hybrid": by.apply(lambda g: g["ndcg_oracle"].mean() - g["ndcg_hybrid"].mean()),
    })
    t1 = t1[t1["n"].notna()]
    t1["n"] = t1["n"].astype(int)
    n_fb = int((real["gate"] == "FALLBACK").sum())
    n_rt = len(real) - n_fb
    n_rt_h = int(((real["gate"] == "router") & (real["final_route"] == "hybrid")).sum())
    shap_r, bias_r = shap_table(shap_real)
    shap_s, bias_s = shap_table(shap_syn)
    md += [
        "## 1. Why the router locks onto hybrid (94–100%)", "",
        f"Real queries instrumented: {len(real)} (8 Track datasets + 3 BRIGHT); synthetic reference: "
        f"{len(queries) - len(real)}.", "",
        "**Mechanism (two stacked gates, both in `run_experiment2d_cost_aware_router.py`):**", "",
        f"1. **Fallback gate** — `choose_epsilon(...)`: `if fallback and utility[best] <= float(R[\"hybrid\"].mean()): "
        f"return HYBRID`. The GBM's inner 3-fold CV nDCG@10 must beat always-hybrid on the training folds, or the "
        f"whole fold is served by hybrid. It fired in **{(folds[~folds.dataset.str.startswith('synthetic')]['gate'] == 'FALLBACK').sum()}"
        f"/{(~folds.dataset.str.startswith('synthetic')).sum()} real folds**, covering **{n_fb}/{len(real)} "
        f"({n_fb / len(real):.1%}) real queries**. Mean inner margin (router − hybrid) on real folds: "
        f"{folds.loc[~folds.dataset.str.startswith('synthetic'), 'inner_margin'].mean():+.4f}.",
        f"2. **Label/margin gate** — `MarginLabeler(epsilon, margin_to=\"default\")`: a query is labelled bm25/dense "
        f"only if it beats hybrid by ≥ ε, else `hybrid`. Hybrid ties or wins most real queries, so in the "
        f"{n_rt} real queries whose fold passed the gate the GBM still chose hybrid for {n_rt_h} "
        f"({n_rt_h / max(n_rt, 1):.1%}).", "",
        "The gate is not the root cause, it is the symptom: when it is removed (`ungated`), the GBM routes "
        "away from hybrid more often but gains nothing (column `ungated − hybrid`: negative on most datasets, "
        "≤ +0.005 on the rest, against 2–7 points of oracle headroom), i.e. the 18 features do not "
        "predict *which* queries are in the oracle headroom. On synthetic data the same code never falls back.", "",
        "| Dataset | n | Folds → fallback | Hybrid share (this run) | Hybrid share (published 2d) | Hybrid share if ungated | "
        "Inner margin | Non-hybrid train labels | Ungated − hybrid | Oracle − hybrid |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]
    for ds, r in t1.iterrows():
        pubv = "—" if pd.isna(r["published_2d_hybrid"]) else f"{r['published_2d_hybrid']:.1%}"
        md.append(f"| {ds} | {r['n']} | {r['folds_fallback']} | {r['share_hybrid_final']:.1%} | {pubv} | "
                  f"{r['share_hybrid_if_ungated']:.1%} | {r['mean_inner_margin']:+.4f} | "
                  f"{r['train_nonhybrid_labels']:.1%} | {r['ungated_minus_hybrid']:+.4f} | {r['oracle_minus_hybrid']:+.4f} |")
    md += ["", "**GBM TreeSHAP (ungated models, held-out queries).** `push to hybrid` = SHAP for the hybrid class "
           "minus the mean of the other classes (positive → pushes the decision to hybrid). "
           f"Bias term push to hybrid: real {bias_r:+.3f}, synthetic {bias_s:+.3f} (log-odds).", "",
           "| Feature | Real mean abs SHAP | Real push to hybrid | Synthetic mean abs SHAP | Synthetic push to hybrid |",
           "| :--- | :---: | :---: | :---: | :---: |"]
    for f_ in shap_r.index:
        md.append(f"| {f_} | {shap_r.loc[f_, 'mean_abs_shap']:.3f} | {shap_r.loc[f_, 'mean_push_to_hybrid']:+.3f} | "
                  f"{shap_s.loc[f_, 'mean_abs_shap']:.3f} | {shap_s.loc[f_, 'mean_push_to_hybrid']:+.3f} |")
    md.append("")

    # ---------------- Task 2 ----------------
    allq = pd.read_csv(NICHE / "all_queries_with_overlap.csv", index_col=0, dtype={"query_id": str, "source_doc": str})
    allq.index = allq.index.astype(str)
    allq["key"] = allq["dataset"] + ":" + allq.index
    allq = zscore_within(allq, ZCOLS)
    t2 = []
    for ds in REAL8:
        d = pd.read_parquet(NICHE / f"{ds}_contrastive_niche.parquet")
        for qid, st in zip(d["query_id"], d["style"]):
            pre = {"semantic": "nat-para", "lexical": "nat-lex"}[st]
            t2.append((f"{ds}:{qid[len(pre) + len(ds) + 2:]}", st))
    t2 = pd.DataFrame(t2, columns=["key", "style"]).set_index("key")
    A = allq.set_index("key")
    sem, lex = A.loc[t2.index[t2["style"] == "semantic"]], A.loc[t2.index[t2["style"] == "lexical"]]
    neutral_pool = A[(A["ndcg_dense"] - A["ndcg_bm25"]).abs() < 0.10]
    neutral = neutral_pool.sample(n=min(628, len(neutral_pool)), random_state=SEED)
    polar = pd.concat([sem, lex])

    zfeats = [f"z_{c}" for c in ZCOLS] + ["zdiff_gap_bm25_minus_dense", "zdiff_entropy_dense_minus_bm25",
                                          "zdiff_top1_bm25_minus_dense"]
    post_hoc = ["raw_token_overlap", "idf_weighted_overlap_ratio"]
    rows = []
    for f_ in FEATURES + zfeats + post_hoc:
        a, b = auc_row(sem, lex, f_), auc_row(polar, neutral, f_)
        rows.append({"feature": f_, "group": "18 features" if f_ in FEATURES else
                     ("within-dataset z" if f_ in zfeats else "post-hoc (uses qrels)"),
                     "auc_A_sem_vs_lex": a, "sep_A": abs(a - 0.5) * 2, "auc_B_polar_vs_neutral": b,
                     **{f"sem_p{q}": sem[f_].quantile(q / 100) for q in (10, 50, 90)},
                     **{f"lex_p{q}": lex[f_].quantile(q / 100) for q in (10, 50, 90)}})
    sep = pd.DataFrame(rows).sort_values("sep_A", ascending=False)
    sep.to_csv(NICHE / "feature_separability.csv", index=False)

    # Track 1 conditions applied to Track 2 queries
    def t1_conds(d):
        p75_idf = A.groupby("dataset")["max_idf"].quantile(0.75)
        p75_r = A.groupby("dataset")["bm25_top1_top2_ratio"].quantile(0.75)
        anchor = ((d["max_idf"] >= d["dataset"].map(p75_idf)) | (d["entity_density"] >= 0.25)
                  | (d["rare_term_density"] >= 0.25))
        return pd.DataFrame({
            "sem: query_length ≥ 5": d["query_length"] >= 5,
            "sem: idf_overlap ≤ 0.35": d["idf_weighted_overlap_ratio"] <= 0.35,
            "sem: jaccard10 ≤ 0.15": d["jaccard_overlap_top10"] <= 0.15,
            "lex: rare anchor": anchor,
            "lex: idf_overlap ≥ 0.70": d["idf_weighted_overlap_ratio"] >= 0.70,
            "lex: bm25 ratio ≥ P75": d["bm25_top1_top2_ratio"] >= d["dataset"].map(p75_r),
            "lex: jaccard10 ≤ 0.20": d["jaccard_overlap_top10"] <= 0.20,
        })
    c_sem, c_lex, c_neu = t1_conds(sem).mean(), t1_conds(lex).mean(), t1_conds(neutral).mean()

    md += ["## 2. What separates Track 2's 314 dense-favored vs 314 BM25-favored queries", "",
           f"Test A: Track 2 semantic (n={len(sem)}) vs lexical (n={len(lex)}); AUC > 0.5 means higher in "
           f"semantic (dense-favored). Test B: all 628 polarized vs {len(neutral)} neutral "
           "(|nDCG_dense − nDCG_bm25| < 0.10, random sample). `sep` = |AUC − 0.5| × 2 (0 = useless, 1 = perfect).", "",
           "| Feature | Group | AUC A (sem vs lex) | sep A | AUC B (polar vs neutral) | Semantic P10/P50/P90 | "
           "Lexical P10/P50/P90 |", "| :--- | :--- | :---: | :---: | :---: | :---: | :---: |"]
    for r in sep.itertuples():
        md.append(f"| {r.feature} | {r.group} | {r.auc_A_sem_vs_lex:.3f} | {r.sep_A:.2f} | {r.auc_B_polar_vs_neutral:.3f} | "
                  f"{r.sem_p10:.3g} / {r.sem_p50:.3g} / {r.sem_p90:.3g} | {r.lex_p10:.3g} / {r.lex_p50:.3g} / {r.lex_p90:.3g} |")
    md += ["", "### Why Track 1's rules picked Easy vs Hard instead", "",
           "Pass rate of each Track 1 condition on the Track 2 buckets (a useful condition passes one bucket "
           "and fails the other):", "",
           "| Track 1 condition | Track 2 semantic | Track 2 lexical | Neutral |", "| :--- | :---: | :---: | :---: |"]
    for c in c_sem.index:
        md.append(f"| {c} | {c_sem[c]:.1%} | {c_lex[c]:.1%} | {c_neu[c]:.1%} |")
    md += ["", "`idf_weighted_overlap_ratio` measures whether the query's informative words are in the relevant "
           "document — that drives BM25 *and* the e5 dense model together (both find documents that share the "
           "query's rare words), so low overlap = both fail (hard), high overlap = both succeed (easy). It "
           "does not encode *which* retriever wins. See its AUC A above vs its AUC on hard-vs-easy below.", ""]
    hard = (A[["ndcg_bm25", "ndcg_dense"]].max(axis=1) < 0.2)
    easy = (A[["ndcg_bm25", "ndcg_dense"]].min(axis=1) > 0.7)
    md.append(f"AUC of `idf_weighted_overlap_ratio` for easy (both > 0.7, n={easy.sum()}) vs hard (both < 0.2, "
              f"n={hard.sum()}): **{auc_row(A[easy], A[hard], 'idf_weighted_overlap_ratio'):.3f}**; for Track 2 "
              f"semantic vs lexical: **{auc_row(sem, lex, 'idf_weighted_overlap_ratio'):.3f}**.")
    md.append("")

    # ---------------- Task 2.2: LODO probes ----------------
    A_all = A.copy()
    zX = {f: f"zf_{f}" for f in FEATURES}
    for f_ in FEATURES:
        g = A_all.groupby("dataset")[f_]
        A_all[zX[f_]] = (A_all[f_] - g.transform("mean")) / g.transform("std").replace(0, 1)
    R_all = pd.DataFrame({"bm25": A_all["ndcg_bm25"], "dense": A_all["ndcg_dense"],
                          "hybrid": A_all[f"ndcg_hyb_{FIXED_ALPHA}"]}, index=A_all.index)
    S = synth.copy()
    for f_ in FEATURES:
        g = S.groupby("dataset")[f_]
        S[zX[f_]] = (S[f_] - g.transform("mean")) / g.transform("std").replace(0, 1)
    R_syn = pd.DataFrame({"bm25": S["ndcg_bm25"], "dense": S["ndcg_dense"], "hybrid": S[f"ndcg_hyb_{FIXED_ALPHA}"]},
                         index=S.index)

    systems = {}
    # (a) synthetic-trained
    pred_fn, tag_a = fit_2d(S[FEATURES], R_syn)
    systems["(a) Synthetic-trained 2d GBM"] = pd.Series(pred_fn(A_all[FEATURES]), index=A_all.index)
    tree_syn = fit_probe("tree", S[FEATURES], R_syn)
    systems["(a) Synthetic-trained depth-2 tree"] = pd.Series(predict_probe(tree_syn, A_all[FEATURES]), index=A_all.index)
    # (b) real-data LODO
    lodo = {k: pd.Series(index=A_all.index, dtype=object) for k in
            ["(b) LODO 2d GBM recipe", "(b) LODO depth-2 tree", "(b) LODO L1 logistic",
             "(b) LODO depth-2 tree (z-feat)", "(b) LODO L1 logistic (z-feat)"]}
    tree_text, lodo_tags = {}, {}
    for ds in REAL8:
        tr, te = A_all.index[A_all["dataset"] != ds], A_all.index[A_all["dataset"] == ds]
        fn, lodo_tags[ds] = fit_2d(A_all.loc[tr, FEATURES], R_all.loc[tr])
        lodo["(b) LODO 2d GBM recipe"][te] = fn(A_all.loc[te, FEATURES])
        for kind, name in [("tree", "depth-2 tree"), ("l1", "L1 logistic")]:
            for cols, suf in [(FEATURES, ""), ([zX[f] for f in FEATURES], " (z-feat)")]:
                m = fit_probe(kind, A_all.loc[tr, cols], R_all.loc[tr])
                lodo[f"(b) LODO {name}{suf}"][te] = predict_probe(m, A_all.loc[te, cols])
                if kind == "tree" and suf == "" and ds == "arguana" and m[0] == "sk":
                    tree_text["example"] = export_text(m[1], feature_names=list(cols))
    systems.update(lodo)

    def score(routes):
        return np.array([R_all.at[q, r] for q, r in routes.items()])

    per_q = pd.DataFrame({"dataset": A_all["dataset"], "hybrid_0.6": R_all["hybrid"],
                          "oracle": R_all.max(axis=1)})
    for name, routes in systems.items():
        per_q[name] = score(routes)
        per_q[f"route::{name}"] = routes
    per_q["track2"] = per_q.index.isin(t2.index)
    per_q.to_csv(NICHE / "lodo_per_query.csv")

    names = list(systems)
    md += ["## 3. Leave-one-dataset-out routing vs fixed hybrid (α = 0.6)", "",
           "Routes: BM25 / dense / hybrid α=0.6. Labels: best route (ties → hybrid), sample weight = nDCG spread "
           "(as in Experiment 2). (a) = trained on the 288 synthetic queries only "
           f"(2d recipe: {tag_a}); (b) = trained on the 7 other real datasets, evaluated on the 8th. "
           "`z-feat` = the 18 features z-scored within each dataset (unsupervised, uses the held-out dataset's "
           "own feature distribution, no labels). Cells: mean nDCG@10; pooled rows add Δ vs hybrid [95% CI].", ""]
    md += ["| Dataset | n | Hybrid 0.6 | " + " | ".join(names) + " | Oracle |",
           "| :--- | :---: | :---: | " + " | ".join(":---:" for _ in names) + " | :---: |"]
    for ds in REAL8:
        g = per_q[per_q["dataset"] == ds]
        md.append(f"| {ds} | {len(g)} | {g['hybrid_0.6'].mean():.4f} | "
                  + " | ".join(f"{g[n].mean():.4f}" for n in names) + f" | {g['oracle'].mean():.4f} |")
    for label, g in [("**pooled (4,856)**", per_q), ("**Track 2 subset (628)**", per_q[per_q["track2"]])]:
        md.append(f"| {label} | {len(g)} | {g['hybrid_0.6'].mean():.4f} | "
                  + " | ".join(f"{g[n].mean():.4f}" for n in names) + f" | {g['oracle'].mean():.4f} |")
    md += ["", "| Subset | " + " | ".join(names) + " |", "| :--- | " + " | ".join(":---:" for _ in names) + " |"]
    for label, g in [("Δ vs hybrid, pooled", per_q), ("Δ vs hybrid, Track 2", per_q[per_q["track2"]])]:
        md.append(f"| {label} | " + " | ".join(ci(g[n], g["hybrid_0.6"]) for n in names) + " |")
    md += ["", "Route shares (pooled, BM25 / dense / hybrid):", ""]
    for n in names:
        r = per_q[f"route::{n}"]
        t2r = r[per_q["track2"]]
        md.append(f"- {n}: all {(r == 'bm25').mean():.1%} / {(r == 'dense').mean():.1%} / {(r == 'hybrid').mean():.1%}; "
                  f"Track 2 {(t2r == 'bm25').mean():.1%} / {(t2r == 'dense').mean():.1%} / {(t2r == 'hybrid').mean():.1%}")
    md += ["", f"LODO 2d-recipe gate per held-out dataset: " + ", ".join(f"{d}: {t}" for d, t in lodo_tags.items()), ""]
    if tree_text:
        md += ["Example LODO depth-2 tree (trained without arguana):", "", "```", tree_text["example"].rstrip(), "```", ""]

    # ---------------- Exp 2d on Track 2 (stress test) and published 2d restricted to Track 2 ----------------
    pub_q = pd.concat([pd.read_csv(ROOT / "results/niche/experiment2d/per_query_heldout.csv", index_col=0),
                       pd.read_csv(ROOT / "results/experiment2d/per_query_heldout.csv", index_col=0)])
    pub_q.index = pub_q.index.astype(str)
    pub_t2 = pub_q.loc[pub_q.index.intersection(t2.index)]
    T = A_all.loc[t2.index]
    out = []
    for tr, te in create_kfold_splits(list(T.index), K, SEED):
        alpha = max(ALPHAS, key=lambda a: T.loc[tr, f"ndcg_hyb_{a}"].mean())
        Rtr, Rte = R_of(T.loc[tr], alpha), R_of(T.loc[te], alpha)
        fn, tag = fit_2d(T.loc[tr, FEATURES], Rtr)
        routes = fn(T.loc[te, FEATURES])
        out.append(pd.DataFrame({"hybrid": Rte["hybrid"], "router": exp2.routed_scores(Rte, routes),
                                 "oracle": Rte.max(axis=1), "route": routes, "gate": tag}, index=te))
    out = pd.concat(out)
    md += ["## 4. Experiment 2d on Track 2", "",
           f"**(i) Published Experiment 2d run, restricted to Track 2 queries** (router trained on each dataset's full "
           f"query set, n = {len(pub_t2)}): cost-aware − hybrid = {ci(pub_t2['cost_aware_ndcg_cut_10'], pub_t2['hybrid_ndcg_cut_10'])}; "
           f"hybrid share {(pub_t2['route_cost_aware'] == 'hybrid').mean():.1%}; oracle − hybrid = "
           f"{ci(pub_t2['oracle_ndcg_cut_10'], pub_t2['hybrid_ndcg_cut_10'])}.", "",
           f"**(ii) Stress test: 2d recipe trained by 5-fold CV on the 628 Track 2 queries themselves** (optimistic: "
           f"the set was selected on nDCG, so every query has a clear winner): router − hybrid = "
           f"{ci(out['router'], out['hybrid'])}; oracle − hybrid = {ci(out['oracle'], out['hybrid'])}; route share "
           f"BM25 / dense / hybrid = {(out.route == 'bm25').mean():.1%} / {(out.route == 'dense').mean():.1%} / "
           f"{(out.route == 'hybrid').mean():.1%}; gate outcome across the 5 folds: {', '.join(dict.fromkeys(out['gate']))}.",
           ""]
    (NICHE / "router_diagnosis.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md))


if __name__ == "__main__":
    main()
