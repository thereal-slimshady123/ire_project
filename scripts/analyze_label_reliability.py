"""How reliable is the per-query "winner" label that routers are trained to predict?

For each query the label is recomputed under small, defensible changes to how it is measured, and compared with the
base label (nDCG@10, e5-small-v2, official qrels):

  alpha-/+     hybrid at alpha* -/+ 0.1 (3-way label) or 0.5 / 0.7 (binary label, base alpha = 0.6)
  ndcg@5/@20   a different nDCG cut-off
  mrr@10       a different metric
  qrels-half   relevance judgments split in two at random (queries with >= 2 relevant docs; 5 splits, seed 42):
               label from half A vs label from half B -- a test-retest reliability of the label itself
  e5-base      the dense retriever swapped for intfloat/e5-base-v2 (where its run exists)

Labels:
  3-way   argmax of {bm25, dense, hybrid(alpha*)}; ties go to hybrid, then dense (the router's target).
  binary  E2(c): dense-winner if dense > hybrid(0.6) and dense > bm25; BM25-winner symmetric; else none.

Reported per perturbation: Cohen's kappa (3-way), binary label retention / flip / lost shares, and the reference AUC'
of the perturbed dense-minus-BM25 margin for the base binary label (how well a near-identical *measurement*
predicts the label; compare with the best query-feature AUC').

Usage:
  python scripts/analyze_label_reliability.py --datasets <ds ...> --exp2_dir <dir ...> --output_dir <dir>
"""

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from router_retrieval.evaluation.ranking_metrics import ndcg_at_k, reciprocal_rank  # noqa: E402

spec = importlib.util.spec_from_file_location("exp1", ROOT / "scripts" / "run_experiment1_hybrid_degradation.py")
exp1 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(exp1)

SEED, N_SPLITS, N_BOOT = 42, 5, 2000
SMALL, BASE = "intfloat__e5-small-v2", "intfloat__e5-base-v2"
METRICS = {
    "ndcg@10": lambda d, r: ndcg_at_k(d, r, 10),
    "ndcg@5": lambda d, r: ndcg_at_k(d, r, 5),
    "ndcg@20": lambda d, r: ndcg_at_k(d, r, 20),
    "mrr@10": lambda d, r: reciprocal_rank(d, r, 10),
}


def qrels_path(ds: str) -> Path:
    return ROOT / "data" / ("processed/quora-sub" if ds == "quora" else f"raw/{ds}") / "qrels" / "test.tsv"


def read_qrels(path: Path) -> dict:
    df = pd.read_csv(path, sep="\t", dtype=str)
    out = {}
    for q, d, s in df.itertuples(index=False):
        out.setdefault(q, {})[d] = int(s)
    return out


def ids(run, q):
    return [c.doc_id for c in sorted(run.get(q, []), key=lambda c: c.rank)]


def label3(b, d, h):
    """Ties: hybrid > dense > bm25."""
    if h >= d and h >= b:
        return "hybrid"
    return "dense" if d >= b else "bm25"


def label_bin(b, d, h):
    if d > h and d > b:
        return "dense"
    if b > h and b > d:
        return "bm25"
    return "none"


def auc_prime(y, s):
    a = roc_auc_score(y, s)
    return max(a, 1 - a)


def auc_ci(y, s, groups, seed=SEED):
    """Bootstrap CI of AUC' resampling whole queries (rows of one query move together)."""
    rng = np.random.default_rng(seed)
    codes, uniq = pd.factorize(groups)
    bs = []
    for _ in range(N_BOOT):
        w = np.bincount(rng.integers(0, len(uniq), len(uniq)), minlength=len(uniq))[codes]
        keep = w > 0
        if len(set(y[keep])) < 2:
            continue
        a = roc_auc_score(y[keep], s[keep], sample_weight=w[keep])
        bs.append(max(a, 1 - a))
    return [float(v) for v in np.percentile(bs, [2.5, 97.5])]


def dataset_scores(ds: str, qids: list, alpha_star: float) -> pd.DataFrame:
    """Per-query scores of bm25 / dense / hybrid under every measurement variant (one row per query)."""
    run_dir = ROOT / "results" / "runs" / ds
    qrels = read_qrels(qrels_path(ds))
    bm25 = exp1.read_trec_run(run_dir / "bm25_k1-1.2_b-0.75.trec")
    models = {"small": SMALL}
    if (run_dir / f"dense_{BASE}.trec").exists():
        models["base"] = BASE
    alphas = sorted({round(alpha_star + x, 1) for x in (-0.1, 0.0, 0.1)} | {0.5, 0.6, 0.7})
    alphas = [a for a in alphas if 0.0 <= a <= 1.0]
    rows = {q: {} for q in qids}
    rng = np.random.default_rng(SEED)
    splits = {}
    for q in qids:  # fixed random halves of the relevant docs (same halves for every system)
        rel = sorted(d for d, r in qrels.get(q, {}).items() if r > 0)
        if len(rel) >= 2:
            splits[q] = []
            for _ in range(N_SPLITS):
                perm = rng.permutation(rel)
                h = len(perm) // 2
                A, B = set(perm[:h]), set(perm[h:])
                splits[q].append(({d: r for d, r in qrels[q].items() if d not in B},
                                  {d: r for d, r in qrels[q].items() if d not in A}))
    for m, slug in models.items():
        dense = exp1.read_trec_run(run_dir / f"dense_{slug}.trec")
        for q in qids:
            bm25.setdefault(q, [])
            dense.setdefault(q, [])
        hyb = exp1.build_hybrid_runs(bm25, dense, alphas, 100, "min_max")
        systems = {"bm25": bm25, "dense": dense, **{f"hyb{a}": hyb[a] for a in alphas}}
        for q in qids:
            r = rows[q]
            for sname, run in systems.items():
                if m == "base" and sname == "bm25":
                    continue
                docs = ids(run, q)
                tag = sname if m == "small" else f"{sname}|base"
                for mname, f in METRICS.items():
                    if m == "base" and mname != "ndcg@10":
                        continue
                    r[f"{tag}|{mname}"] = f(docs, qrels.get(q, {}))
                if m == "small" and q in splits:
                    for i, (qa, qb) in enumerate(splits[q]):
                        r[f"{tag}|halfA{i}"] = ndcg_at_k(docs, qa, 10)
                        r[f"{tag}|halfB{i}"] = ndcg_at_k(docs, qb, 10)
    df = pd.DataFrame.from_dict(rows, orient="index")
    df["n_rel"] = [sum(1 for r in qrels.get(q, {}).values() if r > 0) for q in qids]
    df.attrs.update(alpha_star=alpha_star, alphas=alphas, has_base="base" in models)
    return df


def variants(df: pd.DataFrame):
    """Yield (name, base labels, perturbed labels, perturbed binary margin or None, mask) per perturbation."""
    a = df.attrs["alpha_star"]
    lo, hi = round(a - 0.1, 1), round(a + 0.1, 1)
    col = lambda s, m="ndcg@10": df[f"{s}|{m}"].to_numpy()  # noqa: E731
    B, D = col("bm25"), col("dense")
    base3 = np.array([label3(*t) for t in zip(B, D, col(f"hyb{a}"))])
    baseb = np.array([label_bin(*t) for t in zip(B, D, col("hyb0.6"))])
    every = np.ones(len(df), bool)
    for name, alt_a, alt_b in [("alpha-0.1", lo, 0.5), ("alpha+0.1", hi, 0.7)]:
        if f"hyb{alt_a}|ndcg@10" in df:
            p3 = np.array([label3(*t) for t in zip(B, D, col(f"hyb{alt_a}"))])
            pb = np.array([label_bin(*t) for t in zip(B, D, col(f"hyb{alt_b}"))])
            yield name, base3, p3, baseb, pb, None, every
    for m in ["ndcg@5", "ndcg@20", "mrr@10"]:
        Bm, Dm = col("bm25", m), col("dense", m)
        p3 = np.array([label3(*t) for t in zip(Bm, Dm, col(f"hyb{a}", m))])
        pb = np.array([label_bin(*t) for t in zip(Bm, Dm, col("hyb0.6", m))])
        yield m, base3, p3, baseb, pb, Dm - Bm, every
    if df.attrs["has_base"]:
        Db = df["dense|base|ndcg@10"].to_numpy()
        p3 = np.array([label3(*t) for t in zip(B, Db, df[f"hyb{a}|base|ndcg@10"].to_numpy())])
        pb = np.array([label_bin(*t) for t in zip(B, Db, df["hyb0.6|base|ndcg@10"].to_numpy())])
        yield "e5-base", base3, p3, baseb, pb, Db - B, every
    if "bm25|halfA0" in df:
        m = df["bm25|halfA0"].notna().to_numpy()
        for i in range(N_SPLITS):
            g = lambda s, h: df[f"{s}|half{h}{i}"].to_numpy()  # noqa: E731
            a3 = np.array([label3(*t) for t in zip(g("bm25", "A"), g("dense", "A"), g(f"hyb{a}", "A"))])
            b3 = np.array([label3(*t) for t in zip(g("bm25", "B"), g("dense", "B"), g(f"hyb{a}", "B"))])
            ab = np.array([label_bin(*t) for t in zip(g("bm25", "A"), g("dense", "A"), g("hyb0.6", "A"))])
            bb = np.array([label_bin(*t) for t in zip(g("bm25", "B"), g("dense", "B"), g("hyb0.6", "B"))])
            yield f"qrels-half#{i}", a3, b3, ab, bb, g("dense", "B") - g("bm25", "B"), m


def summarise(df: pd.DataFrame) -> pd.DataFrame:
    out = []
    for name, b3, p3, bb, pb, margin, mask in variants(df):
        b3, p3, bb, pb = b3[mask], p3[mask], bb[mask], pb[mask]
        lab = bb != "none"
        row = {"perturbation": name.split("#")[0], "n": int(mask.sum()), "n_binary": int(lab.sum()),
               "kappa_3way": cohen_kappa_score(b3, p3) if len(set(b3) | set(p3)) > 1 else np.nan,
               "agree_3way": float((b3 == p3).mean()),
               "bin_retained": float((pb[lab] == bb[lab]).mean()) if lab.any() else np.nan,
               "bin_flipped": float(((pb[lab] != bb[lab]) & (pb[lab] != "none")).mean()) if lab.any() else np.nan,
               "bin_lost": float((pb[lab] == "none").mean()) if lab.any() else np.nan,
               "ref_auc": np.nan}
        if margin is not None and lab.sum() >= 20 and len(set(bb[lab])) == 2:
            row["ref_auc"] = auc_prime((bb[lab] == "dense").astype(int), margin[mask][lab])
        out.append(row)
    return pd.DataFrame(out).groupby("perturbation", sort=False).mean(numeric_only=True).reset_index()


def pooled_ref_auc(dfs: dict, which: str):
    """Pooled reference AUC' (+ bootstrap CI) of a perturbed margin for the base binary label."""
    ys, ss, gs = [], [], []
    for ds, df in dfs.items():
        for name, _, _, bb, _, margin, mask in variants(df):
            if name.split("#")[0] != which or margin is None:
                continue
            lab = (bb != "none") & mask
            ys.append((bb[lab] == "dense").astype(int))
            ss.append(margin[lab])
            gs.append(np.array([f"{ds}:{q}" for q in df.index[lab]]))
    if not ys:
        return None
    y, s, g = np.concatenate(ys), np.concatenate(ss), np.concatenate(gs)
    return {"auc": float(auc_prime(y, s)), "ci": auc_ci(y, s, g), "n_rows": int(len(y)), "n": int(len(set(g)))}


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--datasets", nargs="+", required=True)
    ap.add_argument("--exp2_dir", type=Path, nargs="+", required=True, help="dirs holding e5-small exp2 caches")
    ap.add_argument("--output_dir", type=Path, required=True)
    args = ap.parse_args()

    dfs, per_ds = {}, []
    for ds in args.datasets:
        src = next(d for d in args.exp2_dir if (d / "cache" / f"{ds}.csv").exists())
        cache = pd.read_csv(src / "cache" / f"{ds}.csv", index_col=0, dtype={"query_id": str})
        cache.index = cache.index.astype(str)
        hyb_cols = [c for c in cache if c.startswith("ndcg_hyb_")]
        alpha_star = round(float(max(hyb_cols, key=lambda c: cache[c].mean()).split("_")[-1]), 1)
        df = dataset_scores(ds, list(cache.index), alpha_star)
        # Sanity: recomputed base scores must match Experiment 2's cache.
        assert np.allclose(df["bm25|ndcg@10"].to_numpy(), cache["ndcg_bm25"].to_numpy(), atol=1e-9), ds
        assert np.allclose(df["dense|ndcg@10"].to_numpy(), cache["ndcg_dense"].to_numpy(), atol=1e-9), ds
        dfs[ds] = df
        s = summarise(df)
        s.insert(0, "dataset", ds)
        s.insert(1, "alpha_star", alpha_star)
        s["single_rel_share"] = float((df["n_rel"] == 1).mean())
        per_ds.append(s)
        print(f"[{ds}] alpha*={alpha_star} n={len(df)} e5-base={'yes' if df.attrs['has_base'] else 'no'}")
    per_ds = pd.concat(per_ds, ignore_index=True)
    pooled = per_ds.groupby("perturbation", sort=False).apply(
        lambda g: pd.Series({c: np.average(g[c].dropna(), weights=g.loc[g[c].notna(), "n"]) if g[c].notna().any() else np.nan
                             for c in ["kappa_3way", "agree_3way", "bin_retained", "bin_flipped", "bin_lost", "ref_auc"]}
                            | {"datasets": int(len(g)), "n": int(g["n"].sum())}), include_groups=False).reset_index()
    ref = {w: pooled_ref_auc(dfs, w) for w in ["ndcg@5", "ndcg@20", "mrr@10", "e5-base", "qrels-half"]}
    single = float(np.mean(np.concatenate([(d["n_rel"] == 1).to_numpy() for d in dfs.values()])))

    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    per_ds.to_csv(out / "per_dataset.csv", index=False)
    pooled.to_csv(out / "pooled.csv", index=False)
    (out / "summary.json").write_text(json.dumps({"datasets": args.datasets, "pooled_ref_auc": ref,
                                                  "single_relevant_share": single,
                                                  "pooled": pooled.to_dict(orient="records")}, indent=2), encoding="utf-8")
    f = lambda v, p=3: "—" if pd.isna(v) else f"{v:.{p}f}"  # noqa: E731
    L = ["# Label reliability: is the per-query winner a stable property of the query?", "",
         f"Datasets: {', '.join(args.datasets)}. Base label: nDCG@10, e5-small-v2, official qrels. "
         f"Share of queries with a single relevant document: {single:.1%}.", "",
         "Pooled (query-weighted mean over datasets; qrels-half averaged over 5 random splits):", "",
         "| Perturbation | Datasets | n | κ (3-way) | 3-way agreement | Binary retained | Binary flipped | Binary lost | Reference AUC′ |",
         "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |"]
    for r in pooled.itertuples():
        L.append(f"| {r.perturbation} | {int(r.datasets)} | {int(r.n)} | {f(r.kappa_3way)} | {f(r.agree_3way)} | {f(r.bin_retained)} "
                 f"| {f(r.bin_flipped)} | {f(r.bin_lost)} | {f(r.ref_auc)} |")
    L += ["", "Reference AUC′ pooled over queries (perturbed dense − BM25 margin → base binary label):", "",
          "| Perturbation | queries | AUC′ [95% CI] |", "| :--- | :---: | :---: |"]
    for w, v in ref.items():
        if v:
            L.append(f"| {w} | {v['n']} | {v['auc']:.3f} [{v['ci'][0]:.3f}, {v['ci'][1]:.3f}] |")
    L += ["", "## Per dataset", "", "| Dataset | α* | Perturbation | n | κ (3-way) | Binary retained | Binary lost | Reference AUC′ |",
          "| :--- | :---: | :--- | :---: | :---: | :---: | :---: | :---: |"]
    for r in per_ds.itertuples():
        L.append(f"| {r.dataset} | {r.alpha_star} | {r.perturbation} | {r.n} | {f(r.kappa_3way)} | {f(r.bin_retained)} "
                 f"| {f(r.bin_lost)} | {f(r.ref_auc)} |")
    (out / "tables.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L[:20 + len(ref)]))


if __name__ == "__main__":
    main()
