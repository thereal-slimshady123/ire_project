"""E4 (amended): skipping dense retrieval, evaluated at fixed operating points instead of a fixed loss budget.

Pre-registered in results/preregistration/e4_matched_rate_20261107.md. Reuses Experiment 3's features,
label, classifier and heuristics (scripts/run_experiment3_skip_dense.py) unchanged; only the operating point
and the success criteria differ:

  * per test fold, each method skips dense retrieval on the top-k held-out queries by its score
    (k = round(r * n_fold)); no relevance labels of test queries are used to pick k;
  * random skipping at the same realised rate k/n is evaluated in expectation;
  * P1 (r = 20%): router - hybrid CI low >= -0.01 and router - random CI low > 0;
  * P2: fraction of oracle gain captured, FOG = sum_r(router - random) / sum_r(oracle - random),
    r in {5%, 10%, ..., 50%}, bootstrap 95% CI low > 0;
  * secondary: router - H1 (BM25 top-1/top-2 ratio) at 20% and in FOG; per-dataset results.

Usage:
  python scripts/run_e4_matched_rate.py --datasets <ds ...> --exp2_dir <dir> --output_dir <dir>
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
from router_retrieval.evaluation.significance import paired_bootstrap_test  # noqa: E402
from router_retrieval.utils.config import load_config  # noqa: E402
from router_retrieval.utils.seeding import seed_everything  # noqa: E402

spec = importlib.util.spec_from_file_location("exp3", ROOT / "scripts" / "run_experiment3_skip_dense.py")
exp3 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(exp3)

RATES = np.round(np.arange(0.05, 0.5001, 0.05), 2)
PRIMARY_RATE = 0.20
NONINFERIORITY_MARGIN = -0.01
METHODS = ["router", "H1_bm25_top_ratio", "H2_max_idf", "oracle"]


def skip_top(score: np.ndarray, B: np.ndarray, H: np.ndarray, r: float) -> np.ndarray:
    """Per-query nDCG when the top round(r*n) queries by score answer from BM25 only."""
    k = int(round(r * len(score)))
    out = H.copy()
    idx = np.argsort(-score, kind="stable")[:k]
    out[idx] = B[idx]
    return out


def fold_scores(train: pd.DataFrame, test: pd.DataFrame, alphas, router_cfg, seed) -> dict:
    alpha = max(alphas, key=lambda a: train[f"ndcg_hyb_{a}"].mean())
    H_tr, B_tr = train[f"ndcg_hyb_{alpha}"].values, train["ndcg_bm25"].values
    H_te, B_te = test[f"ndcg_hyb_{alpha}"].values, test["ndcg_bm25"].values
    y_tr = (B_tr >= H_tr - exp3.LABEL_MARGIN).astype(int)
    model = exp3.fit_classifier(train[exp3.SKIP_FEATURES], y_tr, router_cfg, seed)
    scores = {"router": exp3.predict_p(model, test[exp3.SKIP_FEATURES])}
    for name, feat in exp3.HEURISTICS.items():
        # Direction: the sign with the higher training-fold mean nDCG at the primary skip rate.
        sign = max((1.0, -1.0), key=lambda s: skip_top(s * train[feat].values, B_tr, H_tr, PRIMARY_RATE).mean())
        scores[name] = sign * test[feat].values
    scores["oracle"] = B_te - H_te
    return {"alpha": alpha, "B": B_te, "H": H_te, "scores": scores}


def bootstrap_ratio(num: np.ndarray, den: np.ndarray, n_boot: int, seed: int):
    rng = np.random.default_rng(seed)
    n, out = len(num), []
    for start in range(0, n_boot, 500):
        idx = rng.integers(0, n, (min(500, n_boot - start), n))
        out.append(num[idx].mean(axis=1) / den[idx].mean(axis=1))
    out = np.concatenate(out)
    lo, hi = np.percentile(out, [2.5, 97.5])
    return {"value": float(num.mean() / den.mean()), "ci_95_low": float(lo), "ci_95_high": float(hi)}


def bootstrap_ratio_diff(n1, n2, den, n_boot, seed):
    rng = np.random.default_rng(seed)
    n, out = len(den), []
    for start in range(0, n_boot, 500):
        idx = rng.integers(0, n, (min(500, n_boot - start), n))
        d = den[idx].mean(axis=1)
        out.append(n1[idx].mean(axis=1) / d - n2[idx].mean(axis=1) / d)
    out = np.concatenate(out)
    lo, hi = np.percentile(out, [2.5, 97.5])
    return {"value": float(n1.mean() / den.mean() - n2.mean() / den.mean()), "ci_95_low": float(lo), "ci_95_high": float(hi)}


def analyse(per_q: pd.DataFrame, n_boot: int, seed: int) -> dict:
    b = lambda a, c: paired_bootstrap_test(per_q[a].values, per_q[c].values, n_boot, seed)  # noqa: E731
    p = f"{PRIMARY_RATE:.2f}"
    res = {
        "n": len(per_q),
        "mean_at_primary": {m: float(per_q[f"{m}@{p}"].mean()) for m in [*METHODS, "random"]} | {"hybrid": float(per_q["hybrid"].mean())},
        "router_minus_hybrid": b(f"router@{p}", "hybrid"),
        "router_minus_random": b(f"router@{p}", f"random@{p}"),
        "H1_minus_random": b(f"H1_bm25_top_ratio@{p}", f"random@{p}"),
        "router_minus_H1": b(f"router@{p}", f"H1_bm25_top_ratio@{p}"),
    }
    gain = {m: sum(per_q[f"{m}@{r:.2f}"] - per_q[f"random@{r:.2f}"] for r in RATES).to_numpy() for m in METHODS}
    res["fog"] = {m: bootstrap_ratio(gain[m], gain["oracle"], n_boot, seed) for m in METHODS if m != "oracle"}
    res["fog_router_minus_H1"] = bootstrap_ratio_diff(gain["router"], gain["H1_bm25_top_ratio"], gain["oracle"], n_boot, seed)
    res["curve"] = {m: [float(per_q[f"{m}@{r:.2f}"].mean()) for r in RATES] for m in [*METHODS, "random"]}
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--datasets", nargs="+", required=True)
    ap.add_argument("--config_dir", type=Path, default=Path("configs"))
    ap.add_argument("--results_dir", type=Path, default=Path("results"))
    ap.add_argument("--exp2_dir", type=Path, nargs="+", required=True,
                    help="Experiment 2 output dir(s); each dataset is read from the first one holding its cache")
    ap.add_argument("--output_dir", type=Path, required=True)
    ap.add_argument("--n_bootstrap", type=int, default=10000)
    args = ap.parse_args()

    exp_cfg = load_config(args.config_dir / "experiment.yaml")
    router_cfg = load_config(args.config_dir / "router.yaml")
    alphas = [float(a) for a in load_config(args.config_dir / "hybrid.yaml")["alpha_grid"]]
    seed, k_folds = exp_cfg["seed"], exp_cfg["k_folds"]
    seed_everything(seed)

    rows = []
    for ds in args.datasets:
        src = next(d for d in args.exp2_dir if (d / "cache" / f"{ds}.csv").exists())
        df = exp3.load_table(ds, argparse.Namespace(exp2_dir=src, results_dir=args.results_dir))
        for tr, te in create_kfold_splits(list(df.index), k_folds, seed):
            f = fold_scores(df.loc[tr], df.loc[te], alphas, router_cfg, seed)
            B, H = f["B"], f["H"]
            out = pd.DataFrame(index=[f"{ds}:{q}" for q in te])
            out["dataset"], out["hybrid"], out["bm25"] = ds, H, B
            for r in RATES:
                k = int(round(r * len(H)))
                out[f"random@{r:.2f}"] = (k / len(H)) * B + (1 - k / len(H)) * H
                for m in METHODS:
                    out[f"{m}@{r:.2f}"] = skip_top(f["scores"][m], B, H, r)
            rows.append(out)
    per_q = pd.concat(rows)

    pooled = analyse(per_q, args.n_bootstrap, seed)
    per_ds = {ds: analyse(per_q[per_q["dataset"] == ds], args.n_bootstrap, seed) for ds in args.datasets}
    p1a = pooled["router_minus_hybrid"]["ci_95_low"] >= NONINFERIORITY_MARGIN
    p1b = pooled["router_minus_random"]["ci_95_low"] > 0
    p2 = pooled["fog"]["router"]["ci_95_low"] > 0
    reject = (pooled["router_minus_random"]["ci_95_high"] <= 0 or pooled["fog"]["router"]["ci_95_high"] <= 0
              or pooled["router_minus_hybrid"]["ci_95_high"] < NONINFERIORITY_MARGIN)
    verdict = "ACCEPTED" if (p1a and p1b and p2) else "REJECTED" if reject else "INCONCLUSIVE"

    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    per_q.to_csv(out / "per_query_heldout.csv")
    summary = {"datasets": args.datasets, "rates": RATES.tolist(), "primary_rate": PRIMARY_RATE,
               "verdict": verdict, "criteria": {"P1a_noninferior": p1a, "P1b_beats_random": p1b, "P2_fog_positive": p2},
               "pooled": pooled, "per_dataset": per_ds}
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=lambda o: bool(o) if isinstance(o, np.bool_) else float(o)),
                                      encoding="utf-8")

    ci = lambda t: f"{t['delta_mean']:+.4f} [{t['ci_95_low']:+.4f}, {t['ci_95_high']:+.4f}]"  # noqa: E731
    cr = lambda t: f"{t['value']:.3f} [{t['ci_95_low']:.3f}, {t['ci_95_high']:.3f}]"  # noqa: E731
    yn = lambda x: "PASS" if x else "FAIL"  # noqa: E731
    L = ["# E4 (amended): skipping dense retrieval at matched skip rates", "",
         "Pre-registered in `results/preregistration/e4_matched_rate_20261107.md`. Held-out nDCG@10, 5-fold CV per dataset, "
         f"paired bootstrap ({args.n_bootstrap} resamples). Datasets: {', '.join(args.datasets)} (n = {pooled['n']}).", "",
         f"**Verdict: {verdict}** — P1a router − hybrid at 20% CI low ≥ −0.01: {yn(p1a)}; "
         f"P1b router − random at 20% CI low > 0: {yn(p1b)}; P2 FOG CI low > 0: {yn(p2)}.", "",
         "| Dataset | n | Hybrid | Router @20% | Router − hybrid | Router − random | H1 − random | Router − H1 | FOG router | FOG H1 | FOG router − H1 |",
         "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |"]
    for name, s in [*per_ds.items(), ("**pooled**", pooled)]:
        L.append(f"| {name} | {s['n']} | {s['mean_at_primary']['hybrid']:.4f} | {s['mean_at_primary']['router']:.4f} "
                 f"| {ci(s['router_minus_hybrid'])} | {ci(s['router_minus_random'])} | {ci(s['H1_minus_random'])} "
                 f"| {ci(s['router_minus_H1'])} | {cr(s['fog']['router'])} | {cr(s['fog']['H1_bm25_top_ratio'])} "
                 f"| {cr(s['fog_router_minus_H1'])} |")
    L += ["", "## Pooled curve: mean nDCG@10 by skip rate", "",
          "| Skip rate | Router | H1 | H2 | Random | Oracle |", "| :---: | :---: | :---: | :---: | :---: | :---: |"]
    c = pooled["curve"]
    for i, r in enumerate(RATES):
        L.append(f"| {r:.0%} | {c['router'][i]:.4f} | {c['H1_bm25_top_ratio'][i]:.4f} | {c['H2_max_idf'][i]:.4f} "
                 f"| {c['random'][i]:.4f} | {c['oracle'][i]:.4f} |")
    (out / "tables.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"VERDICT {verdict}: router-hybrid@20%={ci(pooled['router_minus_hybrid'])} "
          f"router-random@20%={ci(pooled['router_minus_random'])} FOG={cr(pooled['fog']['router'])}")


if __name__ == "__main__":
    main()
