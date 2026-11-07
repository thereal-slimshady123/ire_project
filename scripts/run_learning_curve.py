"""Learning curve: does the router improve with more training queries from one dataset?

A fixed test set (25% of queries, seed 42) is held out. Training subsets of size 250, 500, 1000, 2000, 4000 and all
remaining queries are drawn from the rest (5 seeded repeats each; 1 for "all"). At each size:
  * router - hybrid on the test set, with Experiment 2's fit_and_evaluate unchanged (alpha*, epsilon by inner CV,
    GBM router; per-query differences averaged over repeats, paired bootstrap 10,000);
  * learned winner AUC': a LightGBM classifier (router.yaml settings) on the 18 features predicting the E2(c) binary
    winner (dense-winner vs BM25-winner, hybrid at alpha = 0.6), AUC' on the test set's labelled queries, mean over
    repeats; bootstrap over test queries.
Plateau contrasts: (largest size) - 1000 for both metrics, paired bootstrap 95% CI.

Usage:
  python scripts/run_learning_curve.py --dataset hotpotqa --exp2_dir <dir> --output_dir <dir>
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
from router_retrieval.utils.config import load_config  # noqa: E402

spec = importlib.util.spec_from_file_location("exp2", ROOT / "scripts" / "run_experiment2_heldout_routing.py")
exp2 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(exp2)

SIZES, REPEATS, TEST_SHARE, SEED, N_BOOT = [250, 500, 1000, 2000, 4000], 5, 0.25, 42, 10000


def winner_labels(df):
    h = df["ndcg_hyb_0.6"]
    dense = (df["ndcg_dense"] > h) & (df["ndcg_dense"] > df["ndcg_bm25"])
    bm25 = (df["ndcg_bm25"] > h) & (df["ndcg_bm25"] > df["ndcg_dense"])
    return pd.Series(np.where(dense, 1, np.where(bm25, 0, -1)), index=df.index)


def fit_winner(train, router_cfg, seed):
    import lightgbm as lgb

    y = winner_labels(train)
    t = train[y >= 0]
    g = router_cfg["models"]["gbm"]
    m = lgb.LGBMClassifier(n_estimators=g["n_estimators"], learning_rate=g["learning_rate"], num_leaves=g["num_leaves"],
                           random_state=seed, deterministic=True, force_row_wise=True, verbose=-1)
    return m.fit(t[exp2.FEATURES], y[y >= 0])


def auc_prime(y, s):
    a = roc_auc_score(y, s)
    return max(a, 1 - a)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--exp2_dir", type=Path, required=True)
    ap.add_argument("--output_dir", type=Path, required=True)
    args = ap.parse_args()
    cfg = lambda n: load_config(ROOT / "configs" / n)  # noqa: E731
    router_cfg = cfg("router.yaml")
    alphas = [float(a) for a in cfg("hybrid.yaml")["alpha_grid"]]
    eps_grid = [float(e) for e in cfg("labels.yaml")["epsilon_grid"]]

    df = pd.read_csv(args.exp2_dir / "cache" / f"{args.dataset}.csv", index_col=0, dtype={"query_id": str})
    df.index = df.index.astype(str)
    rng = np.random.default_rng(SEED)
    ids = rng.permutation(df.index.to_numpy())
    n_test = int(round(TEST_SHARE * len(ids)))
    test, pool = df.loc[ids[:n_test]], ids[n_test:]
    y_te = winner_labels(test)
    lab = (y_te >= 0).to_numpy()
    sizes = [s for s in SIZES if s < len(pool)] + [len(pool)]

    diffs, scores = {}, {}
    for s in sizes:
        reps = 1 if s == len(pool) else REPEATS
        d_acc, p_acc = [], []
        for r in range(reps):
            tr_ids = np.random.default_rng(SEED + 1000 * r + s).choice(pool, size=s, replace=False)
            train = df.loc[tr_ids]
            out, _ = exp2.fit_and_evaluate(train, test, alphas, eps_grid, router_cfg, SEED)
            d_acc.append((out["router_gbm"] - out["hybrid"]).to_numpy())
            m = fit_winner(train, router_cfg, SEED)
            p_acc.append(m.predict_proba(test.loc[lab, exp2.FEATURES])[:, 1])
        diffs[s], scores[s] = np.mean(d_acc, axis=0), p_acc
        print(f"[{args.dataset}] n_train={s}: router-hybrid={diffs[s].mean():+.4f} "
              f"AUC'={np.mean([auc_prime(y_te[lab], p) for p in p_acc]):.3f}")

    # Bootstrap over test queries (router diff: all test queries; AUC': labelled test queries).
    b_rng = np.random.default_rng(SEED)
    yl = y_te[lab].to_numpy()
    auc_boot = {s: [] for s in sizes}
    for _ in range(2000):
        idx = b_rng.integers(0, len(yl), len(yl))
        if len(set(yl[idx])) < 2:
            continue
        for s in sizes:
            auc_boot[s].append(np.mean([auc_prime(yl[idx], p[idx]) for p in scores[s]]))
    auc_boot = {s: np.array(v) for s, v in auc_boot.items()}
    rows = []
    for s in sizes:
        t = paired_bootstrap_test(diffs[s], np.zeros(len(diffs[s])), N_BOOT, SEED)
        a = float(np.mean([auc_prime(yl, p) for p in scores[s]]))
        rows.append({"n_train": s, "router_minus_hybrid": t, "auc": a,
                     "auc_ci": [float(v) for v in np.percentile(auc_boot[s], [2.5, 97.5])]})
    big = sizes[-1]
    plateau = {"router_gain_big_minus_1000": paired_bootstrap_test(diffs[big], diffs[1000], N_BOOT, SEED) if 1000 in diffs else None,
               "auc_gain_big_minus_1000": ({"delta": float(rows[-1]["auc"] - rows[sizes.index(1000)]["auc"]),
                                            "ci": [float(v) for v in np.percentile(auc_boot[big] - auc_boot[1000], [2.5, 97.5])]}
                                           if 1000 in diffs else None)}
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps({"dataset": args.dataset, "n_test": n_test, "n_test_labelled": int(lab.sum()),
                                                  "curve": rows, "plateau": plateau}, indent=2), encoding="utf-8")
    ci = lambda t: f"{t['delta_mean']:+.4f} [{t['ci_95_low']:+.4f}, {t['ci_95_high']:+.4f}]"  # noqa: E731
    L = [f"# Learning curve: {args.dataset}", "", f"Fixed test set n = {n_test} ({int(lab.sum())} with a dense/BM25 winner). "
         f"Training subsets from the remaining {len(pool)} queries; {REPEATS} repeats per size (1 at the largest).", "",
         "| Training queries | Router − hybrid [95% CI] | Learned winner AUC′ [95% CI] |", "| :---: | :---: | :---: |"]
    for r in rows:
        L.append(f"| {r['n_train']} | {ci(r['router_minus_hybrid'])} | {r['auc']:.3f} [{r['auc_ci'][0]:.3f}, {r['auc_ci'][1]:.3f}] |")
    if plateau["auc_gain_big_minus_1000"]:
        g = plateau["auc_gain_big_minus_1000"]
        L += ["", f"Plateau contrasts ({big} vs 1000 training queries): router − hybrid change "
              f"{ci(plateau['router_gain_big_minus_1000'])}; AUC′ change {g['delta']:+.3f} [{g['ci'][0]:+.3f}, {g['ci'][1]:+.3f}]."]
    (out / "tables.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
