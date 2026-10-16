"""Experiment 4: does selective cross-encoder escalation beat random escalation at the same budget?

Implements the pre-registered protocol in improvements.md §4.2:
  * base system = always-hybrid (training-fold alpha*); escalation = rerank the hybrid top-50 with
    cross-encoder/ms-marco-MiniLM-L-6-v2 (ranks 51-100 appended unchanged);
  * cross-encoder scores precomputed once for every (query, doc) any fold's hybrid top-50 can contain;
  * policies at budget B: predicted-gain (LightGBM regressor, primary), router-confidence (Experiment 2
    GBM router, lowest max-probability first), random (expected value), oracle (true gain);
  * success (pooled, B = 10%): predicted-gain - random has a 95% CI lower bound > 0.

Usage:
  python scripts/run_experiment4_escalation.py
"""

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from router_retrieval.data.splits import create_kfold_splits  # noqa: E402
from router_retrieval.evaluation.ranking_metrics import ndcg_at_k  # noqa: E402
from router_retrieval.evaluation.significance import paired_bootstrap_test  # noqa: E402
from router_retrieval.features.retrieval_signal_features import compute_score_entropy  # noqa: E402
from router_retrieval.retrieval.bm25 import document_text  # noqa: E402
from router_retrieval.retrieval.cross_encoder import CrossEncoderReranker  # noqa: E402
from router_retrieval.utils.config import load_config  # noqa: E402
from router_retrieval.utils.logging_utils import get_logger  # noqa: E402
from router_retrieval.utils.seeding import seed_everything  # noqa: E402


def _load_script(name: str, file: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / file)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


exp1 = _load_script("exp1", "run_experiment1_hybrid_degradation.py")
exp1b = _load_script("exp1b", "run_experiment1b_query_types.py")
exp2 = _load_script("exp2", "run_experiment2_heldout_routing.py")

log = get_logger("experiment4")
BUDGETS = [0.05, 0.10, 0.15, 0.20, 0.30, 0.50, 1.0]
PRIMARY_BUDGET = 0.10
POOL = 50
GAIN_FEATURES = exp2.FEATURES + ["hyb_top1_score", "hyb_top1_top2_gap", "hyb_score_entropy"]
POLICIES = ["predicted_gain", "router_confidence", "random", "oracle"]


def parse_args():
    p = argparse.ArgumentParser(description="Experiment 4: selective cross-encoder escalation")
    p.add_argument("--datasets", nargs="+", default=exp2.DATASETS)
    p.add_argument("--data_dir", type=Path, default=Path("data/raw"))
    p.add_argument("--processed_dir", type=Path, default=Path("data/processed"))
    p.add_argument("--config_dir", type=Path, default=Path("configs"))
    p.add_argument("--results_dir", type=Path, default=Path("results"))
    p.add_argument("--exp2_dir", type=Path, default=Path("results/experiment2"))
    p.add_argument("--output_dir", type=Path, default=Path("results/experiment4"))
    p.add_argument("--dense_model", type=str, default="intfloat/e5-small-v2")
    p.add_argument("--n_bootstrap", type=int, default=10000)
    p.add_argument("--no_plots", action="store_true")
    return p.parse_args()


def fold_alphas(df: pd.DataFrame, alphas, k_folds, seed) -> List[float]:
    """alpha* of every training fold (label use limited to training folds, as in Experiments 2/3)."""
    return [max(alphas, key=lambda a: df.loc[tr, f"ndcg_hyb_{a}"].mean())
            for tr, _ in create_kfold_splits(list(df.index), k_folds, seed)]


def ce_scores_for_dataset(ds: str, path: Path, needed: Dict[str, set], ce: CrossEncoderReranker, args) -> Dict:
    """Cross-encoder score for every needed (query, doc) pair; cached as TSV."""
    cache = args.results_dir / "runs" / ds / f"ce_{ce.model_name.replace('/', '__')}.tsv"
    scores: Dict[str, Dict[str, float]] = {}
    if cache.exists():
        for line in open(cache, encoding="utf-8"):
            q, d, s = line.rstrip("\n").split("\t")
            scores.setdefault(q, {})[d] = float(s)
    missing = [(q, d) for q, docs in needed.items() for d in docs if d not in scores.get(q, {})]
    if missing:
        corpus, queries, _ = exp1b.load_split(path)
        log.info(f"[{ds}] cross-encoder scoring {len(missing)} pairs on {ce.device}")
        vals = ce.score_pairs([(queries[q].text, document_text(corpus[d])) for q, d in missing], show_progress_bar=True)
        with open(cache, "a", encoding="utf-8") as f:
            for (q, d), s in zip(missing, vals):
                scores.setdefault(q, {})[d] = s
                f.write(f"{q}\t{d}\t{s:.6f}\n")
    return scores


def main():
    args = parse_args()
    exp_cfg = load_config(args.config_dir / "experiment.yaml")
    ret_cfg = load_config(args.config_dir / "retrievers.yaml")
    hyb_cfg = load_config(args.config_dir / "hybrid.yaml")
    router_cfg = load_config(args.config_dir / "router.yaml")
    ce_cfg = load_config(args.config_dir / "cross_encoder.yaml")
    eps_grid = [float(e) for e in load_config(args.config_dir / "labels.yaml")["epsilon_grid"]]
    seed, k_folds = exp_cfg["seed"], exp_cfg["k_folds"]
    seed_everything(seed)
    alphas = [float(a) for a in hyb_cfg["alpha_grid"]]
    k = ret_cfg["dense"].get("top_k", 100)
    ce = CrossEncoderReranker(model_name=ce_cfg["model_name"], batch_size=64, device="auto",
                              candidate_pool_size=POOL, max_length=ce_cfg["max_length"])

    heldout, fold_log, rerank_gain = [], {}, {}
    for ds in args.datasets:
        df = pd.read_csv(args.exp2_dir / "cache" / f"{ds}.csv", index_col=0, dtype={"query_id": str})
        df.index = df.index.astype(str)
        path = args.processed_dir / "quora-sub" if ds == "quora" else args.data_dir / ds
        _, queries, qrels = exp1b.load_split(path)
        bm25_run = exp1.read_trec_run(args.results_dir / "runs" / ds / "bm25_k1-1.2_b-0.75.trec")
        dense_run = exp1.read_trec_run(args.results_dir / "runs" / ds / f"dense_{args.dense_model.replace('/', '__')}.trec")
        for q in queries:
            bm25_run.setdefault(q, [])
            dense_run.setdefault(q, [])

        used = sorted(set(fold_alphas(df, alphas, k_folds, seed)))
        hyb = exp1.build_hybrid_runs(bm25_run, dense_run, used, k, hyb_cfg["normalization"]["method"])
        needed = {q: {c.doc_id for a in used for c in hyb[a][q][:POOL]} for q in queries}
        ce_scores = ce_scores_for_dataset(ds, path, needed, ce, args)

        # Per alpha: hybrid nDCG, escalated nDCG, hybrid score-distribution features.
        per_alpha = {}
        for a in used:
            rows = {}
            for q in queries:
                ranked = [c.doc_id for c in hyb[a][q]]
                head = sorted(ranked[:POOL], key=lambda d: (-ce_scores[q][d], ranked.index(d)))
                escalated = head + ranked[POOL:]
                cands = hyb[a][q]
                rows[q] = {
                    "ndcg_hyb": ndcg_at_k(ranked, qrels[q], 10),
                    "ndcg_ce": ndcg_at_k(escalated, qrels[q], 10),
                    "hyb_top1_score": cands[0].score if cands else 0.0,
                    "hyb_top1_top2_gap": cands[0].score - cands[1].score if len(cands) > 1 else 0.0,
                    "hyb_score_entropy": compute_score_entropy(cands),
                }
            per_alpha[a] = pd.DataFrame.from_dict(rows, orient="index")

        fold_log[ds] = []
        for i, (tr, te) in enumerate(create_kfold_splits(list(df.index), k_folds, seed)):
            alpha = max(alphas, key=lambda a: df.loc[tr, f"ndcg_hyb_{a}"].mean())
            pa = per_alpha[alpha]
            X = df[exp2.FEATURES].join(pa[["hyb_top1_score", "hyb_top1_top2_gap", "hyb_score_entropy"]])
            gain = pa["ndcg_ce"] - pa["ndcg_hyb"]

            # Predicted-gain policy: LightGBM regressor on training-fold gains.
            import lightgbm as lgb
            g = router_cfg["models"]["gbm"]
            reg = lgb.LGBMRegressor(n_estimators=g["n_estimators"], learning_rate=g["learning_rate"],
                                    num_leaves=g["num_leaves"], random_state=seed, deterministic=True,
                                    force_row_wise=True, verbose=-1)
            reg.fit(X.loc[tr, GAIN_FEATURES], gain.loc[tr])
            # Router-confidence policy: Experiment 2's GBM router, retrained on this training fold.
            R_tr = exp2.route_matrix(df.loc[tr], alpha)
            eps = exp2.tune_epsilon(df.loc[tr], alpha, eps_grid, router_cfg, seed)
            router = exp2.fit_router("gbm", df.loc[tr, exp2.FEATURES], R_tr, eps, router_cfg, seed)
            conf = (np.ones(len(te)) if getattr(router, "constant", None) is not None
                    else router.predict_proba(df.loc[te, exp2.FEATURES]).max(axis=1))

            out = pd.DataFrame(index=te)
            out["dataset"], out["fold"] = ds, i
            out["hybrid"], out["reranked"], out["gain"] = pa.loc[te, "ndcg_hyb"], pa.loc[te, "ndcg_ce"], gain.loc[te]
            out["score_predicted_gain"] = reg.predict(X.loc[te, GAIN_FEATURES])
            out["score_router_confidence"] = -conf  # least confident first
            out["score_oracle"] = gain.loc[te]
            heldout.append(out)
            fold_log[ds].append({"alpha": alpha, "epsilon": eps})
        rerank_gain[ds] = float((per_alpha[used[0]]["ndcg_ce"] - per_alpha[used[0]]["ndcg_hyb"]).mean())
        log.info(f"[{ds}] alphas {used}; always-rerank gain (alpha={used[0]}) = {rerank_gain[ds]:+.4f}")

    heldout = pd.concat(heldout)
    heldout.index = [f"{d}:{q}" for d, q in zip(heldout["dataset"], heldout.index)]

    # Apply each policy at each budget within every (dataset, fold): escalate the top-B share by score.
    for B in BUDGETS:
        for pol in ("predicted_gain", "router_confidence", "oracle"):
            col = np.empty(len(heldout))
            for _, idx in heldout.groupby(["dataset", "fold"]).indices.items():
                sub = heldout.iloc[idx]
                n_esc = int(round(B * len(sub)))
                order = np.argsort(-sub[f"score_{pol}"].values, kind="stable")
                esc = np.zeros(len(sub), dtype=bool)
                esc[order[:n_esc]] = True
                col[idx] = np.where(esc, sub["reranked"].values, sub["hybrid"].values)
            heldout[f"{pol}@{B}"] = col
        heldout[f"random@{B}"] = B * heldout["reranked"] + (1 - B) * heldout["hybrid"]

    def block(df):
        h, full = df["hybrid"].mean(), df["reranked"].mean()
        res = {"n": len(df), "always_hybrid": float(h), "always_rerank": float(full), "budgets": {}}
        for B in BUDGETS:
            res["budgets"][str(B)] = {
                pol: {"mean": float(df[f"{pol}@{B}"].mean()),
                      "retained": float((df[f"{pol}@{B}"].mean() - h) / (full - h)) if full != h else None}
                for pol in POLICIES
            }
        res["primary_pg_minus_random"] = paired_bootstrap_test(
            df[f"predicted_gain@{PRIMARY_BUDGET}"].values, df[f"random@{PRIMARY_BUDGET}"].values, args.n_bootstrap, seed)
        res["rc_minus_random"] = paired_bootstrap_test(
            df[f"router_confidence@{PRIMARY_BUDGET}"].values, df[f"random@{PRIMARY_BUDGET}"].values, args.n_bootstrap, seed)
        return res

    pooled = block(heldout)
    summary = {"preregistration": "improvements.md §4.2", "pooled": pooled,
               "per_dataset": {ds: block(heldout[heldout["dataset"] == ds]) for ds in args.datasets},
               "always_rerank_gain_by_dataset": rerank_gain, "fold_info": fold_log,
               "verdict": "success" if pooled["primary_pg_minus_random"]["ci_95_low"] > 0 else "negative"}
    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    heldout.to_csv(out_dir / "per_query_heldout.csv")
    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=float)

    if not args.no_plots:
        plt = exp1._style()
        fig, ax = plt.subplots(figsize=(6.6, 4.2))
        xs = [0.0] + BUDGETS
        colors = {"predicted_gain": "#2a78d6", "router_confidence": "#eb6834"}
        labels = {"predicted_gain": "Predicted gain (primary)", "router_confidence": "Router confidence",
                  "random": "Random escalation", "oracle": "Oracle (bound)"}
        for pol in POLICIES:
            ys = [pooled["always_hybrid"]] + [pooled["budgets"][str(B)][pol]["mean"] for B in BUDGETS]
            style = {"random": dict(color=exp1.INK_2, linestyle=(0, (4, 3)), linewidth=1.4),
                     "oracle": dict(color=exp1.INK_2, linestyle=(0, (1, 2)), linewidth=1.4)}.get(
                pol, dict(color=colors.get(pol), marker="o", markersize=4))
            ax.plot(xs, ys, label=labels[pol], **style)
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
        ax.set(xlabel="Share of queries escalated to the cross-encoder", ylabel="Mean held-out nDCG@10 (pooled)",
               title="Experiment 4: quality vs. escalation budget")
        ax.legend(loc="lower right")
        fig.tight_layout()
        (out_dir / "figures").mkdir(parents=True, exist_ok=True)
        fig.savefig(out_dir / "figures" / "escalation_curve.png", dpi=200)
        plt.close(fig)

    ci = lambda t: f"{t['delta_mean']:+.4f} [{t['ci_95_low']:+.4f}, {t['ci_95_high']:+.4f}]"  # noqa: E731
    lines = [
        "# Experiment 4 results: selective cross-encoder escalation",
        "",
        "Pre-registered in improvements.md §4.2. Held-out nDCG@10; base = always-hybrid (training-fold α*); "
        f"escalation = MiniLM cross-encoder rerank of the hybrid top-{POOL}.",
        "",
        f"**Primary (pooled, B = 10%): predicted-gain − random = {ci(pooled['primary_pg_minus_random'])} → "
        f"{summary['verdict'].upper()}.** Router-confidence − random = {ci(pooled['rc_minus_random'])}.",
        "",
        "| Dataset | n | Always-hybrid | Always-rerank | Pred-gain @10% | Router-conf @10% | Random @10% | Oracle @10% "
        "| Pred-gain − random @10% | Retained @10% (pred-gain) |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]
    for name, s in [*summary["per_dataset"].items(), ("**pooled**", pooled)]:
        b = s["budgets"][str(PRIMARY_BUDGET)]
        ret = b["predicted_gain"]["retained"]
        lines.append(
            f"| {name} | {s['n']} | {s['always_hybrid']:.4f} | {s['always_rerank']:.4f} | {b['predicted_gain']['mean']:.4f} "
            f"| {b['router_confidence']['mean']:.4f} | {b['random']['mean']:.4f} | {b['oracle']['mean']:.4f} "
            f"| {ci(s['primary_pg_minus_random'])} | {'n/a' if ret is None else f'{ret:.0%}'} |"
        )
    lines += ["", "## Pooled: quality retained = (policy − hybrid) / (always-rerank − hybrid)", "",
              "| Budget | Predicted gain | Router confidence | Random | Oracle |", "| :---: | :---: | :---: | :---: | :---: |"]
    for B in BUDGETS:
        b = pooled["budgets"][str(B)]
        lines.append(f"| {B:.0%} | " + " | ".join(
            f"{b[p]['mean']:.4f} ({b[p]['retained']:.0%})" for p in POLICIES) + " |")
    (out_dir / "tables.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    log.info(f"VERDICT {summary['verdict'].upper()}: pred-gain − random @10% = {ci(pooled['primary_pg_minus_random'])}")


if __name__ == "__main__":
    main()
