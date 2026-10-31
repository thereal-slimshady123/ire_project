"""Experiment 4b (exploratory, pre-declared): escalation to a cross-encoder blended with the hybrid score.

Design fixed in results/preregistration/experiment4b_interpolated_rerank_20261031.md (hash in SHA256SUMS.txt):
  * Experiment 4's datasets, folds, alpha*, hybrid top-50 pools and cached MiniLM-L6 scores, unchanged;
  * blended backend: within the top-50, lambda * minmax(CE) + (1 - lambda) * minmax(hybrid), lambda chosen
    on training queries only; ranks 51-100 keep hybrid order;
  * prerequisite: pooled always-blended - always-hybrid CI lower bound > 0;
  * H3 (pooled, B = 10%): predicted-gain - hybrid CI lower bound > 0 and retained gain >= 50%;
  * H4: per-query latency composed from the measured component timings (NFCorpus, SciFact, FiQA).

Usage:
  python scripts/run_experiment4b_interpolated_rerank.py
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
from router_retrieval.retrieval.cross_encoder import CrossEncoderReranker  # noqa: E402
from router_retrieval.utils.config import load_config  # noqa: E402
from router_retrieval.utils.logging_utils import get_logger  # noqa: E402
from router_retrieval.utils.seeding import seed_everything  # noqa: E402


def _load_script(name: str, file: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / file)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


exp4 = _load_script("exp4", "run_experiment4_escalation.py")
exp1, exp1b, exp2 = exp4.exp1, exp4.exp1b, exp4.exp2

log = get_logger("experiment4b")
PREREG = "results/preregistration/experiment4b_interpolated_rerank_20261031.md"
POOL = exp4.POOL
LAMBDAS = [round(0.1 * i, 1) for i in range(11)]
BUDGETS = [0.05, 0.10, 0.15, 0.20, 0.30, 0.50]
PRIMARY_BUDGET = 0.10
MIN_RETAINED = 0.5
POLICIES = ["predicted_gain", "router_confidence", "random", "oracle"]
HYBRID_MS = ["bm25_search", "dense_encode", "faiss_search", "hybrid_fusion"]


def parse_args():
    p = argparse.ArgumentParser(description="Experiment 4b: escalation to a blended cross-encoder")
    p.add_argument("--datasets", nargs="+", default=exp2.DATASETS)
    p.add_argument("--data_dir", type=Path, default=Path("data/raw"))
    p.add_argument("--processed_dir", type=Path, default=Path("data/processed"))
    p.add_argument("--config_dir", type=Path, default=Path("configs"))
    p.add_argument("--results_dir", type=Path, default=Path("results"))
    p.add_argument("--exp2_dir", type=Path, default=Path("results/experiment2"))
    p.add_argument("--latency_dir", type=Path, default=Path("results/policy_latency"))
    p.add_argument("--output_dir", type=Path, default=Path("results/experiment4b"))
    p.add_argument("--dense_model", type=str, default="intfloat/e5-small-v2")
    p.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    p.add_argument("--n_bootstrap", type=int, default=10000)
    return p.parse_args()


def _minmax(x: np.ndarray) -> np.ndarray:
    span = x.max() - x.min()
    return (x - x.min()) / span if span > 0 else np.zeros_like(x)


def blended_ranking(cands, ce_q: Dict[str, float], lam: float) -> List[str]:
    """Top-POOL reordered by lam * mm(CE) + (1 - lam) * mm(hybrid); exact ties keep hybrid order."""
    ranked = [c.doc_id for c in cands]
    head = cands[:POOL]
    if not head:
        return ranked
    s = lam * _minmax(np.array([ce_q[c.doc_id] for c in head])) + (1 - lam) * _minmax(np.array([c.score for c in head]))
    order = sorted(range(len(head)), key=lambda i: (-s[i], i))
    return [head[i].doc_id for i in order] + ranked[POOL:]


def apply_budgets(heldout: pd.DataFrame) -> None:
    """Experiment 4's rule: within every (dataset, fold), escalate the top-B share by each policy's score."""
    for B in BUDGETS:
        for pol in ("predicted_gain", "router_confidence", "oracle"):
            col = np.empty(len(heldout))
            esc_col = np.zeros(len(heldout), dtype=bool)
            for _, idx in heldout.groupby(["dataset", "fold"]).indices.items():
                sub = heldout.iloc[idx]
                order = np.argsort(-sub[f"score_{pol}"].values, kind="stable")
                esc = np.zeros(len(sub), dtype=bool)
                esc[order[: int(round(B * len(sub)))]] = True
                col[idx] = np.where(esc, sub["blended"].values, sub["hybrid"].values)
                esc_col[idx] = esc
            heldout[f"{pol}@{B}"] = col
            heldout[f"esc_{pol}@{B}"] = esc_col
        heldout[f"random@{B}"] = B * heldout["blended"] + (1 - B) * heldout["hybrid"]


def block(df: pd.DataFrame, n_boot: int, seed: int) -> dict:
    h, full = df["hybrid"].mean(), df["blended"].mean()
    res = {"n": len(df), "always_hybrid": float(h), "always_blended": float(full),
           "always_rerank_pure_ce": float(df["reranked_pure"].mean()),
           "blended_minus_hybrid": paired_bootstrap_test(df["blended"].values, df["hybrid"].values, n_boot, seed),
           "budgets": {}}
    for B in BUDGETS:
        b = {pol: {"mean": float(df[f"{pol}@{B}"].mean()),
                   "retained": float((df[f"{pol}@{B}"].mean() - h) / (full - h)) if full > h else None}
             for pol in POLICIES}
        for pol in ("predicted_gain", "router_confidence"):
            b[pol]["minus_hybrid"] = paired_bootstrap_test(df[f"{pol}@{B}"].values, df["hybrid"].values, n_boot, seed)
            b[pol]["minus_random"] = paired_bootstrap_test(df[f"{pol}@{B}"].values, df[f"random@{B}"].values, n_boot, seed)
        res["budgets"][str(B)] = b
    return res


def latency_table(heldout: pd.DataFrame, latency_dir: Path, datasets) -> dict:
    """Per-query latency composed from measured components, using the held-out escalation decisions."""
    out = {}
    for ds in datasets:
        path = latency_dir / f"{ds}_components.csv"
        if not path.exists():
            continue
        comp = pd.read_csv(path, index_col=0)
        sub = heldout[heldout["dataset"] == ds]
        comp = comp.loc[sub.index]
        hyb = comp[HYBRID_MS].sum(axis=1)
        ce = comp["cross_encoder_rerank"]
        systems = {"always_hybrid": hyb, "always_blended": hyb + ce}
        for B in BUDGETS:
            systems[f"predicted_gain@{B}"] = (hyb + comp["feature_extraction"] + comp["gain_model_inference"]
                                              + sub[f"esc_predicted_gain@{B}"] * ce)
            systems[f"router_confidence@{B}"] = (hyb + comp["feature_extraction"] + comp["router_inference"]
                                                 + sub[f"esc_router_confidence@{B}"] * ce)
        out[ds] = {name: {"mean_ms": float(v.mean()), "p95_ms": float(v.quantile(0.95))} for name, v in systems.items()}
    return out


def fmt_ci(t: dict) -> str:
    return f"{t['delta_mean']:+.4f} [{t['ci_95_low']:+.4f}, {t['ci_95_high']:+.4f}]"


def write_tables(summary: dict, path: Path) -> None:
    pooled = summary["pooled"]
    pb = pooled["budgets"][str(PRIMARY_BUDGET)]
    ret = pb["predicted_gain"]["retained"]
    rows = [*summary["per_dataset"].items(), ("**pooled**", pooled)]
    lines = [
        "# Experiment 4b results: escalation to a blended cross-encoder (exploratory)",
        "",
        f"Design fixed before running in `{PREREG}`. Held-out nDCG@10, 5-fold CV within each dataset (Experiment 4's "
        f"folds, α* and top-{POOL} pools). Blended backend = λ·mm(CE) + (1 − λ)·mm(hybrid), λ chosen on training folds.",
        "",
        f"**Prerequisite (pooled): always-blended − always-hybrid = {fmt_ci(pooled['blended_minus_hybrid'])} → "
        f"{summary['prerequisite'].upper()}.**",
        "",
        f"**H3 (pooled, B = {PRIMARY_BUDGET:.0%}): predicted-gain − always-hybrid = "
        f"{fmt_ci(pb['predicted_gain']['minus_hybrid'])}, retained = {'n/a' if ret is None else f'{ret:.0%}'} → "
        f"{summary['h3_verdict'].upper()}.**",
        "",
        "## Always-on systems",
        "",
        "| Dataset | n | Always-hybrid | Always-rerank (pure CE) | Always-blended | Blended − hybrid | λ per fold |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]
    for name, s in rows:
        lam = ", ".join(f"{f['lambda']:g}" for f in summary["fold_info"].get(name, []))
        lines.append(f"| {name} | {s['n']} | {s['always_hybrid']:.4f} | {s['always_rerank_pure_ce']:.4f} "
                     f"| {s['always_blended']:.4f} | {fmt_ci(s['blended_minus_hybrid'])} | {lam} |")
    lines += [
        "",
        f"## Escalation at B = {PRIMARY_BUDGET:.0%}",
        "",
        "| Dataset | Pred-gain | Router-conf | Random | Oracle | Pred-gain − hybrid | Pred-gain − random | Retained (pred-gain) |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]
    for name, s in rows:
        b = s["budgets"][str(PRIMARY_BUDGET)]
        r = b["predicted_gain"]["retained"]
        lines.append(f"| {name} | {b['predicted_gain']['mean']:.4f} | {b['router_confidence']['mean']:.4f} "
                     f"| {b['random']['mean']:.4f} | {b['oracle']['mean']:.4f} "
                     f"| {fmt_ci(b['predicted_gain']['minus_hybrid'])} | {fmt_ci(b['predicted_gain']['minus_random'])} "
                     f"| {'n/a' if r is None else f'{r:.0%}'} |")
    lines += [
        "",
        "## Pooled, every budget: mean nDCG@10 (retained share of the always-blended gain)",
        "",
        "| Budget | Predicted gain | Router confidence | Random | Oracle | Pred-gain − random |",
        "| :---: | :---: | :---: | :---: | :---: | :---: |",
    ]
    for B in BUDGETS:
        b = pooled["budgets"][str(B)]
        cells = []
        for p in POLICIES:
            r = b[p]["retained"]
            cells.append(f"{b[p]['mean']:.4f} ({'n/a' if r is None else f'{r:.0%}'})")
        lines.append(f"| {B:.0%} | " + " | ".join(cells) + f" | {fmt_ci(b['predicted_gain']['minus_random'])} |")
    if summary["latency"]:
        lines += [
            "",
            "## Latency, composed from measured per-query components (ms, mean / p95)",
            "",
            "| Dataset | Always-hybrid | Pred-gain @10% | Router-conf @10% | Always-blended |",
            "| :--- | :---: | :---: | :---: | :---: |",
        ]
        for ds, L in summary["latency"].items():
            cell = lambda k: f"{L[k]['mean_ms']:.1f} / {L[k]['p95_ms']:.1f}"  # noqa: E731
            lines.append(f"| {ds} | {cell('always_hybrid')} | {cell(f'predicted_gain@{PRIMARY_BUDGET}')} "
                         f"| {cell(f'router_confidence@{PRIMARY_BUDGET}')} | {cell('always_blended')} |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


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
    # Scores come from Experiment 4's cache; the model is only loaded if a pair is missing.
    ce = CrossEncoderReranker(model_name=ce_cfg["model_name"], batch_size=ce_cfg["batch_size"], device=args.device,
                              candidate_pool_size=POOL, max_length=ce_cfg["max_length"])

    heldout, fold_log = [], {}
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

        used = sorted(set(exp4.fold_alphas(df, alphas, k_folds, seed)))
        hyb = exp1.build_hybrid_runs(bm25_run, dense_run, used, k, hyb_cfg["normalization"]["method"])
        needed = {q: {c.doc_id for a in used for c in hyb[a][q][:POOL]} for q in queries}
        ce_scores = exp4.ce_scores_for_dataset(ds, path, needed, ce, args)

        # Per alpha: hybrid nDCG, blended nDCG at every lambda, hybrid score-distribution features.
        per_alpha = {}
        for a in used:
            rows = {}
            for q in queries:
                cands = hyb[a][q]
                row = {"ndcg_hyb": ndcg_at_k([c.doc_id for c in cands], qrels[q], 10),
                       "hyb_top1_score": cands[0].score if cands else 0.0,
                       "hyb_top1_top2_gap": cands[0].score - cands[1].score if len(cands) > 1 else 0.0,
                       "hyb_score_entropy": compute_score_entropy(cands)}
                for lam in LAMBDAS:
                    row[f"ndcg_blend_{lam}"] = ndcg_at_k(blended_ranking(cands, ce_scores.get(q, {}), lam), qrels[q], 10)
                rows[q] = row
            per_alpha[a] = pd.DataFrame.from_dict(rows, orient="index").loc[df.index]

        fold_log[ds] = []
        for i, (tr, te) in enumerate(create_kfold_splits(list(df.index), k_folds, seed)):
            alpha = max(alphas, key=lambda a: df.loc[tr, f"ndcg_hyb_{a}"].mean())
            pa = per_alpha[alpha]
            lam = max(LAMBDAS, key=lambda l: pa.loc[tr, f"ndcg_blend_{l}"].mean())
            X = df[exp2.FEATURES].join(pa[["hyb_top1_score", "hyb_top1_top2_gap", "hyb_score_entropy"]])
            gain = pa[f"ndcg_blend_{lam}"] - pa["ndcg_hyb"]

            import lightgbm as lgb
            g = router_cfg["models"]["gbm"]
            reg = lgb.LGBMRegressor(n_estimators=g["n_estimators"], learning_rate=g["learning_rate"],
                                    num_leaves=g["num_leaves"], random_state=seed, deterministic=True,
                                    force_row_wise=True, verbose=-1)
            reg.fit(X.loc[tr, exp4.GAIN_FEATURES], gain.loc[tr])
            R_tr = exp2.route_matrix(df.loc[tr], alpha)
            eps = exp2.tune_epsilon(df.loc[tr], alpha, eps_grid, router_cfg, seed)
            router = exp2.fit_router("gbm", df.loc[tr, exp2.FEATURES], R_tr, eps, router_cfg, seed)
            conf = (np.ones(len(te)) if getattr(router, "constant", None) is not None
                    else router.predict_proba(df.loc[te, exp2.FEATURES]).max(axis=1))

            out = pd.DataFrame(index=te)
            out["dataset"], out["fold"] = ds, i
            out["hybrid"], out["blended"] = pa.loc[te, "ndcg_hyb"], pa.loc[te, f"ndcg_blend_{lam}"]
            out["reranked_pure"] = pa.loc[te, "ndcg_blend_1.0"]
            out["score_predicted_gain"] = reg.predict(X.loc[te, exp4.GAIN_FEATURES])
            out["score_router_confidence"] = -conf
            out["score_oracle"] = gain.loc[te]
            heldout.append(out)
            fold_log[ds].append({"alpha": alpha, "lambda": lam, "epsilon": eps})
        log.info(f"[{ds}] folds done: " + "; ".join(f"α={f['alpha']:g} λ={f['lambda']:g}" for f in fold_log[ds]))

    heldout = pd.concat(heldout)
    heldout.index = [f"{d}:{q}" for d, q in zip(heldout["dataset"], heldout.index)]
    apply_budgets(heldout)

    pooled = block(heldout, args.n_bootstrap, seed)
    prereq = pooled["blended_minus_hybrid"]["ci_95_low"] > 0
    pb = pooled["budgets"][str(PRIMARY_BUDGET)]["predicted_gain"]
    h3 = prereq and pb["minus_hybrid"]["ci_95_low"] > 0 and (pb["retained"] or 0) >= MIN_RETAINED
    summary = {
        "preregistration": PREREG,
        "status": "exploratory",
        "cross_encoder_model": ce.model_name,
        "pooled": pooled,
        "per_dataset": {ds: block(heldout[heldout["dataset"] == ds], args.n_bootstrap, seed) for ds in fold_log},
        "prerequisite": "met" if prereq else "not met",
        "h3_verdict": "supported" if h3 else ("not supported" if prereq else "not testable"),
        "latency": latency_table(heldout, args.latency_dir, args.datasets),
        "fold_info": fold_log,
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    heldout.to_csv(args.output_dir / "per_query_heldout.csv")
    with open(args.output_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=float)
    write_tables(summary, args.output_dir / "tables.md")
    log.info(f"PREREQUISITE: always-blended − always-hybrid = {fmt_ci(pooled['blended_minus_hybrid'])} → "
             f"{summary['prerequisite'].upper()}")
    log.info(f"H3: predicted-gain@10% − hybrid = {fmt_ci(pb['minus_hybrid'])}, retained = {pb['retained']} → "
             f"{summary['h3_verdict'].upper()}")


if __name__ == "__main__":
    main()
