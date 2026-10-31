"""Experiment 4c (exploratory, pre-declared): blended cross-encoder reranking of every query's top-k.

Design fixed in results/preregistration/experiment4c_topk_blended_rerank_20261031.md (hash in SHA256SUMS.txt):
  * Experiment 4b's datasets, folds, alpha*, hybrid candidates and cached MiniLM-L6 scores (the model is
    never loaded); the blended score lambda * minmax(CE) + (1 - lambda) * minmax(hybrid) is applied to the
    hybrid top-k only, with lambda chosen per fold and per k on training queries;
  * k in {3, 5, 10, 20, 50}; k = 5 is primary (5 CE pairs per query, the average cost of escalating 10% of
    queries over 50 documents); k = 50 is always-blended, the retained-gain reference;
  * H3' (pooled, k = 5): top-5 - hybrid CI lower bound > 0 and retained gain >= 50%;
  * secondary: top-5 vs Experiment 4b's escalation policies at B = 10% (matched cost), every k, per dataset;
  * H4: latency lower bound = measured hybrid path + k x measured CE time per pair (NFCorpus, SciFact, FiQA).

Usage:
  python scripts/run_experiment4c_topk_blended_rerank.py
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
from router_retrieval.utils.config import load_config  # noqa: E402
from router_retrieval.utils.logging_utils import get_logger  # noqa: E402
from router_retrieval.utils.seeding import seed_everything  # noqa: E402


def _load_script(name: str, file: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / file)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


exp4b = _load_script("exp4b", "run_experiment4b_interpolated_rerank.py")
exp2e = _load_script("exp2e", "run_experiment2e_ce_feature_router.py")
exp4, exp1, exp1b, exp2 = exp4b.exp4, exp4b.exp1, exp4b.exp1b, exp4b.exp2

log = get_logger("experiment4c")
PREREG = "results/preregistration/experiment4c_topk_blended_rerank_20261031.md"
LAMBDAS = exp4b.LAMBDAS
KS = [3, 5, 10, 20, 50]
PRIMARY_K = 5
FULL_K = 50  # always-blended (Experiment 4b's backend)
MIN_RETAINED = exp4b.MIN_RETAINED
MATCHED = ["predicted_gain@0.1", "router_confidence@0.1", "random@0.1"]  # Experiment 4b, same mean CE pairs


def parse_args():
    p = argparse.ArgumentParser(description="Experiment 4c: blended reranking of every query's top-k")
    p.add_argument("--datasets", nargs="+", default=exp2.DATASETS)
    p.add_argument("--data_dir", type=Path, default=Path("data/raw"))
    p.add_argument("--processed_dir", type=Path, default=Path("data/processed"))
    p.add_argument("--config_dir", type=Path, default=Path("configs"))
    p.add_argument("--results_dir", type=Path, default=Path("results"))
    p.add_argument("--exp2_dir", type=Path, default=Path("results/experiment2"))
    p.add_argument("--exp4b_dir", type=Path, default=Path("results/experiment4b"))
    p.add_argument("--latency_dir", type=Path, default=Path("results/policy_latency"))
    p.add_argument("--output_dir", type=Path, default=Path("results/experiment4c"))
    p.add_argument("--dense_model", type=str, default="intfloat/e5-small-v2")
    p.add_argument("--n_bootstrap", type=int, default=10000)
    return p.parse_args()


def topk_blended_ranking(cands, ce_q: Dict[str, float], lam: float, k: int) -> List[str]:
    """Experiment 4b's blended score on the hybrid top-k only; ranks below k keep hybrid order."""
    ranked = [c.doc_id for c in cands]
    head = cands[:k]
    if not head:
        return ranked
    s = (lam * exp4b._minmax(np.array([ce_q[c.doc_id] for c in head]))
         + (1 - lam) * exp4b._minmax(np.array([c.score for c in head])))
    order = sorted(range(len(head)), key=lambda i: (-s[i], i))
    return [head[i].doc_id for i in order] + ranked[k:]


# --------------------------------------------------------------------------------------
# Reporting
# --------------------------------------------------------------------------------------

def block(df: pd.DataFrame, n_boot: int, seed: int) -> dict:
    h, full = df["hybrid"].mean(), df[f"top{FULL_K}"].mean()
    res = {"n": len(df), "always_hybrid": float(h), "always_blended": float(full), "k": {}, "matched_cost": {}}
    for k in KS:
        m = float(df[f"top{k}"].mean())
        res["k"][str(k)] = {"mean": m, "retained": float((m - h) / (full - h)) if full > h else None,
                            "minus_hybrid": paired_bootstrap_test(df[f"top{k}"].values, df["hybrid"].values, n_boot, seed)}
    for col in MATCHED:
        res["matched_cost"][col] = {
            "mean": float(df[col].mean()),
            "top5_minus": paired_bootstrap_test(df[f"top{PRIMARY_K}"].values, df[col].values, n_boot, seed)}
    return res


def latency_table(heldout: pd.DataFrame, latency_dir: Path, exp4b_latency: dict) -> dict:
    """Lower bound: hybrid path + k x the query's measured CE time per pair (50-pair rerank / 50)."""
    out = {}
    for ds in heldout["dataset"].unique():
        path = latency_dir / f"{ds}_components.csv"
        if not path.exists():
            continue
        comp = pd.read_csv(path, index_col=0).loc[heldout.index[heldout["dataset"] == ds]]
        hyb = comp[exp4b.HYBRID_MS].sum(axis=1)
        per_pair = comp["cross_encoder_rerank"] / FULL_K
        systems = {f"top{k}": hyb + k * per_pair for k in KS if k != FULL_K}
        out[ds] = {name: {"mean_ms": float(v.mean()), "p95_ms": float(v.quantile(0.95))} for name, v in systems.items()}
        for name in ("always_hybrid", "always_blended", "predicted_gain@0.1", "router_confidence@0.1"):
            out[ds][name] = exp4b_latency[ds][name]
    return out


fmt_ci = exp4b.fmt_ci


def write_tables(summary: dict, path: Path) -> None:
    pooled = summary["pooled"]
    prim = pooled["k"][str(PRIMARY_K)]
    rows = [*summary["per_dataset"].items(), ("**pooled**", pooled)]
    lines = [
        "# Experiment 4c results: blended reranking of every query's top-k (exploratory)", "",
        f"Design fixed before running in `{PREREG}` (not blind). Held-out nDCG@10, 5-fold CV within each dataset "
        "(Experiment 4/4b folds and α*); λ chosen per fold and per k on training queries. Top-50 = always-blended "
        "(Experiment 4b's backend).", "",
        f"**Prerequisite (pooled): always-blended − always-hybrid = {fmt_ci(summary['prerequisite'])} → "
        f"{'MET' if summary['prerequisite']['ci_95_low'] > 0 else 'NOT MET'}.**", "",
        f"**H3′ (pooled, k = {PRIMARY_K}, 5 CE pairs per query): top-{PRIMARY_K} − always-hybrid = "
        f"{fmt_ci(prim['minus_hybrid'])}, retained = {prim['retained']:.0%} → {summary['h3prime_verdict'].upper()}.**", "",
        "## Every query reranked at depth k: mean nDCG@10 (retained share of the always-blended gain)", "",
        "| Dataset | n | Always-hybrid | " + " | ".join(f"Top-{k}" for k in KS) + " | λ per fold (k = 5) |",
        "| :--- | :---: | :---: | " + " | ".join(":---:" for _ in KS) + " | :---: |"]
    for name, s in rows:
        lam = ", ".join(f"{f['lambda'][str(PRIMARY_K)]:g}" for f in summary["fold_info"].get(name, []))
        cells = []
        for k in KS:
            r = s["k"][str(k)]
            cells.append(f"{r['mean']:.4f}" + (f" ({r['retained']:.0%})" if r["retained"] is not None and k != FULL_K else ""))
        lines.append(f"| {name} | {s['n']} | {s['always_hybrid']:.4f} | " + " | ".join(cells) + f" | {lam} |")
    lines += ["", "## Top-k − always-hybrid [95% CI]", "",
              "| Dataset | " + " | ".join(f"Top-{k}" for k in KS) + " |",
              "| :--- | " + " | ".join(":---:" for _ in KS) + " |"]
    for name, s in rows:
        lines.append(f"| {name} | " + " | ".join(fmt_ci(s["k"][str(k)]["minus_hybrid"]) for k in KS) + " |")
    lines += ["", "## Matched cost: top-5 for every query vs Experiment 4b escalation of 10% of queries (5 CE pairs per query on average)", "",
              "| Dataset | Top-5 | Pred-gain @10% | Router-conf @10% | Random @10% | Top-5 − pred-gain | Top-5 − router-conf | Top-5 − random |",
              "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |"]
    for name, s in rows:
        mc = s["matched_cost"]
        lines.append(f"| {name} | {s['k'][str(PRIMARY_K)]['mean']:.4f} | " + " | ".join(f"{mc[c]['mean']:.4f}" for c in MATCHED)
                     + " | " + " | ".join(fmt_ci(mc[c]["top5_minus"]) for c in MATCHED) + " |")
    lines += ["", "## Latency (ms, mean / p95; top-k rows are lower bounds: hybrid + k × measured CE time per pair)", "",
              "| Dataset | Always-hybrid | Top-3 | Top-5 | Top-10 | Top-20 | Pred-gain @10% | Router-conf @10% | Always-blended |",
              "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |"]
    for ds, L in summary["latency"].items():
        cell = lambda key: f"{L[key]['mean_ms']:.1f} / {L[key]['p95_ms']:.1f}"  # noqa: E731
        lines.append(f"| {ds} | " + " | ".join(cell(key) for key in ("always_hybrid", "top3", "top5", "top10", "top20",
                                                                        "predicted_gain@0.1", "router_confidence@0.1",
                                                                        "always_blended")) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    args = parse_args()
    exp_cfg = load_config(args.config_dir / "experiment.yaml")
    ret_cfg = load_config(args.config_dir / "retrievers.yaml")
    hyb_cfg = load_config(args.config_dir / "hybrid.yaml")
    seed, k_folds = exp_cfg["seed"], exp_cfg["k_folds"]
    seed_everything(seed)
    alphas = [float(a) for a in hyb_cfg["alpha_grid"]]
    top_k = ret_cfg["dense"].get("top_k", 100)
    e4b = pd.read_csv(args.exp4b_dir / "per_query_heldout.csv", index_col=0)

    heldout, fold_log = [], {}
    for ds in args.datasets:
        df = pd.read_csv(args.exp2_dir / "cache" / f"{ds}.csv", index_col=0, dtype={"query_id": str})
        df.index = df.index.astype(str)
        path = args.processed_dir / "quora-sub" if ds == "quora" else args.data_dir / ds
        _, queries, qrels = exp1b.load_split(path)
        run_dir = args.results_dir / "runs" / ds
        bm25_run = exp1.read_trec_run(run_dir / f"bm25_k1-{ret_cfg['bm25']['k1']}_b-{ret_cfg['bm25']['b']}.trec")
        dense_run = exp1.read_trec_run(run_dir / f"dense_{args.dense_model.replace('/', '__')}.trec")
        for q in queries:
            bm25_run.setdefault(q, [])
            dense_run.setdefault(q, [])
        used = sorted(set(exp4.fold_alphas(df, alphas, k_folds, seed)))
        hyb = exp1.build_hybrid_runs(bm25_run, dense_run, used, top_k, hyb_cfg["normalization"]["method"])
        ce = exp2e.load_ce_cache(ds, args.results_dir)
        missing = sum(c.doc_id not in ce.get(q, {}) for a in used for q in queries for c in hyb[a][q][:FULL_K])
        if missing:
            raise RuntimeError(f"[{ds}] {missing} top-{FULL_K} pairs are not in the cross-encoder cache")

        per_alpha = {}
        for a in used:
            rows = {}
            for q in queries:
                cands = hyb[a][q]
                row = {"ndcg_hyb": ndcg_at_k([c.doc_id for c in cands], qrels[q], 10)}
                for k in KS:
                    for lam in LAMBDAS:
                        row[f"k{k}_{lam}"] = ndcg_at_k(topk_blended_ranking(cands, ce.get(q, {}), lam, k), qrels[q], 10)
                rows[q] = row
            per_alpha[a] = pd.DataFrame.from_dict(rows, orient="index").loc[df.index]

        fold_log[ds] = []
        for i, (tr, te) in enumerate(create_kfold_splits(list(df.index), k_folds, seed)):
            alpha = max(alphas, key=lambda a: df.loc[tr, f"ndcg_hyb_{a}"].mean())
            pa = per_alpha[alpha]
            out = pd.DataFrame(index=te)
            out["dataset"], out["fold"], out["hybrid"] = ds, i, pa.loc[te, "ndcg_hyb"]
            lams = {}
            for k in KS:
                lams[str(k)] = max(LAMBDAS, key=lambda l: pa.loc[tr, f"k{k}_{l}"].mean())
                out[f"top{k}"] = pa.loc[te, f"k{k}_{lams[str(k)]}"]
            heldout.append(out)
            fold_log[ds].append({"alpha": alpha, "lambda": lams})
        log.info(f"[{ds}] λ (k=5) per fold: " + ", ".join(f"{f['lambda']['5']:g}" for f in fold_log[ds]))
    heldout = pd.concat(heldout)
    heldout.index = [f"{d}:{q}" for d, q in zip(heldout["dataset"], heldout.index)]

    # The 4b backend must be reproduced exactly before 4b's escalation results are used as the matched-cost comparators.
    drift = np.abs(heldout[f"top{FULL_K}"].values - e4b.loc[heldout.index, "blended"].values).max()
    if drift > 1e-9:
        raise RuntimeError(f"always-blended differs from Experiment 4b (max |Δ| = {drift:.2e})")
    heldout = heldout.join(e4b[MATCHED])

    pooled = block(heldout, args.n_bootstrap, seed)
    prereq = paired_bootstrap_test(heldout[f"top{FULL_K}"].values, heldout["hybrid"].values, args.n_bootstrap, seed)
    prim = pooled["k"][str(PRIMARY_K)]
    if prereq["ci_95_low"] <= 0:
        verdict = "not testable"
    else:
        verdict = "supported" if prim["minus_hybrid"]["ci_95_low"] > 0 and prim["retained"] >= MIN_RETAINED else "not supported"
    exp4b_latency = json.loads((args.exp4b_dir / "summary.json").read_text(encoding="utf-8"))["latency"]
    summary = {
        "preregistration": PREREG,
        "status": "exploratory (not blind)",
        "prerequisite": prereq,
        "h3prime_verdict": verdict,
        "pooled": pooled,
        "per_dataset": {ds: block(heldout[heldout["dataset"] == ds], args.n_bootstrap, seed) for ds in fold_log},
        "latency": latency_table(heldout, args.latency_dir, exp4b_latency),
        "fold_info": fold_log,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    heldout.to_csv(args.output_dir / "per_query_heldout.csv")
    with open(args.output_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=float)
    write_tables(summary, args.output_dir / "tables.md")
    log.info(f"H3′ (pooled, top-{PRIMARY_K}): {fmt_ci(prim['minus_hybrid'])}, retained {prim['retained']:.0%} "
             f"→ {verdict.upper()}")
    log.info(f"Wrote {args.output_dir / 'tables.md'}")


if __name__ == "__main__":
    main()
