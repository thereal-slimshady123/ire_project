"""End-to-end evaluation of the proposal's system: route among BM25/dense/hybrid, escalate when max P < tau.

Exploratory (not pre-registered). Experiment 4 escalates a fixed share of each held-out fold ranked by
confidence; this script instead applies the proposal's thresholding rule with tau calibrated on the
training folds only:

  * per dataset and fold, the GBM router is refitted with Experiment 2's training-fold alpha* and epsilon
    (held-out routes must reproduce Experiment 2's `route_gbm`; checked below);
  * tau is chosen with EscalationDecider.find_tau_for_target_rate on out-of-fold training probabilities
    (inner 3-fold CV), for each target escalation rate in configs/escalation.yaml;
  * a non-escalated query scores the nDCG@10 of its routed list; an escalated one scores Experiment 4's
    MiniLM-L6 rerank of the fold's hybrid top-50;
  * controls: always-hybrid, router without escalation, random escalation at the realized held-out rate
    (expected value), and confidence-gated escalation on top of always-hybrid;
  * risk-coverage curves of the router's cheap decision (risk = oracle - routed nDCG@10).

Usage:
  python scripts/run_tau_escalation.py
"""

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from router_retrieval.data.splits import create_kfold_splits  # noqa: E402
from router_retrieval.evaluation.risk_coverage import compute_risk_coverage_curve  # noqa: E402
from router_retrieval.evaluation.significance import paired_bootstrap_test  # noqa: E402
from router_retrieval.router.escalation import EscalationDecider  # noqa: E402
from router_retrieval.utils.config import load_config  # noqa: E402
from router_retrieval.utils.logging_utils import get_logger  # noqa: E402
from router_retrieval.utils.seeding import seed_everything  # noqa: E402


def _load_script(name: str, file: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / file)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


exp1 = _load_script("exp1", "run_experiment1_hybrid_degradation.py")
exp2 = _load_script("exp2", "run_experiment2_heldout_routing.py")

log = get_logger("tau_escalation")
ROUTES = exp2.ROUTES
INNER_FOLDS = 3
PRIMARY_RATE = 0.10


def parse_args():
    p = argparse.ArgumentParser(description="Router + confidence-threshold escalation, end to end")
    p.add_argument("--datasets", nargs="+", default=exp2.DATASETS)
    p.add_argument("--config_dir", type=Path, default=Path("configs"))
    p.add_argument("--exp2_dir", type=Path, default=Path("results/experiment2"))
    p.add_argument("--exp4_dir", type=Path, default=Path("results/experiment4"))
    p.add_argument("--output_dir", type=Path, default=Path("results/tau_escalation"))
    p.add_argument("--n_bootstrap", type=int, default=10000)
    p.add_argument("--no_plots", action="store_true")
    return p.parse_args()


def router_proba(model, X: pd.DataFrame) -> np.ndarray:
    """Class probabilities ordered as ROUTES; a constant (degenerate-fold) router is fully confident."""
    if getattr(model, "constant", None) is not None:
        out = np.zeros((len(X), len(ROUTES)))
        out[:, ROUTES.index(model.constant)] = 1.0
        return out
    return model.predict_proba(X)


def oof_train_proba(train: pd.DataFrame, alpha: float, eps: float, router_cfg: dict, seed: int) -> np.ndarray:
    """Out-of-fold router probabilities on the training queries (inner k-fold), for calibrating tau."""
    R = exp2.route_matrix(train, alpha)
    out = np.zeros((len(train), len(ROUTES)))
    pos = {q: i for i, q in enumerate(train.index)}
    for itr, ite in create_kfold_splits(list(train.index), INNER_FOLDS, seed):
        m = exp2.fit_router("gbm", train.loc[itr, exp2.FEATURES], R.loc[itr], eps, router_cfg, seed)
        out[[pos[q] for q in ite]] = router_proba(m, train.loc[ite, exp2.FEATURES])
    return out


def apply_tau_policy(routed, hybrid, reranked, escalated) -> dict:
    """Per-query nDCG@10 of router+tau, hybrid+tau and their random-escalation controls at the same rate."""
    esc = np.asarray(escalated, dtype=bool)
    rate = float(esc.mean()) if len(esc) else 0.0
    return {
        "router_tau": np.where(esc, reranked, routed),
        "hybrid_tau": np.where(esc, reranked, hybrid),
        "router_random": (1 - rate) * np.asarray(routed) + rate * np.asarray(reranked),
        "hybrid_random": (1 - rate) * np.asarray(hybrid) + rate * np.asarray(reranked),
    }


def aurc(conf: np.ndarray, risk: np.ndarray) -> float:
    """Area under the risk-coverage curve (lower = confidence ranks risky queries last)."""
    order = np.argsort(-conf, kind="stable")
    cum = np.cumsum(risk[order]) / np.arange(1, len(risk) + 1)
    return float(cum.mean()) if len(cum) else 0.0


def main():
    args = parse_args()
    exp_cfg = load_config(args.config_dir / "experiment.yaml")
    router_cfg = load_config(args.config_dir / "router.yaml")
    esc_cfg = load_config(args.config_dir / "escalation.yaml")
    seed, k_folds = exp_cfg["seed"], exp_cfg["k_folds"]
    seed_everything(seed)
    rates = [float(r) for r in esc_cfg["target_escalation_rates"]]
    taus_grid = [float(t) for t in esc_cfg["tau_grid"]]

    exp2_summary = json.loads((args.exp2_dir / "summary.json").read_text(encoding="utf-8"))
    exp2_held = pd.read_csv(args.exp2_dir / "per_query_heldout.csv", index_col=0)
    exp4_held = pd.read_csv(args.exp4_dir / "per_query_heldout.csv", index_col=0)

    rows, fold_log = [], {}
    for ds in args.datasets:
        df = pd.read_csv(args.exp2_dir / "cache" / f"{ds}.csv", index_col=0, dtype={"query_id": str})
        df.index = df.index.astype(str)
        fold_log[ds] = []
        for i, (tr, te) in enumerate(create_kfold_splits(list(df.index), k_folds, seed)):
            info = exp2_summary["fold_info"][ds][i]
            alpha, eps = float(info["alpha"]), float(info["epsilon"])
            R_te = exp2.route_matrix(df.loc[te], alpha)
            model = exp2.fit_router("gbm", df.loc[tr, exp2.FEATURES], exp2.route_matrix(df.loc[tr], alpha),
                                    eps, router_cfg, seed)
            proba = router_proba(model, df.loc[te, exp2.FEATURES])
            oof = oof_train_proba(df.loc[tr], alpha, eps, router_cfg, seed)
            routes = np.array([ROUTES[j] for j in proba.argmax(axis=1)], dtype=object)

            keys = [f"{ds}:{q}" for q in te]
            out = pd.DataFrame(index=keys)
            out["dataset"], out["fold"] = ds, i
            out["hybrid"] = R_te["hybrid"].values
            out["routed"] = exp2.routed_scores(R_te, routes)
            out["route"] = routes
            out["oracle"] = R_te.max(axis=1).values
            out["regret"] = out["oracle"] - out["routed"]
            out["confidence"] = proba.max(axis=1)
            out["reranked"] = exp4_held.loc[keys, "reranked"].values
            out["route_matches_exp2"] = exp2_held.loc[keys, "route_gbm"].values == routes

            fold_rates = {}
            for rate in rates:
                decider = EscalationDecider(classes=ROUTES)
                decider.tau = decider.find_tau_for_target_rate(oof, rate)
                _, escalated, _ = decider.decide(proba)
                out[f"escalated@{rate}"] = escalated
                for name, vals in apply_tau_policy(out["routed"], out["hybrid"], out["reranked"], escalated).items():
                    out[f"{name}@{rate}"] = vals
                fold_rates[str(rate)] = {"tau": float(decider.tau), "train_oof_rate": float((oof.max(axis=1) < decider.tau).mean()),
                                         "heldout_rate": float(np.mean(escalated))}
            rows.append(out)
            fold_log[ds].append({"alpha": alpha, "epsilon": eps, "rates": fold_rates})
        match = pd.concat(rows)
        match = match[match["dataset"] == ds]["route_matches_exp2"].mean()
        log.info(f"[{ds}] held-out routes reproduce Experiment 2 on {match:.1%} of queries")

    held = pd.concat(rows)

    def block(frame: pd.DataFrame) -> dict:
        res = {"n": len(frame),
               "mean": {c: float(frame[c].mean()) for c in ("hybrid", "routed", "reranked", "oracle")},
               "route_match_exp2": float(frame["route_matches_exp2"].mean()),
               "spearman_conf_regret": float(stats.spearmanr(frame["confidence"], frame["regret"]).statistic),
               "spearman_conf_rerank_gain": float(stats.spearmanr(frame["confidence"],
                                                                  frame["reranked"] - frame["routed"]).statistic),
               "aurc": aurc(frame["confidence"].values, frame["regret"].values),
               "aurc_random": float(frame["regret"].mean()),
               "rates": {}}
        for rate in rates:
            r = str(rate)
            res["rates"][r] = {
                "heldout_rate": float(frame[f"escalated@{rate}"].mean()),
                "mean": {name: float(frame[f"{name}@{rate}"].mean())
                         for name in ("router_tau", "hybrid_tau", "router_random", "hybrid_random")},
                "router_tau_minus_hybrid": paired_bootstrap_test(
                    frame[f"router_tau@{rate}"].values, frame["hybrid"].values, args.n_bootstrap, seed),
                "router_tau_minus_router_random": paired_bootstrap_test(
                    frame[f"router_tau@{rate}"].values, frame[f"router_random@{rate}"].values, args.n_bootstrap, seed),
                "hybrid_tau_minus_hybrid_random": paired_bootstrap_test(
                    frame[f"hybrid_tau@{rate}"].values, frame[f"hybrid_random@{rate}"].values, args.n_bootstrap, seed),
            }
        return res

    datasets = list(dict.fromkeys(held["dataset"]))
    summary = {"exploratory": True, "escalation_backend": "Experiment 4 MiniLM-L6 rerank of hybrid top-50",
               "pooled": block(held), "per_dataset": {ds: block(held[held["dataset"] == ds]) for ds in datasets},
               "fold_info": fold_log}

    curves = []
    for name, frame in [("pooled", held), *((ds, held[held["dataset"] == ds]) for ds in datasets)]:
        c = compute_risk_coverage_curve(frame["confidence"].values, frame["regret"].values, taus_grid)
        c.insert(0, "scope", name)
        curves.append(c)
    curves = pd.concat(curves, ignore_index=True)

    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    held.to_csv(out_dir / "per_query_heldout.csv")
    curves.to_csv(out_dir / "risk_coverage.csv", index=False)
    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=float)

    if not args.no_plots:
        plt = exp1._style()
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
        conf, risk = held["confidence"].values, held["regret"].values
        order = np.argsort(-conf, kind="stable")
        coverage = np.arange(1, len(risk) + 1) / len(risk)
        axes[0].plot(coverage, np.cumsum(risk[order]) / np.arange(1, len(risk) + 1), color="#2a78d6",
                     label=f"GBM router confidence (AURC {summary['pooled']['aurc']:.4f})")
        axes[0].axhline(risk.mean(), color=exp1.INK_2, linestyle=(0, (4, 3)), linewidth=1.4,
                        label=f"Random order ({risk.mean():.4f})")
        axes[0].xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
        axes[0].set(xlabel="Coverage (queries kept on the cheap route, most confident first)",
                    ylabel="Mean regret of covered queries (oracle − routed nDCG@10)",
                    title="Risk-coverage of the router's cheap decision (pooled)")
        axes[0].legend(loc="lower right")

        p = summary["pooled"]
        xs = [0.0] + [p["rates"][str(r)]["heldout_rate"] for r in rates]
        for key, label, style in (
            ("router_tau", "Router + τ escalation", dict(color="#2a78d6", marker="o", markersize=4)),
            ("router_random", "Router + random escalation", dict(color="#2a78d6", linestyle=(0, (4, 3)), linewidth=1.4)),
            ("hybrid_tau", "Hybrid + τ escalation", dict(color="#1baf7a", marker="o", markersize=4)),
            ("hybrid_random", "Hybrid + random escalation", dict(color="#1baf7a", linestyle=(0, (4, 3)), linewidth=1.4)),
        ):
            start = p["mean"]["routed"] if key.startswith("router") else p["mean"]["hybrid"]
            axes[1].plot(xs, [start] + [p["rates"][str(r)]["mean"][key] for r in rates], label=label, **style)
        axes[1].axhline(p["mean"]["hybrid"], color=exp1.INK, linewidth=0.8)
        axes[1].xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
        axes[1].set(xlabel="Realized held-out escalation rate (τ fit on training folds)",
                    ylabel="Mean held-out nDCG@10 (pooled)", title="Proposal system: router + τ escalation")
        axes[1].legend(loc="lower left")
        fig.tight_layout()
        (out_dir / "figures").mkdir(parents=True, exist_ok=True)
        fig.savefig(out_dir / "figures" / "risk_coverage_and_tau.png", dpi=200)
        plt.close(fig)

    ci = lambda t: f"{t['delta_mean']:+.4f} [{t['ci_95_low']:+.4f}, {t['ci_95_high']:+.4f}]"  # noqa: E731
    p = summary["pooled"]
    prim = p["rates"][str(PRIMARY_RATE)]
    lines = [
        "# Router + confidence-threshold (τ) escalation, end to end",
        "",
        "Exploratory, not pre-registered. τ is calibrated per fold on out-of-fold *training* router probabilities "
        "(inner 3-fold CV) to hit each target rate; held-out rates are what the threshold actually produces. "
        "Escalation = Experiment 4's MiniLM-L6 rerank of the hybrid top-50. Random controls escalate at the same "
        "realized rate (expected value). nDCG@10.",
        "",
        f"Held-out GBM routes reproduce Experiment 2 on {p['route_match_exp2']:.1%} of queries.",
        "",
        f"**Pooled, target {PRIMARY_RATE:.0%} (realized {prim['heldout_rate']:.1%}): router+τ − always-hybrid = "
        f"{ci(prim['router_tau_minus_hybrid'])}; router+τ − router+random = {ci(prim['router_tau_minus_router_random'])}.**",
        "",
        "| Dataset | n | Always-hybrid | Router (no esc.) | Target | Realized rate | Router+τ | Router+random | "
        "Router+τ − hybrid | Router+τ − random | Hybrid+τ − hybrid+random |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]
    for name, s in [*summary["per_dataset"].items(), ("**pooled**", p)]:
        for rate in rates:
            b = s["rates"][str(rate)]
            lines.append(
                f"| {name} | {s['n']} | {s['mean']['hybrid']:.4f} | {s['mean']['routed']:.4f} | {rate:.0%} "
                f"| {b['heldout_rate']:.1%} | {b['mean']['router_tau']:.4f} | {b['mean']['router_random']:.4f} "
                f"| {ci(b['router_tau_minus_hybrid'])} | {ci(b['router_tau_minus_router_random'])} "
                f"| {ci(b['hybrid_tau_minus_hybrid_random'])} |")
    lines += ["", "## Does router confidence identify risky queries?", "",
              "Risk = oracle − routed nDCG@10 of the router's cheap decision. AURC = mean risk over coverage "
              "levels when queries are kept most-confident-first; a random order gives the mean risk.", "",
              "| Dataset | AURC (confidence) | AURC (random) | Spearman(conf, regret) | Spearman(conf, rerank gain) |",
              "| :--- | :---: | :---: | :---: | :---: |"]
    for name, s in [*summary["per_dataset"].items(), ("**pooled**", p)]:
        lines.append(f"| {name} | {s['aurc']:.4f} | {s['aurc_random']:.4f} | {s['spearman_conf_regret']:+.3f} "
                     f"| {s['spearman_conf_rerank_gain']:+.3f} |")
    lines += ["", "## Risk-coverage at the configured τ grid (pooled, held-out confidences)", "",
              "| τ | Coverage | Escalation rate | Risk (mean regret of covered) |", "| :---: | :---: | :---: | :---: |"]
    for _, r in curves[curves["scope"] == "pooled"].iterrows():
        lines.append(f"| {r.tau:.2f} | {r.coverage:.1%} | {r.escalation_rate:.1%} | {r.risk:.4f} |")
    (out_dir / "tables.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    log.info(f"router+τ@{PRIMARY_RATE:.0%} − hybrid = {ci(prim['router_tau_minus_hybrid'])}; "
             f"− random = {ci(prim['router_tau_minus_router_random'])}")


if __name__ == "__main__":
    main()
