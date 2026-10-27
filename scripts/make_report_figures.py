"""Collect and draw the figures referenced by report/report.md into report/figures/.

Draws two summary figures from saved artifacts (no recomputation):
  * comparisons_forest.png  -- pooled paired differences vs. always-hybrid / matched random, with 95% CIs;
  * reranker_benchmark.png  -- per-dataset reranker − hybrid nDCG@10 with 95% CIs;
and copies the per-experiment figures (Experiment 4 escalation curve, risk-coverage / tau escalation,
quality-vs-latency) when they exist.

Usage:
  python scripts/make_report_figures.py
"""
import argparse
import importlib.util
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _load_script(name: str, file: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / file)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


exp1 = _load_script("exp1", "run_experiment1_hybrid_degradation.py")

COPIES = {
    "experiment4/figures/escalation_curve.png": "exp4_escalation_curve.png",
    "tau_escalation/figures/risk_coverage_and_tau.png": "risk_coverage_and_tau.png",
    "policy_latency/figures/quality_vs_latency.png": "quality_vs_latency.png",
    "experiment3/figures/cost_quality_curve.png": "exp3_cost_quality_curve.png",
}


def _json(path: Path):
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def forest(ax, rows, title):
    """rows: (label, delta, lo, hi) drawn top to bottom."""
    for i, (label, d, lo, hi) in enumerate(rows):
        y = len(rows) - 1 - i
        color = "#2a78d6" if lo > 0 else ("#d43d51" if hi < 0 else exp1.INK_2)
        ax.plot([lo, hi], [y, y], color=color, linewidth=2)
        ax.scatter([d], [y], color=color, zorder=3, s=22)
    ax.axvline(0, color=exp1.INK, linewidth=0.8)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([r[0] for r in reversed(rows)])
    ax.set(xlabel="Δ nDCG@10 (paired bootstrap 95% CI)", title=title)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--results_dir", type=Path, default=Path("results"))
    p.add_argument("--output_dir", type=Path, default=Path("report/figures"))
    args = p.parse_args()
    r, out = args.results_dir, args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    plt = exp1._style()

    def t(test):
        return test["delta_mean"], test["ci_95_low"], test["ci_95_high"]

    rows = []
    s2, s3, s4 = (_json(r / f"experiment{i}" / "summary.json") for i in (2, 3, 4))
    tau = _json(r / "tau_escalation" / "summary.json")
    if s2:
        rows.append(("Exp 2: GBM router − hybrid", *t(s2["pooled"]["tests"]["router_gbm - hybrid"])))
        rows.append(("Exp 2: GBM router − random routing", *t(s2["pooled"]["tests"]["router_gbm - random_gbm"])))
    if s3:
        scope = "+".join(s3["per_dataset"])
        rows.append((f"Exp 3 ({scope}): skip-dense − hybrid", *t(s3["pooled"]["router_minus_hybrid"])))
        rows.append((f"Exp 3 ({scope}): skip-dense − random skip", *t(s3["pooled"]["router_minus_random"])))
    if s4:
        rows.append(("Exp 4: pred-gain − random @10%", *t(s4["pooled"]["primary_pg_minus_random"])))
        rows.append(("Exp 4: router-conf − random @10%", *t(s4["pooled"]["rc_minus_random"])))
    if tau:
        b = tau["pooled"]["rates"]["0.1"]
        rows.append(("Router + τ esc. (10%) − hybrid", *t(b["router_tau_minus_hybrid"])))
        rows.append(("Router + τ esc. − random esc.", *t(b["router_tau_minus_router_random"])))
    if rows:
        fig, ax = plt.subplots(figsize=(8.4, 0.42 * len(rows) + 1.2))
        forest(ax, rows, "Pooled held-out comparisons (blue: CI > 0, red: CI < 0)")
        fig.tight_layout()
        fig.savefig(out / "comparisons_forest.png", dpi=200)
        plt.close(fig)

    bench = _json(r / "reranker_benchmark" / "summary.json")
    if bench:
        rows = []
        for ds, res in bench["results"].items():
            for model, m in res["models"].items():
                rows.append((f"{ds}: {model.split('/')[-1]}", *t(m["reranker_minus_hybrid"])))
        fig, ax = plt.subplots(figsize=(7.4, 0.42 * len(rows) + 1.2))
        forest(ax, rows, "Reranker − hybrid on matched top-50 pools")
        fig.tight_layout()
        fig.savefig(out / "reranker_benchmark.png", dpi=200)
        plt.close(fig)

    for src, dst in COPIES.items():
        if (r / src).exists():
            shutil.copyfile(r / src, out / dst)
    print("\n".join(sorted(p.name for p in out.glob("*.png"))))


if __name__ == "__main__":
    main()
