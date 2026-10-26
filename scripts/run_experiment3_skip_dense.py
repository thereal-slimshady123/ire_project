"""Experiment 3: can we skip the dense retriever on queries where BM25 alone is enough?

Implements the pre-registered protocol in improvements.md §4.1:
  * per query, answer from BM25 only or run hybrid; features use only what exists before dense runs;
  * label "BM25 suffices" = nDCG@10(BM25) >= nDCG@10(hybrid at training alpha*) - 0.02;
  * LightGBM classifier; skip threshold tau chosen on training folds (inner 3-fold OOF) as the most
    aggressive tau with training loss >= -0.005; heuristics (BM25 top1/top2 ratio, max-IDF) tuned the same way;
  * success (pooled): router - hybrid CI low >= -0.01, skip rate >= 20%, router - random(same rate) CI low > 0.

Usage:
  python scripts/run_experiment3_skip_dense.py
"""

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from typing import Callable, Dict, List, Tuple

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from router_retrieval.data.splits import create_kfold_splits  # noqa: E402
from router_retrieval.evaluation.significance import paired_bootstrap_test  # noqa: E402
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

log = get_logger("experiment3")
LABEL_MARGIN = 0.02
TRAIN_LOSS_TOLERANCE = -0.005
NONINFERIORITY_MARGIN = -0.01
MIN_SKIP_RATE = 0.20
INNER_FOLDS = 3
SKIP_FEATURES = [
    "query_length", "max_idf", "mean_idf", "rare_term_density", "entity_density", "oov_fraction",
    "bm25_top1_score", "bm25_top1_top2_gap", "bm25_top1_top2_ratio", "bm25_score_entropy",
    "bm25_top10_query_term_coverage", "bm25_num_hits",
]
HEURISTICS = {"H1_bm25_top_ratio": "bm25_top1_top2_ratio", "H2_max_idf": "max_idf"}


def parse_args():
    p = argparse.ArgumentParser(description="Experiment 3: skip dense retrieval when BM25 suffices")
    p.add_argument("--datasets", nargs="+", default=exp2.DATASETS)
    p.add_argument("--config_dir", type=Path, default=Path("configs"))
    p.add_argument("--results_dir", type=Path, default=Path("results"))
    p.add_argument("--exp2_dir", type=Path, default=Path("results/experiment2"))
    p.add_argument("--output_dir", type=Path, default=Path("results/experiment3"))
    p.add_argument("--dense_model", type=str, default="intfloat/e5-small-v2")
    p.add_argument("--n_bootstrap", type=int, default=10000)
    p.add_argument("--no_plots", action="store_true")
    return p.parse_args()


def load_table(ds: str, args) -> pd.DataFrame:
    """Experiment 2's cached per-query table plus the number of BM25 hits (from the cached run)."""
    df = pd.read_csv(args.exp2_dir / "cache" / f"{ds}.csv", index_col=0, dtype={"query_id": str})
    df.index = df.index.astype(str)
    run = exp1.read_trec_run(args.results_dir / "runs" / ds / "bm25_k1-1.2_b-0.75.trec")
    df["bm25_num_hits"] = [len(run.get(q, [])) for q in df.index]
    return df


def choose_threshold(score: np.ndarray, bm25: np.ndarray, hybrid: np.ndarray, tolerance: float) -> float:
    """
    Most aggressive threshold t ("skip iff score >= t") whose mean loss vs. always-hybrid is >= tolerance.

    Only cut points between distinct score values are considered, so ties are never split.
    Returns +inf when even the first cut violates the tolerance (never skip).
    """
    order = np.argsort(-score, kind="stable")
    s, delta = score[order], (bm25 - hybrid)[order]
    loss = np.cumsum(delta) / len(score)  # loss after skipping the first k+1 queries
    valid_cut = np.append(s[1:] != s[:-1], True)  # can cut after position k
    ok = np.flatnonzero(valid_cut & (loss >= tolerance))
    return float(s[ok.max()]) if ok.size else float("inf")


def fit_classifier(X: pd.DataFrame, y: np.ndarray, router_cfg: dict, seed: int):
    import lightgbm as lgb

    g = router_cfg["models"]["gbm"]
    if len(np.unique(y)) < 2:
        return ("constant", float(y[0]))
    model = lgb.LGBMClassifier(n_estimators=g["n_estimators"], learning_rate=g["learning_rate"],
                               num_leaves=g["num_leaves"], class_weight=g["class_weight"], random_state=seed,
                               deterministic=True, force_row_wise=True, verbose=-1)
    return model.fit(X, y)


def predict_p(model, X: pd.DataFrame) -> np.ndarray:
    if isinstance(model, tuple):
        return np.full(len(X), model[1])
    return model.predict_proba(X)[:, list(model.classes_).index(1)]


def oof_probabilities(train: pd.DataFrame, y: np.ndarray, router_cfg: dict, seed: int) -> np.ndarray:
    oof = np.zeros(len(train))
    pos = {q: i for i, q in enumerate(train.index)}
    for itr, ite in create_kfold_splits(list(train.index), INNER_FOLDS, seed):
        tr_idx, te_idx = [pos[q] for q in itr], [pos[q] for q in ite]
        m = fit_classifier(train.loc[itr, SKIP_FEATURES], y[tr_idx], router_cfg, seed)
        oof[te_idx] = predict_p(m, train.loc[ite, SKIP_FEATURES])
    return oof


def evaluate_fold(train: pd.DataFrame, test: pd.DataFrame, alphas, router_cfg, seed) -> Tuple[pd.DataFrame, dict]:
    alpha = max(alphas, key=lambda a: train[f"ndcg_hyb_{a}"].mean())
    H_tr, B_tr = train[f"ndcg_hyb_{alpha}"].values, train["ndcg_bm25"].values
    H_te, B_te = test[f"ndcg_hyb_{alpha}"].values, test["ndcg_bm25"].values
    y_tr = (B_tr >= H_tr - LABEL_MARGIN).astype(int)

    out = pd.DataFrame(index=test.index)
    out["dataset"] = test["dataset"].values
    out["hybrid"], out["bm25"], out["oracle"] = H_te, B_te, np.maximum(B_te, H_te)
    info = {"alpha": alpha}

    # Router: tau from inner-CV out-of-fold probabilities on training queries only.
    oof = oof_probabilities(train, y_tr, router_cfg, seed)
    tau = choose_threshold(oof, B_tr, H_tr, TRAIN_LOSS_TOLERANCE)
    model = fit_classifier(train[SKIP_FEATURES], y_tr, router_cfg, seed)
    p_te = predict_p(model, test[SKIP_FEATURES])
    scores = {"router": p_te}
    skip = p_te >= tau
    out["skip_router"] = skip
    out["router"] = np.where(skip, B_te, H_te)
    rate = skip.mean()
    out["random"] = rate * B_te + (1 - rate) * H_te  # expected value of random skipping at the same rate
    info.update(tau_router=tau, skip_rate_router=float(rate), label_pos_rate_train=float(y_tr.mean()))

    # Heuristics: direction and threshold chosen on training queries by the same rule.
    for name, feat in HEURISTICS.items():
        best = None
        for sign in (1.0, -1.0):
            s_tr = sign * train[feat].values
            t = choose_threshold(s_tr, B_tr, H_tr, TRAIN_LOSS_TOLERANCE)
            train_rate = float((s_tr >= t).mean())
            if best is None or train_rate > best[2]:
                best = (sign, t, train_rate)
        sign, t, _ = best
        s_te = sign * test[feat].values
        hskip = s_te >= t
        out[f"skip_{name}"] = hskip
        out[name] = np.where(hskip, B_te, H_te)
        out[f"random_{name}"] = hskip.mean() * B_te + (1 - hskip.mean()) * H_te
        scores[name] = s_te
        info[f"skip_rate_{name}"] = float(hskip.mean())
        info[f"direction_{name}"] = "high->skip" if sign > 0 else "low->skip"

    # Curves: skip the top-r share of held-out queries by each method's score.
    rates = np.round(np.arange(0, 1.0001, 0.05), 2)
    curve = {}
    for name, s in [*scores.items(), ("oracle", B_te - H_te)]:
        order = np.argsort(-s, kind="stable")
        pts = []
        for r in rates:
            k = int(round(r * len(s)))
            routed = H_te.copy()
            routed[order[:k]] = B_te[order[:k]]
            pts.append(float(routed.sum()))
        curve[name] = pts
    curve["random"] = [float((r * B_te + (1 - r) * H_te).sum()) for r in rates]
    info["curve_sums"], info["curve_n"], info["curve_rates"] = curve, len(test), rates.tolist()
    return out, info


def make_curve_figure(curves: Dict[str, np.ndarray], rates: np.ndarray, hybrid_mean: float, path: Path) -> None:
    plt = exp1._style()
    fig, ax = plt.subplots(figsize=(6.6, 4.2))
    dense_rate = 1 - rates
    ax.plot(dense_rate, curves["oracle"], color=exp1.INK_2, linestyle=(0, (1, 2)), linewidth=1.4, label="Oracle skipping (bound)")
    ax.plot(dense_rate, curves["random"], color=exp1.INK_2, linestyle=(0, (4, 3)), linewidth=1.4, label="Random skipping")
    styles = [("router", "#2a78d6", "LightGBM router"), ("H1_bm25_top_ratio", "#eb6834", "H1: BM25 top-1/top-2 ratio"),
              ("H2_max_idf", "#1baf7a", "H2: max-IDF")]
    for key, color, label in styles:
        ax.plot(dense_rate, curves[key], color=color, label=label)
    ax.axhline(hybrid_mean + NONINFERIORITY_MARGIN, color=exp1.GRID, linewidth=1)
    ax.text(0.42, hybrid_mean + NONINFERIORITY_MARGIN, "non-inferiority margin (always-hybrid − 0.01)",
            color=exp1.INK_2, fontsize=7.5, va="bottom")
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax.set(xlabel="Share of queries that still run the dense retriever", ylabel="Mean held-out nDCG@10 (pooled)",
           title="Experiment 3: quality vs. dense calls")
    ax.legend(loc="lower right")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200)
    plt.close(fig)


def main():
    args = parse_args()
    exp_cfg = load_config(args.config_dir / "experiment.yaml")
    router_cfg = load_config(args.config_dir / "router.yaml")
    alphas = [float(a) for a in load_config(args.config_dir / "hybrid.yaml")["alpha_grid"]]
    seed, k_folds = exp_cfg["seed"], exp_cfg["k_folds"]
    seed_everything(seed)
    tables = {ds: load_table(ds, args) for ds in args.datasets}

    heldout, fold_info = [], {}
    for ds, df in tables.items():
        fold_info[ds] = []
        for tr, te in create_kfold_splits(list(df.index), k_folds, seed):
            out, info = evaluate_fold(df.loc[tr], df.loc[te], alphas, router_cfg, seed)
            heldout.append(out)
            fold_info[ds].append(info)
        log.info(f"[{ds}] skip rates per fold: " + ", ".join(f"{f['skip_rate_router']:.0%}" for f in fold_info[ds]))
    heldout = pd.concat(heldout)
    heldout.index = [f"{d}:{q}" for d, q in zip(heldout["dataset"], heldout.index)]

    # Leave-one-dataset-out transfer (secondary).
    transfer = {}
    if len(tables) > 1:
        for target in tables:
            source = pd.concat([t for n, t in tables.items() if n != target])
            source.index = [f"{d}:{q}" for d, q in zip(source["dataset"], source.index)]
            out, info = evaluate_fold(source, tables[target], alphas, router_cfg, seed)
            transfer[target] = {
                "n": len(out), "skip_rate": float(out["skip_router"].mean()),
                "router_minus_hybrid": paired_bootstrap_test(out["router"].values, out["hybrid"].values, args.n_bootstrap, seed),
                "router_minus_random": paired_bootstrap_test(out["router"].values, out["random"].values, args.n_bootstrap, seed),
            }

    def block(df):
        b = lambda a, c: paired_bootstrap_test(df[a].values, df[c].values, args.n_bootstrap, seed)  # noqa: E731
        res = {
            "n": len(df),
            "mean": {c: float(df[c].mean()) for c in ["hybrid", "bm25", "router", "random", "oracle", *HEURISTICS]},
            "skip_rate": {"router": float(df["skip_router"].mean()),
                          **{h: float(df[f"skip_{h}"].mean()) for h in HEURISTICS},
                          "oracle": float((df["bm25"] >= df["hybrid"]).mean())},
            "router_minus_hybrid": b("router", "hybrid"),
            "router_minus_random": b("router", "random"),
        }
        for h in HEURISTICS:
            res[f"{h}_minus_hybrid"] = b(h, "hybrid")
            res[f"{h}_minus_random"] = b(h, f"random_{h}")
        return res

    pooled = block(heldout)
    verdict = {
        "noninferior": pooled["router_minus_hybrid"]["ci_95_low"] >= NONINFERIORITY_MARGIN,
        "savings": pooled["skip_rate"]["router"] >= MIN_SKIP_RATE,
        "beats_random": pooled["router_minus_random"]["ci_95_low"] > 0,
    }
    verdict["success"] = all(verdict.values())

    # Pooled curves (sum over all folds / n).
    rates = np.array(next(iter(fold_info.values()))[0]["curve_rates"])
    total_n = sum(f["curve_n"] for fs in fold_info.values() for f in fs)
    curves = {
        k: np.sum([f["curve_sums"][k] for fs in fold_info.values() for f in fs], axis=0) / total_n
        for k in ["router", *HEURISTICS, "random", "oracle"]
    }

    summary = {"preregistration": "improvements.md §4.1", "pooled": pooled, "verdict": verdict,
               "per_dataset": {ds: block(heldout[heldout["dataset"] == ds]) for ds in tables},
               "transfer": transfer, "pooled_curves": {k: v.tolist() for k, v in curves.items()},
               "curve_skip_rates": rates.tolist(),
               "fold_info": {ds: [{k: v for k, v in f.items() if not k.startswith("curve")} for f in fs]
                             for ds, fs in fold_info.items()}}
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    heldout.to_csv(out / "per_query_heldout.csv")
    with open(out / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=lambda o: bool(o) if isinstance(o, np.bool_) else float(o))
    if not args.no_plots:
        make_curve_figure(curves, rates, pooled["mean"]["hybrid"], out / "figures" / "cost_quality_curve.png")

    ci = lambda t: f"{t['delta_mean']:+.4f} [{t['ci_95_low']:+.4f}, {t['ci_95_high']:+.4f}]"  # noqa: E731
    yn = lambda b: "PASS" if b else "FAIL"  # noqa: E731
    lines = [
        "# Experiment 3 results: skipping the dense retriever",
        "",
        "Pre-registered in improvements.md §4.1. Held-out nDCG@10 (5-fold CV per dataset); thresholds fitted on training folds.",
        "",
        f"**Verdict: {'SUCCESS' if verdict['success'] else 'NEGATIVE'}** — "
        f"non-inferior (CI low ≥ −0.01): {yn(verdict['noninferior'])}; "
        f"skip rate ≥ 20%: {yn(verdict['savings'])} ({pooled['skip_rate']['router']:.1%}); "
        f"beats random skipping: {yn(verdict['beats_random'])}.",
        "",
        "| Dataset | n | Always-hybrid | Always-BM25 | Router | Router skip rate | Router − hybrid | Router − random (same rate) "
        "| H1 skip / − random | H2 skip / − random | Oracle skip rate |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]
    for name, s in [*summary["per_dataset"].items(), ("**pooled**", pooled)]:
        m, sr = s["mean"], s["skip_rate"]
        lines.append(
            f"| {name} | {s['n']} | {m['hybrid']:.4f} | {m['bm25']:.4f} | {m['router']:.4f} | {sr['router']:.1%} "
            f"| {ci(s['router_minus_hybrid'])} | {ci(s['router_minus_random'])} "
            f"| {sr['H1_bm25_top_ratio']:.1%} / {ci(s['H1_bm25_top_ratio_minus_random'])} "
            f"| {sr['H2_max_idf']:.1%} / {ci(s['H2_max_idf_minus_random'])} | {sr['oracle']:.1%} |"
        )
    lines += ["", "## Transfer (train on the other four datasets)", "",
              "| Target | Skip rate | Router − hybrid | Router − random |", "| :--- | :---: | :---: | :---: |"]
    for name, t in transfer.items():
        lines.append(f"| {name} | {t['skip_rate']:.1%} | {ci(t['router_minus_hybrid'])} | {ci(t['router_minus_random'])} |")
    lines += ["", "## Pooled curve: mean nDCG@10 at matched skip rates", "",
              "| Skip rate | Router | H1 | H2 | Random | Oracle |", "| :---: | :---: | :---: | :---: | :---: | :---: |"]
    for i, r in enumerate(rates):
        if round(r * 100) % 10 == 0:
            lines.append(f"| {r:.0%} | {curves['router'][i]:.4f} | {curves['H1_bm25_top_ratio'][i]:.4f} "
                         f"| {curves['H2_max_idf'][i]:.4f} | {curves['random'][i]:.4f} | {curves['oracle'][i]:.4f} |")
    (out / "tables.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    log.info(f"VERDICT {'SUCCESS' if verdict['success'] else 'NEGATIVE'}: skip={pooled['skip_rate']['router']:.1%} "
             f"router-hybrid={ci(pooled['router_minus_hybrid'])} router-random={ci(pooled['router_minus_random'])}")


if __name__ == "__main__":
    main()
