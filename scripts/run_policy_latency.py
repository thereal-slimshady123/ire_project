"""Measured per-query latency of every evaluated policy, matched to its held-out nDCG@10 (H4).

Every component is real: the cached BM25 analyzer/index, the e5-small-v2 query encoder, FAISS search over
the cached corpus embeddings, min-max fusion, the 18-feature extractor used by Experiments 2-4, LightGBM
router / gain-model inference, and cross-encoder/ms-marco-MiniLM-L-6-v2 reranking of the hybrid top-50.
Each test query is timed once per component (after warm-up); a policy's per-query latency is the sum of the
components it runs, with the cross-encoder added only on queries that the policy escalates.

Escalation decisions are replayed from the saved held-out artifacts (Experiment 4 policies at each budget,
router + tau escalation from run_tau_escalation.py, Experiment 3 skip decisions), so latency and quality are
reported for the same per-query decisions. The routers/gain model timed here are fit on the full dataset;
only their inference time is used, never their predictions.

Also writes Contract 4 (plan.md §3): results/experiment4/{dataset}_escalation.parquet.

Usage:
  python scripts/run_policy_latency.py [--device cpu]
"""

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from router_retrieval.data.schema import Candidate  # noqa: E402
from router_retrieval.features.query_features import extract_query_features, rare_idf_threshold  # noqa: E402
from router_retrieval.features.retrieval_signal_features import (  # noqa: E402
    compute_query_term_coverage, compute_score_entropy, compute_top_gap, compute_top_ratio,
    extract_retrieval_signal_features,
)
from router_retrieval.retrieval.bm25 import BM25Retriever, document_text, tokenize  # noqa: E402
from router_retrieval.retrieval.cross_encoder import CrossEncoderReranker  # noqa: E402
from router_retrieval.retrieval.dense import DenseRetriever  # noqa: E402
from router_retrieval.retrieval.hybrid import BM25_TRUNCATED_FLOOR, fuse_normalized_scores, normalize_for_fusion  # noqa: E402
from router_retrieval.retrieval.normalization import min_max_normalize  # noqa: E402
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

log = get_logger("policy_latency")
COMPONENTS = ["bm25_search", "dense_encode", "faiss_search", "hybrid_fusion", "feature_extraction",
              "skip_feature_extraction", "router_inference", "gain_model_inference", "cross_encoder_rerank"]
BUDGETS = [0.05, 0.10, 0.15, 0.20, 0.30, 0.50, 1.0]
EXP4_POLICIES = ["predicted_gain", "router_confidence", "random", "oracle"]
POOL = 50


def parse_args():
    p = argparse.ArgumentParser(description="Measured per-policy latency matched to held-out quality")
    p.add_argument("--datasets", nargs="+", default=["nfcorpus", "scifact", "fiqa"])
    p.add_argument("--data_dir", type=Path, default=Path("data/raw"))
    p.add_argument("--config_dir", type=Path, default=Path("configs"))
    p.add_argument("--results_dir", type=Path, default=Path("results"))
    p.add_argument("--output_dir", type=Path, default=Path("results/policy_latency"))
    p.add_argument("--dense_model", type=str, default="intfloat/e5-small-v2")
    p.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--no_contract4", action="store_true", help="Skip writing results/experiment4/*_escalation.parquet")
    p.add_argument("--no_plots", action="store_true")
    return p.parse_args()


def escalation_masks(frame: pd.DataFrame, budget: float, seed: int) -> Dict[str, np.ndarray]:
    """Experiment 4's rule: per (dataset, fold) escalate the top round(B * n) queries by policy score.

    Random escalation is a seeded sample of the same size (Experiment 4 reports its expected value).
    """
    masks = {pol: np.zeros(len(frame), dtype=bool) for pol in EXP4_POLICIES}
    rng = np.random.default_rng(seed)
    for _, idx in frame.groupby(["dataset", "fold"], sort=True).indices.items():
        n_esc = int(round(budget * len(idx)))
        for pol in ("predicted_gain", "router_confidence", "oracle"):
            order = np.argsort(-frame[f"score_{pol}"].values[idx], kind="stable")
            masks[pol][idx[order[:n_esc]]] = True
        masks["random"][rng.choice(idx, size=n_esc, replace=False)] = True
    return masks


def policy_latency(c: pd.DataFrame, policy: str, escalated=None, skipped=None) -> np.ndarray:
    """Per-query latency (ms) of a policy from per-query component timings."""
    hybrid = c["bm25_search"] + c["dense_encode"] + c["faiss_search"] + c["hybrid_fusion"]
    ce = c["cross_encoder_rerank"] * (np.asarray(escalated, dtype=float) if escalated is not None else 1.0)
    if policy == "always_bm25":
        out = c["bm25_search"]
    elif policy == "always_hybrid":
        out = hybrid
    elif policy == "always_rerank":
        out = hybrid + c["cross_encoder_rerank"]
    elif policy == "gbm_router":
        out = hybrid + c["feature_extraction"] + c["router_inference"]
    elif policy in ("router_confidence", "router_tau"):
        out = hybrid + c["feature_extraction"] + c["router_inference"] + ce
    elif policy == "predicted_gain":
        out = hybrid + c["feature_extraction"] + c["gain_model_inference"] + ce
    elif policy in ("random", "oracle"):
        out = hybrid + ce
    elif policy == "skip_dense":
        dense_part = (c["dense_encode"] + c["faiss_search"] + c["hybrid_fusion"]) * (1.0 - np.asarray(skipped, dtype=float))
        out = c["bm25_search"] + c["skip_feature_extraction"] + c["router_inference"] + dense_part
    else:
        raise ValueError(f"unknown policy {policy!r}")
    return np.asarray(out, dtype=float)


def time_dataset(ds: str, args, ret_cfg, hyb_cfg, router_cfg, ce_cfg, alpha: float, seed: int) -> pd.DataFrame:
    corpus, queries, _ = exp1b.load_split(args.data_dir / ds)
    k = ret_cfg["dense"].get("top_k", 100)
    bm25 = BM25Retriever(k1=ret_cfg["bm25"]["k1"], b=ret_cfg["bm25"]["b"])
    bm25.index(corpus)
    idf_vocab = {t: float(bm25.idf[j]) for t, j in bm25.vocab.items()}
    vocab, rare = set(bm25.vocab), rare_idf_threshold(bm25.num_docs)
    prefixes = exp1b.E5_PREFIXES if "e5" in args.dense_model.lower() else {}
    dense = DenseRetriever.load(args.results_dir / "indexes" / ds / args.dense_model.replace("/", "__"),
                                model_name=args.dense_model, batch_size=ret_cfg["dense"]["batch_size"],
                                device=args.device, **prefixes)
    ce = CrossEncoderReranker(model_name=ce_cfg["model_name"], batch_size=ce_cfg["batch_size"], device=args.device,
                              candidate_pool_size=POOL, max_length=ce_cfg["max_length"])

    # Timing-only models: same classes / hyper-parameters as the evaluated ones, fit on this dataset.
    table = pd.read_csv(args.results_dir / "experiment2" / "cache" / f"{ds}.csv", index_col=0, dtype={"query_id": str})
    table.index = table.index.astype(str)
    router = exp2.fit_router("gbm", table[exp2.FEATURES], exp2.route_matrix(table, alpha), 0.02, router_cfg, seed)
    import lightgbm as lgb
    g = router_cfg["models"]["gbm"]
    gain_cols = exp2.FEATURES + ["hyb_top1_score", "hyb_top1_top2_gap", "hyb_score_entropy"]
    gain_model = lgb.LGBMRegressor(n_estimators=g["n_estimators"], learning_rate=g["learning_rate"],
                                   num_leaves=g["num_leaves"], random_state=seed, verbose=-1)
    rng = np.random.default_rng(seed)
    gain_model.fit(pd.DataFrame(rng.random((len(table), len(gain_cols))), columns=gain_cols), rng.normal(size=len(table)))

    def doc_terms(doc_id):  # analyzed at query time, as a deployment without a term-set cache would
        return set(tokenize(document_text(corpus[doc_id])))

    def run_query(q) -> Dict[str, float]:
        t: Dict[str, float] = {}
        clock = time.perf_counter
        s = clock(); b = bm25.search(q, k); t["bm25_search"] = clock() - s
        s = clock(); emb = dense.encode([dense.query_prefix + q.text]); t["dense_encode"] = clock() - s
        s = clock()
        scores, idx = dense._index.search(emb, k)
        d = [Candidate(q.query_id, dense.doc_ids[i], float(sc), r + 1) for r, (sc, i) in enumerate(zip(scores[0], idx[0]))]
        t["faiss_search"] = clock() - s
        s = clock()
        hyb = fuse_normalized_scores(q.query_id, normalize_for_fusion(b, k, "min_max", BM25_TRUNCATED_FLOOR),
                                     normalize_for_fusion(d, k, "min_max"), alpha, k)
        t["hybrid_fusion"] = clock() - s
        s = clock()  # same features as exp1.compute_features, without rebuilding the vocabulary per call
        f = extract_query_features(q, idf_vocab, vocab, rare_threshold=rare)
        f.update(extract_retrieval_signal_features(b, d))
        f["bm25_top1_top2_ratio"] = compute_top_ratio(b)
        q_terms = tokenize(q.text)
        f["bm25_top10_query_term_coverage"] = compute_query_term_coverage(q_terms, [doc_terms(c.doc_id) for c in b[:10]])
        f["dense_top10_query_term_coverage"] = compute_query_term_coverage(q_terms, [doc_terms(c.doc_id) for c in d[:10]])
        feats = pd.DataFrame([f])[exp1.FEATURE_COLUMNS]
        t["feature_extraction"] = clock() - s
        s = clock()
        skip = extract_query_features(q, idf_vocab, vocab, rare_threshold=rare)
        skip.update({"bm25_top1_score": b[0].score if b else 0.0, "bm25_top1_top2_gap": compute_top_gap(min_max_normalize(b)),
                     "bm25_top1_top2_ratio": compute_top_ratio(b), "bm25_score_entropy": compute_score_entropy(b),
                     "bm25_top10_query_term_coverage": compute_query_term_coverage(
                         tokenize(q.text), [set(tokenize(document_text(corpus[c.doc_id]))) for c in b[:10]])})
        t["skip_feature_extraction"] = clock() - s
        s = clock(); router.predict_proba(feats[exp2.FEATURES]); t["router_inference"] = clock() - s
        s = clock()
        row = feats[exp2.FEATURES].copy()
        row["hyb_top1_score"] = hyb[0].score if hyb else 0.0
        row["hyb_top1_top2_gap"] = hyb[0].score - hyb[1].score if len(hyb) > 1 else 0.0
        row["hyb_score_entropy"] = compute_score_entropy(hyb)
        gain_model.predict(row[gain_cols])
        t["gain_model_inference"] = clock() - s
        s = clock(); ce.rerank(q, hyb, corpus, top_k=k); t["cross_encoder_rerank"] = clock() - s
        return {name: 1000.0 * v for name, v in t.items()}

    qlist = [queries[q] for q in sorted(queries)]
    for q in qlist[: args.warmup]:
        run_query(q)
    rows = {}
    for i, q in enumerate(qlist, start=1):
        rows[q.query_id] = run_query(q)
        if i % 100 == 0 or i == len(qlist):
            log.info(f"[{ds}] timed {i}/{len(qlist)} queries")
    out = pd.DataFrame.from_dict(rows, orient="index")[COMPONENTS]
    out.index = [f"{ds}:{q}" for q in out.index]
    out.insert(0, "dataset", ds)
    log.info(f"[{ds}] device={ce.device}; mean ms: " + ", ".join(f"{c}={out[c].mean():.2f}" for c in COMPONENTS))
    return out


def main():
    args = parse_args()
    exp_cfg = load_config(args.config_dir / "experiment.yaml")
    ret_cfg = load_config(args.config_dir / "retrievers.yaml")
    hyb_cfg = load_config(args.config_dir / "hybrid.yaml")
    router_cfg = load_config(args.config_dir / "router.yaml")
    ce_cfg = load_config(args.config_dir / "cross_encoder.yaml")
    seed = exp_cfg["seed"]
    seed_everything(seed)
    r = args.results_dir
    exp2_summary = json.loads((r / "experiment2" / "summary.json").read_text(encoding="utf-8"))
    exp2_held = pd.read_csv(r / "experiment2" / "per_query_heldout.csv", index_col=0)
    exp3_held = pd.read_csv(r / "experiment3" / "per_query_heldout.csv", index_col=0)
    exp4_held = pd.read_csv(r / "experiment4" / "per_query_heldout.csv", index_col=0)
    tau_path = r / "tau_escalation" / "per_query_heldout.csv"
    tau_held = pd.read_csv(tau_path, index_col=0) if tau_path.exists() else None

    args.output_dir.mkdir(parents=True, exist_ok=True)
    timings = []
    for ds in args.datasets:
        cache = args.output_dir / f"{ds}_components.csv"
        if cache.exists():
            timings.append(pd.read_csv(cache, index_col=0))
            continue
        alphas = [f["alpha"] for f in exp2_summary["fold_info"][ds]]
        comp = time_dataset(ds, args, ret_cfg, hyb_cfg, router_cfg, ce_cfg, max(set(alphas), key=alphas.count), seed)
        comp.to_csv(cache)
        timings.append(comp)
    timings = pd.concat(timings)

    systems, per_query = [], {}

    def add(ds, name, ndcg, lat):
        lat = np.asarray(lat, dtype=float)
        systems.append({"dataset": ds, "system": name, "ndcg_10": float(np.mean(ndcg)), "n": len(lat),
                        "mean_ms": float(lat.mean()), "p50_ms": float(np.percentile(lat, 50)),
                        "p95_ms": float(np.percentile(lat, 95))})
        per_query.setdefault(ds, {})[name] = lat

    for ds in args.datasets:
        c = timings[timings["dataset"] == ds]
        keys = list(c.index)
        e4 = exp4_held.loc[keys]
        e2 = exp2_held.loc[keys]
        table = pd.read_csv(r / "experiment2" / "cache" / f"{ds}.csv", index_col=0, dtype={"query_id": str})
        table.index = [f"{ds}:{q}" for q in table.index.astype(str)]
        add(ds, "always_bm25", table.loc[keys, "ndcg_bm25"], policy_latency(c, "always_bm25"))
        add(ds, "always_hybrid", e4["hybrid"], policy_latency(c, "always_hybrid"))
        add(ds, "gbm_router", e2["router_gbm"], policy_latency(c, "gbm_router"))
        add(ds, "always_rerank_minilm_l6", e4["reranked"], policy_latency(c, "always_rerank"))
        for B in BUDGETS[:-1]:
            masks = escalation_masks(e4, B, seed)
            for pol in EXP4_POLICIES:
                ndcg = np.where(masks[pol], e4["reranked"], e4["hybrid"])
                add(ds, f"{pol}_escalation_{B:.0%}", ndcg, policy_latency(c, pol, escalated=masks[pol]))
        if tau_held is not None:
            t = tau_held.loc[keys]
            for rate in (0.05, 0.10, 0.15):
                esc = t[f"escalated@{rate}"].astype(bool).values
                add(ds, f"router_tau_escalation_{rate:.0%}", t[f"router_tau@{rate}"],
                    policy_latency(c, "router_tau", escalated=esc))
        if set(keys) <= set(exp3_held.index):
            e3 = exp3_held.loc[keys]
            add(ds, "skip_dense_gbm", e3["router"], policy_latency(c, "skip_dense", skipped=e3["skip_router"].astype(bool)))

    table = pd.DataFrame(systems)
    table.to_csv(args.output_dir / "policy_latency.csv", index=False)
    comp_summary = {ds: {comp: {"mean": float(g[comp].mean()), "p50": float(g[comp].median()),
                                "p95": float(g[comp].quantile(0.95))} for comp in COMPONENTS}
                    for ds, g in timings.groupby("dataset")}
    device = CrossEncoderReranker(device=args.device).device
    (args.output_dir / "summary.json").write_text(json.dumps(
        {"device": device, "dense_model": args.dense_model, "cross_encoder": ce_cfg["model_name"],
         "cross_encoder_batch_size": ce_cfg["batch_size"], "candidate_pool": POOL,
         "components_ms": comp_summary, "systems": systems}, indent=2), encoding="utf-8")

    lines = ["# Measured policy latency matched to held-out quality", "",
             f"Device: `{device}` (dense encoder and cross-encoder). Every component is timed for real on every test "
             "query (one pass after warm-up). Policy latency = sum of the components it runs; the cross-encoder "
             "(MiniLM-L6, hybrid top-50) is added only on queries the policy escalates. Escalation decisions are "
             "replayed from the saved held-out artifacts. Random escalation uses one seeded sample, so its nDCG "
             "here differs slightly from Experiment 4's expected value.", "",
             "## Component latency (ms)", "", "| Dataset | Component | Mean | p50 | p95 |", "| :--- | :--- | ---: | ---: | ---: |"]
    for ds, comps in comp_summary.items():
        for comp, v in comps.items():
            lines.append(f"| {ds} | {comp} | {v['mean']:.2f} | {v['p50']:.2f} | {v['p95']:.2f} |")
    lines += ["", "## Policies (nDCG@10 and per-query latency)", "",
              "| Dataset | System | nDCG@10 | Mean ms | p50 ms | p95 ms |", "| :--- | :--- | ---: | ---: | ---: | ---: |"]
    keep = ("always_bm25", "always_hybrid", "gbm_router", "always_rerank_minilm_l6", "skip_dense_gbm")
    for row in systems:
        if row["system"] in keep or "10%" in row["system"]:
            ndcg = "n/a" if np.isnan(row["ndcg_10"]) else f"{row['ndcg_10']:.4f}"
            lines.append(f"| {row['dataset']} | {row['system']} | {ndcg} | {row['mean_ms']:.2f} "
                         f"| {row['p50_ms']:.2f} | {row['p95_ms']:.2f} |")
    lines += ["", "All budgets are in `policy_latency.csv`."]
    (args.output_dir / "tables.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    if not args.no_contract4:
        lat_by_key = {ds: timings[timings["dataset"] == ds] for ds in args.datasets}
        for ds, e4 in exp4_held.groupby("dataset", sort=False):
            c = lat_by_key.get(ds)
            frames = []
            for B in BUDGETS:
                masks = escalation_masks(e4, B, seed)
                for pol in EXP4_POLICIES:
                    frames.append(pd.DataFrame({
                        "query_id": [k.split(":", 1)[1] for k in e4.index], "budget": np.float32(B), "policy": pol,
                        "escalated": masks[pol],
                        "ndcg_final": np.where(masks[pol], e4["reranked"], e4["hybrid"]).astype(np.float32),
                        "latency_ms": (policy_latency(c.loc[e4.index], pol, escalated=masks[pol]).astype(np.float32)
                                       if c is not None else np.float32(np.nan)),
                    }))
            pd.concat(frames, ignore_index=True).to_parquet(r / "experiment4" / f"{ds}_escalation.parquet", index=False)
        log.info("Wrote Contract 4 escalation parquets to results/experiment4/")

    if not args.no_plots:
        plt = exp1._style()
        fig, axes = plt.subplots(1, len(args.datasets), figsize=(4.4 * len(args.datasets), 4.0), squeeze=False)
        style = {"always_bm25": ("BM25", "#8a8a8a", "s"), "always_hybrid": ("Always-hybrid", "#1baf7a", "D"),
                 "gbm_router": ("GBM router", "#2a78d6", "^"), "always_rerank_minilm_l6": ("Always-rerank", "#d43d51", "X"),
                 "skip_dense_gbm": ("Skip-dense", "#7b4fd6", "P")}
        for ax, ds in zip(axes[0], args.datasets):
            sub = table[table["dataset"] == ds].set_index("system")
            for pol, color in (("predicted_gain", "#eb6834"), ("random", "#52514e")):
                rows_ = sub.loc[[f"{pol}_escalation_{B:.0%}" for B in BUDGETS[:-1]]]
                ax.plot(rows_["mean_ms"], rows_["ndcg_10"], marker="o", markersize=3, color=color, linewidth=1.2,
                        label=f"{pol.replace('_', '-')} escalation (5–50%)")
            for name, (label, color, marker) in style.items():
                if name in sub.index and not np.isnan(sub.loc[name, "ndcg_10"]):
                    ax.scatter(sub.loc[name, "mean_ms"], sub.loc[name, "ndcg_10"], color=color, marker=marker, s=46,
                               label=label, zorder=3)
            ax.set_xscale("log")
            ax.set(xlabel=f"Mean per-query latency (ms, log, {device})", ylabel="Held-out nDCG@10", title=ds)
        axes[0][0].legend(loc="lower right", fontsize=7)
        fig.tight_layout()
        (args.output_dir / "figures").mkdir(parents=True, exist_ok=True)
        fig.savefig(args.output_dir / "figures" / "quality_vs_latency.png", dpi=200)
        plt.close(fig)


if __name__ == "__main__":
    main()
