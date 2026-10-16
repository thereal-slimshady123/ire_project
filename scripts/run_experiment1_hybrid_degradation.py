"""Experiment 1: does always picking a fixed-alpha hybrid degrade accuracy on identifiable queries?

Implements the protocol in improvements.md §1:
  Step 1  BM25 / dense top-k runs (cached), per-query min-max fusion over the alpha grid,
          alpha* chosen by k-fold CV on training folds (hybrid scores are out-of-fold).
  Step 2  per-query nDCG@10 / MRR@10 / MAP / Recall@100 and the deltas vs. Hybrid_alpha*.
  Step 3  win / tie / loss partitioning with margin epsilon.
  Step 4  H1 (prevalence), H2 (oracle headroom), partition-level paired tests.
  §1.5    H3 diagnostics: query + retrieval-signal features vs. degradation.
  §1.6    result tables + figures.

Usage:
  python scripts/run_experiment1_hybrid_degradation.py --datasets nfcorpus scifact fiqa
"""

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from router_retrieval.data.beir_loader import load_beir_dataset, qrels_to_dict  # noqa: E402
from router_retrieval.data.schema import Candidate  # noqa: E402
from router_retrieval.data.splits import create_kfold_splits  # noqa: E402
from router_retrieval.evaluation.ranking_metrics import compute_ranking_metrics  # noqa: E402
from router_retrieval.evaluation.significance import paired_bootstrap_test, paired_wilcoxon_test  # noqa: E402
from router_retrieval.features.query_features import extract_query_features, rare_idf_threshold  # noqa: E402
from router_retrieval.features.retrieval_signal_features import (  # noqa: E402
    compute_query_term_coverage,
    compute_top_ratio,
    extract_retrieval_signal_features,
)
from router_retrieval.retrieval.bm25 import BM25Retriever, document_text, tokenize  # noqa: E402
from router_retrieval.retrieval.dense import DenseRetriever  # noqa: E402
from router_retrieval.retrieval.hybrid import (  # noqa: E402
    BM25_TRUNCATED_FLOOR,
    fuse_normalized_scores,
    normalize_for_fusion,
)
from router_retrieval.utils.config import load_config  # noqa: E402
from router_retrieval.utils.logging_utils import get_logger  # noqa: E402
from router_retrieval.utils.seeding import seed_everything  # noqa: E402

METRICS = ["ndcg_cut_10", "mrr_10", "map", "recall_100"]
PRIMARY = "ndcg_cut_10"
ROUTES = ["bm25", "dense", "hybrid"]
H1_MIN_PREVALENCE = 0.15
H2_MIN_HEADROOM = 0.02
H2_ALPHA = 0.01

FEATURE_COLUMNS = [
    "query_length", "max_idf", "mean_idf", "rare_term_density", "entity_density", "oov_fraction",
    "jaccard_overlap_top10", "jaccard_overlap_top20", "rank_correlation_spearman",
    "bm25_top1_score", "bm25_top1_top2_gap", "bm25_top1_top2_ratio", "bm25_score_entropy",
    "dense_top1_score", "dense_top1_top2_gap", "dense_score_entropy",
    "bm25_top10_query_term_coverage", "dense_top10_query_term_coverage",
]

log = get_logger("experiment1")


def parse_args():
    parser = argparse.ArgumentParser(description="Experiment 1: fixed-alpha hybrid degradation analysis")
    parser.add_argument("--datasets", nargs="+", default=None, help="Datasets (default: configs/experiment.yaml)")
    parser.add_argument("--data_dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--config_dir", type=Path, default=Path("configs"))
    parser.add_argument("--results_dir", type=Path, default=Path("results"), help="Root for cached runs/indexes")
    parser.add_argument("--output_dir", type=Path, default=Path("results/experiment1"))
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--epsilon", type=float, default=None, help="Margin (default: labels.yaml default_epsilon)")
    parser.add_argument("--n_bootstrap", type=int, default=10000)
    parser.add_argument("--dense_model", type=str, default=None, help="Override retrievers.yaml dense.model_name")
    parser.add_argument("--device", type=str, default=None, help="Override retrievers.yaml dense.device")
    parser.add_argument("--no_plots", action="store_true")
    return parser.parse_args()


# --------------------------------------------------------------------------------------
# Step 1: run execution & persistence
# --------------------------------------------------------------------------------------

def write_trec_run(run: Dict[str, List[Candidate]], path: Path, tag: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for qid, cands in run.items():
            for c in cands:
                f.write(f"{qid} Q0 {c.doc_id} {c.rank} {c.score:.8f} {tag}\n")


def read_trec_run(path: Path) -> Dict[str, List[Candidate]]:
    run: Dict[str, List[Candidate]] = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            qid, _, doc_id, rank, score, _ = line.split()
            run.setdefault(qid, []).append(Candidate(qid, doc_id, float(score), int(rank)))
    for cands in run.values():
        cands.sort(key=lambda c: c.rank)
    return run


def get_runs(dataset: str, corpus, queries, args, ret_cfg) -> Tuple[BM25Retriever, dict, dict]:
    """Return (bm25 index, bm25 run, dense run), computing and caching runs on first use."""
    k = ret_cfg["dense"].get("top_k", 100)
    bm25_cfg = ret_cfg["bm25"]
    log.info(f"[{dataset}] indexing BM25 (k1={bm25_cfg['k1']}, b={bm25_cfg['b']}) over {len(corpus)} docs")
    bm25 = BM25Retriever(k1=bm25_cfg["k1"], b=bm25_cfg["b"])
    bm25.index(corpus)  # cheap; also needed for IDF features

    run_dir = args.results_dir / "runs" / dataset
    qlist = [queries[q] for q in sorted(queries)]

    bm25_path = run_dir / f"bm25_k1-{bm25_cfg['k1']}_b-{bm25_cfg['b']}.trec"
    if bm25_path.exists():
        bm25_run = read_trec_run(bm25_path)
    else:
        bm25_run = bm25.batch_search(qlist, k)
        write_trec_run(bm25_run, bm25_path, "bm25")

    dense_cfg = dict(ret_cfg["dense"])
    if args.dense_model:
        dense_cfg["model_name"] = args.dense_model
    if args.device:
        dense_cfg["device"] = args.device
    slug = dense_cfg["model_name"].replace("/", "__")
    dense_path = run_dir / f"dense_{slug}.trec"
    if dense_path.exists():
        dense_run = read_trec_run(dense_path)
    else:
        index_dir = args.results_dir / "indexes" / dataset / slug
        kwargs = dict(
            batch_size=dense_cfg["batch_size"], device=dense_cfg["device"],
            faiss_index_type=dense_cfg["faiss_index_type"], max_length=dense_cfg.get("max_length"),
        )
        if (index_dir / "embeddings.npy").exists():
            dense = DenseRetriever.load(index_dir, dense_cfg["model_name"], **kwargs)
        else:
            dense = DenseRetriever(model_name=dense_cfg["model_name"], **kwargs)
            log.info(f"[{dataset}] encoding corpus with {dense_cfg['model_name']} on {dense.device} (one-off, cached)")
            dense.index(corpus)
            dense.save(index_dir)
        dense_run = dense.batch_search(qlist, k)
        write_trec_run(dense_run, dense_path, slug)

    # Queries with no BM25 match must still be present (empty list) for per-query evaluation.
    for qid in queries:
        bm25_run.setdefault(qid, [])
        dense_run.setdefault(qid, [])
    return bm25, bm25_run, dense_run


def build_hybrid_runs(bm25_run, dense_run, alphas, k, method) -> Dict[float, Dict[str, List[Candidate]]]:
    norm_b = {q: normalize_for_fusion(c_list, k, method, BM25_TRUNCATED_FLOOR) for q, c_list in bm25_run.items()}
    norm_d = {q: normalize_for_fusion(c_list, k, method) for q, c_list in dense_run.items()}
    return {
        a: {q: fuse_normalized_scores(q, norm_b[q], norm_d.get(q, {}), a, k) for q in bm25_run}
        for a in alphas
    }


def select_alpha_cv(
    alpha_ndcg: pd.DataFrame, n_folds: int, seed: int
) -> Tuple[pd.Series, List[dict]]:
    """Pick alpha* on each training fold; return per-query out-of-fold alpha and fold log."""
    oof_alpha = pd.Series(index=alpha_ndcg.index, dtype=np.float64)
    folds = []
    for i, (train, test) in enumerate(create_kfold_splits(list(alpha_ndcg.index), n_folds, seed)):
        train_means = alpha_ndcg.loc[train].mean(axis=0)
        best = float(train_means.idxmax())
        oof_alpha.loc[test] = best
        folds.append({
            "fold": i, "alpha_star": best, "train_ndcg": float(train_means.max()),
            "test_ndcg": float(alpha_ndcg.loc[test, best].mean()), "n_test": len(test),
        })
    return oof_alpha, folds


# --------------------------------------------------------------------------------------
# Step 3: partitioning
# --------------------------------------------------------------------------------------

def classify_query(b: float, d: float, h: float, eps: float) -> str:
    """
    Mutually exclusive, exhaustive partition of a query by which route wins by >= eps.

    bm25_dominant / dense_dominant / hybrid_dominant: winner beats both others by >= eps.
    tie: all three within eps.
    near_tie_degraded / near_tie_safe: no single winner, but the spread is >= eps; split by
    whether the better single retriever beats hybrid by >= eps (i.e. hybrid still degrades).
    """
    scores = {"bm25": b, "dense": d, "hybrid": h}
    ranked = sorted(scores.items(), key=lambda p: -p[1])
    if ranked[0][1] - ranked[-1][1] < eps:
        return "tie"
    if ranked[0][1] - ranked[1][1] >= eps:
        return f"{ranked[0][0]}_dominant"
    return "near_tie_degraded" if max(b, d) - h >= eps else "near_tie_safe"


PARTITIONS = ["bm25_dominant", "dense_dominant", "hybrid_dominant", "near_tie_degraded", "near_tie_safe", "tie"]


# --------------------------------------------------------------------------------------
# §1.5: features
# --------------------------------------------------------------------------------------

def compute_features(queries, corpus, bm25: BM25Retriever, bm25_run, dense_run) -> pd.DataFrame:
    idf_vocab = {t: float(bm25.idf[j]) for t, j in bm25.vocab.items()}
    vocab = set(bm25.vocab)
    rare = rare_idf_threshold(bm25.num_docs)
    doc_terms_cache: Dict[str, set] = {}

    def doc_terms(doc_id):
        if doc_id not in doc_terms_cache:
            doc_terms_cache[doc_id] = set(tokenize(document_text(corpus[doc_id])))
        return doc_terms_cache[doc_id]

    rows = {}
    for qid, query in queries.items():
        b, d = bm25_run[qid], dense_run[qid]
        feats = extract_query_features(query, idf_vocab, vocab, rare_threshold=rare)
        feats.update(extract_retrieval_signal_features(b, d))
        feats["bm25_top1_top2_ratio"] = compute_top_ratio(b)
        q_terms = tokenize(query.text)
        feats["bm25_top10_query_term_coverage"] = compute_query_term_coverage(q_terms, [doc_terms(c.doc_id) for c in b[:10]])
        feats["dense_top10_query_term_coverage"] = compute_query_term_coverage(q_terms, [doc_terms(c.doc_id) for c in d[:10]])
        rows[qid] = feats
    df = pd.DataFrame.from_dict(rows, orient="index")[FEATURE_COLUMNS]
    df.index.name = "query_id"
    return df


# --------------------------------------------------------------------------------------
# Step 4 + §1.5: statistics
# --------------------------------------------------------------------------------------

def wilson_ci(successes: int, n: int, z: float = 1.96) -> Tuple[float, float]:
    if n == 0:
        return 0.0, 0.0
    p = successes / n
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return centre - half, centre + half


def paired_tests(a, b, n_boot, seed) -> dict:
    boot = paired_bootstrap_test(a, b, n_resamples=n_boot, seed=seed)
    wil = paired_wilcoxon_test(a, b, alternative="greater")
    return {**boot, "wilcoxon_p_greater": wil["p_value"], "wilcoxon_n_nonzero": wil["n_nonzero"]}


def h1_prevalence(pq: pd.DataFrame, eps: float, hybrid_col: str) -> dict:
    n = len(pq)
    max_single = pq[["ndcg_bm25", "ndcg_dense"]].max(axis=1)
    degraded = int((max_single - pq[hybrid_col] >= eps).sum())
    lo, hi = wilson_ci(degraded, n)
    binom = stats.binomtest(degraded, n, H1_MIN_PREVALENCE, alternative="greater")
    return {
        "n_queries": n,
        "n_degraded": degraded,
        "prevalence": degraded / n,
        "prevalence_ci95": [lo, hi],
        "binomtest_p_vs_15pct": float(binom.pvalue),
        "share_bm25_beats_hybrid": float((pq["ndcg_bm25"] - pq[hybrid_col] >= eps).mean()),
        "share_dense_beats_hybrid": float((pq["ndcg_dense"] - pq[hybrid_col] >= eps).mean()),
        "share_hybrid_beats_both": float((pq[hybrid_col] - max_single >= eps).mean()),
        "supported": bool(lo >= H1_MIN_PREVALENCE) if n else False,
        "supported_point_estimate": bool(degraded / n >= H1_MIN_PREVALENCE) if n else False,
    }


def h2_headroom(pq: pd.DataFrame, hybrid_col: str, n_boot: int, seed: int) -> dict:
    oracle3 = pq[["ndcg_bm25", "ndcg_dense", hybrid_col]].max(axis=1)
    oracle2 = pq[["ndcg_bm25", "ndcg_dense"]].max(axis=1)
    t3 = paired_tests(oracle3.values, pq[hybrid_col].values, n_boot, seed)
    t2 = paired_tests(oracle2.values, pq[hybrid_col].values, n_boot, seed)
    return {
        "oracle3_ndcg": float(oracle3.mean()),
        "hybrid_ndcg": float(pq[hybrid_col].mean()),
        "headroom": t3["delta_mean"],
        "oracle3_vs_hybrid": t3,
        # Router restricted to the two single retrievers (never fuses) vs. always-fuse.
        "oracle2_ndcg": float(oracle2.mean()),
        "oracle2_vs_hybrid": t2,
        "supported": bool(t3["delta_mean"] >= H2_MIN_HEADROOM and t3["p_value"] < H2_ALPHA),
    }


def partition_tests(pq: pd.DataFrame, n_boot: int, seed: int) -> dict:
    out = {}
    for part, single in (("bm25_dominant", "ndcg_bm25"), ("dense_dominant", "ndcg_dense")):
        sub = pq[pq["partition"] == part]
        if len(sub) < 2:
            out[part] = {"n": len(sub), "note": "too few queries to test"}
            continue
        out[part] = paired_tests(sub[single].values, sub["ndcg_hybrid"].values, n_boot, seed)
    return out


def correlation_table(pq: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for target in ("delta_bm25", "delta_dense", "delta_max_single"):
        for feat in FEATURE_COLUMNS:
            x, y = pq[feat], pq[target]
            if x.nunique() < 2:
                rho, p = np.nan, np.nan
            else:
                res = stats.spearmanr(x, y)
                rho, p = float(res.statistic), float(res.pvalue)
            rows.append({"target": target, "feature": feat, "spearman_rho": rho, "p_value": p})
    df = pd.DataFrame(rows)
    valid = df["p_value"].notna()
    df.loc[valid, "q_value_bh"] = stats.false_discovery_control(df.loc[valid, "p_value"].values, method="bh")
    return df


def mann_whitney(group_a, group_b, alternative="greater") -> dict:
    group_a, group_b = np.asarray(group_a), np.asarray(group_b)
    if len(group_a) < 2 or len(group_b) < 2:
        return {"n_a": len(group_a), "n_b": len(group_b), "p_value": None}
    res = stats.mannwhitneyu(group_a, group_b, alternative=alternative)
    return {
        "n_a": int(len(group_a)), "n_b": int(len(group_b)),
        "mean_a": float(group_a.mean()), "mean_b": float(group_b.mean()),
        "median_a": float(np.median(group_a)), "median_b": float(np.median(group_b)),
        "U": float(res.statistic), "p_value": float(res.pvalue),
    }


def h3_diagnostics(pq: pd.DataFrame, eps: float) -> dict:
    out = {}
    q_idf = pq["max_idf"].quantile(0.75)
    out["H3a_maxidf_top_quartile_vs_rest__delta_bm25"] = {
        "threshold": float(q_idf),
        **mann_whitney(pq.loc[pq["max_idf"] > q_idf, "delta_bm25"], pq.loc[pq["max_idf"] <= q_idf, "delta_bm25"]),
    }
    q_rare = pq["rare_term_density"] > 0
    out["H3a_has_rare_term_vs_none__delta_bm25"] = mann_whitney(
        pq.loc[q_rare, "delta_bm25"], pq.loc[~q_rare, "delta_bm25"]
    )
    q_cov = pq["dense_top10_query_term_coverage"].quantile(0.25)
    out["H3b_low_dense_keyword_coverage_vs_rest__delta_dense"] = {
        "threshold": float(q_cov),
        **mann_whitney(
            pq.loc[pq["dense_top10_query_term_coverage"] <= q_cov, "delta_dense"],
            pq.loc[pq["dense_top10_query_term_coverage"] > q_cov, "delta_dense"],
        ),
    }
    q_gap = pq["bm25_top1_top2_gap"].quantile(0.25)
    out["H3b_low_bm25_gap_vs_rest__delta_dense"] = {
        "threshold": float(q_gap),
        **mann_whitney(
            pq.loc[pq["bm25_top1_top2_gap"] <= q_gap, "delta_dense"],
            pq.loc[pq["bm25_top1_top2_gap"] > q_gap, "delta_dense"],
        ),
    }
    zero = pq["jaccard_overlap_top10"] == 0
    out["H3c_jaccard10_zero_vs_positive__delta_max_single"] = mann_whitney(
        pq.loc[zero, "delta_max_single"], pq.loc[~zero, "delta_max_single"]
    )
    bins = pd.cut(pq["jaccard_overlap_top10"], [-0.001, 0.0, 0.2, 0.4, 1.0], labels=["0", "(0,0.2]", "(0.2,0.4]", ">0.4"])
    by_bin = pq.groupby(bins, observed=False).agg(
        n=("delta_max_single", "size"),
        mean_delta_max_single=("delta_max_single", "mean"),
        degradation_rate=("degraded", "mean"),
    )
    out["H3c_by_jaccard_bin"] = by_bin.reset_index().rename(columns={"jaccard_overlap_top10": "bin"}).to_dict("records")
    kw = {}
    for feat in ("max_idf", "jaccard_overlap_top10", "dense_top10_query_term_coverage", "bm25_top1_top2_gap"):
        groups = [g[feat].values for _, g in pq.groupby("partition") if len(g) >= 5]
        if len(groups) >= 2:
            kw[feat] = float(stats.kruskal(*groups).pvalue)
    out["kruskal_wallis_feature_across_partitions_p"] = kw
    return out


# --------------------------------------------------------------------------------------
# §1.6: figures
# --------------------------------------------------------------------------------------

# Reference categorical palette, slots 1-3 (validated all-pairs, light mode).
ROUTE_COLORS = {"bm25": "#2a78d6", "dense": "#eb6834", "hybrid": "#1baf7a"}
INK, INK_2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"


def _style():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": GRID, "axes.labelcolor": INK_2, "axes.titlecolor": INK,
        "axes.titlesize": 11, "axes.titleweight": "semibold", "axes.labelsize": 9.5,
        "xtick.color": INK_2, "ytick.color": INK_2, "xtick.labelsize": 8.5, "ytick.labelsize": 8.5,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
        "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
        "legend.fontsize": 8.5, "font.size": 9.5, "lines.linewidth": 2,
    })
    return plt


def make_figures(dataset: str, pq: pd.DataFrame, alpha_curve: pd.DataFrame, alpha_star: float, eps: float, fig_dir: Path):
    plt = _style()
    fig_dir.mkdir(parents=True, exist_ok=True)

    # 1. Degradation scatter.
    fig, ax = plt.subplots(figsize=(5.2, 5.0))
    max_single = pq[["ndcg_bm25", "ndcg_dense"]].max(axis=1)
    bm25_better = pq["ndcg_bm25"] >= pq["ndcg_dense"]
    for mask, route, label in ((bm25_better, "bm25", "Better single = BM25"), (~bm25_better, "dense", "Better single = Dense")):
        ax.scatter(max_single[mask], pq.loc[mask, "ndcg_hybrid"], s=22, alpha=0.6, color=ROUTE_COLORS[route],
                   edgecolors=SURFACE, linewidths=0.6, label=label, zorder=3)
    ax.plot([0, 1], [0, 1], color=INK_2, linewidth=1, zorder=2)
    ax.plot([eps, 1], [0, 1 - eps], color=INK_2, linewidth=0.8, linestyle=(0, (3, 3)), zorder=2)
    share = pq["degraded"].mean()
    ax.text(0.97, 0.05, f"{share:.0%} of queries ≥ {eps} below y = x\n(hybrid degraded)",
            ha="right", va="bottom", transform=ax.transAxes, color=INK, fontsize=8.5)
    ax.set(xlim=(-0.02, 1.02), ylim=(-0.02, 1.02), xlabel="max(nDCG@10 BM25, nDCG@10 Dense)",
           ylabel=f"nDCG@10 Hybrid (CV α*)", title=f"{dataset}: per-query hybrid vs. best single retriever")
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(fig_dir / "degradation_scatter.png", dpi=200)
    plt.close(fig)

    # 2. CDF of delta_max_single.
    fig, ax = plt.subplots(figsize=(5.6, 3.6))
    x = np.sort(pq["delta_max_single"].values)
    ax.step(x, np.arange(1, len(x) + 1) / len(x), where="post", color=ROUTE_COLORS["bm25"])
    ax.axvline(0, color=INK_2, linewidth=0.8)
    ax.axvline(eps, color=INK_2, linewidth=0.8, linestyle=(0, (3, 3)))
    ax.text(eps, 0.04, f"  ε = {eps}", color=INK_2, fontsize=8)
    ax.set(xlabel="Δ_MaxSingle = max(BM25, Dense) − Hybrid  (nDCG@10)", ylabel="Fraction of queries",
           title=f"{dataset}: CDF of per-query hybrid degradation", ylim=(0, 1.01))
    fig.tight_layout()
    fig.savefig(fig_dir / "delta_cdf.png", dpi=200)
    plt.close(fig)

    # 3. Feature boxplots by partition (category on the axis; no color encoding needed).
    order = [p for p in PARTITIONS if (pq["partition"] == p).any()]
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))
    for ax, feat, label in ((axes[0], "max_idf", "Max-IDF"), (axes[1], "jaccard_overlap_top10", "Jaccard@10 (BM25 vs Dense)")):
        data = [pq.loc[pq["partition"] == p, feat].values for p in order]
        ax.boxplot(data, widths=0.5, patch_artist=True, showfliers=True,
                   boxprops=dict(facecolor="#dfe9f7", edgecolor=INK_2, linewidth=0.8),
                   medianprops=dict(color=INK, linewidth=1.4), whiskerprops=dict(color=INK_2, linewidth=0.8),
                   capprops=dict(color=INK_2, linewidth=0.8),
                   flierprops=dict(marker="o", markersize=3, markerfacecolor=INK_2, markeredgecolor="none", alpha=0.5))
        ax.set_xticks(range(1, len(order) + 1))
        tick_names = [p.replace("near_tie_", "near tie\n").replace("_", "\n") for p in order]
        ax.set_xticklabels(
            [f"{name}\n(n={(pq['partition'] == p).sum()})" for name, p in zip(tick_names, order)], fontsize=7.5
        )
        ax.set(ylabel=label, title=label)
        ax.grid(axis="x", visible=False)
    fig.suptitle(f"{dataset}: query features by winning-route partition", color=INK, fontsize=11, fontweight="semibold")
    fig.tight_layout()
    fig.savefig(fig_dir / "feature_boxplots.png", dpi=200)
    plt.close(fig)

    # 4. Alpha sweep: mean nDCG and degradation share are different measures -> two panels, no dual axis.
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.5, 3.6))
    ax1.plot(alpha_curve["alpha"], alpha_curve["mean_ndcg"], color=ROUTE_COLORS["hybrid"], marker="o", markersize=4)
    ax1.axvline(alpha_star, color=INK_2, linewidth=0.8, linestyle=(0, (3, 3)))
    ax1.text(alpha_star, ax1.get_ylim()[0], f" α* = {alpha_star:g}", color=INK_2, fontsize=8, va="bottom")
    ax1.set(xlabel="α (weight on dense)", ylabel="Mean nDCG@10", title="Mean hybrid nDCG@10 vs α")
    ax2.plot(alpha_curve["alpha"], alpha_curve["degraded_share"], color=ROUTE_COLORS["dense"], marker="o", markersize=4,
             label="Hybrid degraded (single ≥ ε better)")
    ax2.plot(alpha_curve["alpha"], alpha_curve["hybrid_dominant_share"], color=ROUTE_COLORS["bm25"], marker="o",
             markersize=4, label="Hybrid dominant (≥ ε better than both)")
    ax2.axvline(alpha_star, color=INK_2, linewidth=0.8, linestyle=(0, (3, 3)))
    ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    top = max(alpha_curve["degraded_share"].max(), alpha_curve["hybrid_dominant_share"].max())
    ax2.set(xlabel="α (weight on dense)", ylabel="Share of queries", title="No single α removes degradation",
            ylim=(0, top * 1.45))  # headroom so the legend sits above the lines
    ax2.legend(loc="upper center", ncol=1)
    fig.suptitle(dataset, color=INK, fontsize=11, fontweight="semibold")
    fig.tight_layout()
    fig.savefig(fig_dir / "alpha_sweep.png", dpi=200)
    plt.close(fig)


def make_pooled_cdf(all_pq: Dict[str, pd.DataFrame], eps: float, fig_dir: Path):
    plt = _style()
    fig_dir.mkdir(parents=True, exist_ok=True)
    colors = ["#2a78d6", "#eb6834", "#1baf7a"]  # slots 1-3 (datasets, not routes)
    fig, ax = plt.subplots(figsize=(6.0, 3.8))
    for (name, pq), color in zip(all_pq.items(), colors):
        x = np.sort(pq["delta_max_single"].values)
        ax.step(x, np.arange(1, len(x) + 1) / len(x), where="post", color=color, label=name)
    ax.axvline(0, color=INK_2, linewidth=0.8)
    ax.axvline(eps, color=INK_2, linewidth=0.8, linestyle=(0, (3, 3)))
    ax.set(xlabel="Δ_MaxSingle (nDCG@10)", ylabel="Fraction of queries", ylim=(0, 1.01),
           title="CDF of per-query hybrid degradation, all datasets")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(fig_dir / "delta_cdf_all_datasets.png", dpi=200)
    plt.close(fig)


# --------------------------------------------------------------------------------------
# Driver
# --------------------------------------------------------------------------------------

def run_dataset(dataset: str, args, cfg) -> Tuple[pd.DataFrame, dict]:
    eps, seed = cfg["eps"], cfg["seed"]
    corpus, queries, qrels = load_beir_dataset(args.data_dir / dataset, args.split)
    log.info(f"[{dataset}] {len(corpus)} docs, {len(queries)} {args.split} queries, {len(qrels)} qrels")
    qrels_d = {q: r for q, r in qrels_to_dict(qrels).items() if q in queries}

    bm25, bm25_run, dense_run = get_runs(dataset, corpus, queries, args, cfg["retrievers"])
    k = cfg["retrievers"]["dense"].get("top_k", 100)

    # Per-route metrics for the single retrievers.
    m_bm25 = compute_ranking_metrics(qrels_d, bm25_run, METRICS)
    m_dense = compute_ranking_metrics(qrels_d, dense_run, METRICS)

    # Hybrid over the alpha grid, alpha* by CV on training folds.
    alphas = [float(a) for a in cfg["hybrid"]["alpha_grid"]]
    hybrid_runs = build_hybrid_runs(bm25_run, dense_run, alphas, k, cfg["hybrid"]["normalization"]["method"])
    m_alpha = {a: compute_ranking_metrics(qrels_d, hybrid_runs[a], METRICS) for a in alphas}
    alpha_ndcg = pd.DataFrame({a: m_alpha[a][PRIMARY] for a in alphas})
    oof_alpha, folds = select_alpha_cv(alpha_ndcg, cfg["k_folds"], seed)
    global_alpha = float(alpha_ndcg.mean(axis=0).idxmax())

    pq = pd.DataFrame(index=m_bm25.index)
    for metric in METRICS:
        short = metric.replace("ndcg_cut_10", "ndcg")
        pq[f"{short}_bm25"] = m_bm25[metric]
        pq[f"{short}_dense"] = m_dense[metric]
        pq[f"{short}_hybrid"] = [m_alpha[oof_alpha[q]].at[q, metric] for q in pq.index]
    pq["alpha_star_oof"] = oof_alpha
    # Sensitivity: alpha tuned on ALL queries (in-sample, maximally favourable to hybrid).
    pq["ndcg_hybrid_insample"] = alpha_ndcg[global_alpha]
    pq["best_alpha_per_query"] = alpha_ndcg.idxmax(axis=1)

    pq["delta_bm25"] = pq["ndcg_bm25"] - pq["ndcg_hybrid"]
    pq["delta_dense"] = pq["ndcg_dense"] - pq["ndcg_hybrid"]
    pq["delta_max_single"] = pq[["ndcg_bm25", "ndcg_dense"]].max(axis=1) - pq["ndcg_hybrid"]
    pq["degraded"] = pq["delta_max_single"] >= eps
    pq["partition"] = [classify_query(b, d, h, eps) for b, d, h in pq[["ndcg_bm25", "ndcg_dense", "ndcg_hybrid"]].values]

    pq = pq.join(compute_features(queries, corpus, bm25, bm25_run, dense_run))

    # How degradation and synergy move with alpha (each alpha applied to every query).
    curve = []
    max_single = pq[["ndcg_bm25", "ndcg_dense"]].max(axis=1)
    for a in alphas:
        h = alpha_ndcg[a]
        curve.append({
            "alpha": a,
            "mean_ndcg": float(h.mean()),
            "degraded_share": float((max_single - h >= eps).mean()),
            "hybrid_dominant_share": float((h - max_single >= eps).mean()),
        })
    alpha_curve = pd.DataFrame(curve)

    n_boot = args.n_bootstrap
    summary = {
        "dataset": dataset,
        "split": args.split,
        "n_docs": len(corpus),
        "n_queries": len(pq),
        "epsilon": eps,
        "k": k,
        "dense_model": args.dense_model or cfg["retrievers"]["dense"]["model_name"],
        "alpha_grid": alphas,
        "cv_folds": folds,
        "alpha_star_insample": global_alpha,
        "mean_metrics": {
            route: {m: float(pq[f"{m.replace('ndcg_cut_10', 'ndcg')}_{route}"].mean()) for m in METRICS}
            for route in ROUTES
        },
        "partition_shares": {p: float((pq["partition"] == p).mean()) for p in PARTITIONS},
        "partition_counts": {p: int((pq["partition"] == p).sum()) for p in PARTITIONS},
        "H1": h1_prevalence(pq, eps, "ndcg_hybrid"),
        "H1_insample_alpha": h1_prevalence(pq, eps, "ndcg_hybrid_insample"),
        "H2": h2_headroom(pq, "ndcg_hybrid", n_boot, seed),
        "H2_insample_alpha": h2_headroom(pq, "ndcg_hybrid_insample", n_boot, seed),
        "partition_tests_posthoc": partition_tests(pq, n_boot, seed),
        "H3": h3_diagnostics(pq, eps),
        "best_alpha_per_query_distribution": {
            str(a): float((pq["best_alpha_per_query"] == a).mean()) for a in alphas
        },
    }

    out_dir = args.output_dir / dataset
    out_dir.mkdir(parents=True, exist_ok=True)
    pq.to_csv(out_dir / "per_query.csv")
    alpha_curve.to_csv(out_dir / "alpha_sweep.csv", index=False)
    correlation_table(pq).to_csv(out_dir / "feature_correlations.csv", index=False)
    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=float)
    if not args.no_plots:
        make_figures(dataset, pq, alpha_curve, float(pd.Series(oof_alpha).mode()[0]), eps, out_dir / "figures")

    h1, h2 = summary["H1"], summary["H2"]
    log.info(
        f"[{dataset}] nDCG@10 BM25={summary['mean_metrics']['bm25'][PRIMARY]:.4f} "
        f"Dense={summary['mean_metrics']['dense'][PRIMARY]:.4f} Hybrid(CV)={h2['hybrid_ndcg']:.4f} "
        f"Oracle={h2['oracle3_ndcg']:.4f} | degraded={h1['prevalence']:.1%} | headroom={h2['headroom']:.4f}"
    )
    return pq, summary


def write_tables(summaries: List[dict], path: Path) -> None:
    fmt = lambda x: f"{x:.4f}"  # noqa: E731
    pct = lambda x: f"{x:.1%}"  # noqa: E731
    lines = [
        "# Experiment 1 results: fixed-α hybrid degradation",
        "",
        f"ε = {summaries[0]['epsilon']}; hybrid uses α* selected on training folds "
        f"({len(summaries[0]['cv_folds'])}-fold CV, out-of-fold scores). Metric: nDCG@10.",
        "",
        "## Query win/loss table",
        "",
        "| Dataset | Total Queries | Hybrid Wins (%) | BM25 Wins (%) | Dense Wins (%) | Near-tie, hybrid degraded (%) "
        "| Near-tie, hybrid safe (%) | Ties (%) | Total Degraded by Hybrid (%) [95% CI] |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]
    for s in summaries:
        p, h1 = s["partition_shares"], s["H1"]
        lo, hi = h1["prevalence_ci95"]
        lines.append(
            f"| {s['dataset']} | {s['n_queries']} | {pct(p['hybrid_dominant'])} | {pct(p['bm25_dominant'])} "
            f"| {pct(p['dense_dominant'])} | {pct(p['near_tie_degraded'])} | {pct(p['near_tie_safe'])} | {pct(p['tie'])} "
            f"| **{pct(h1['prevalence'])}** [{pct(lo)}, {pct(hi)}] |"
        )
    lines += [
        "",
        "## Oracle headroom table",
        "",
        "| Dataset | BM25 nDCG | Dense nDCG | Best Hybrid (α*) | Oracle Router | Headroom (Δ) [95% CI] | p (bootstrap) "
        "| p (Wilcoxon) | Oracle{BM25,Dense} − Hybrid |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]
    for s in summaries:
        h2, mm = s["H2"], s["mean_metrics"]
        t = h2["oracle3_vs_hybrid"]
        alphas = sorted({f["alpha_star"] for f in s["cv_folds"]})
        lines.append(
            f"| {s['dataset']} | {fmt(mm['bm25'][PRIMARY])} | {fmt(mm['dense'][PRIMARY])} "
            f"| {fmt(h2['hybrid_ndcg'])} (α*∈{{{', '.join(f'{a:g}' for a in alphas)}}}) | {fmt(h2['oracle3_ndcg'])} "
            f"| **{fmt(h2['headroom'])}** [{fmt(t['ci_95_low'])}, {fmt(t['ci_95_high'])}] | {t['p_value']:.1e} "
            f"| {t['wilcoxon_p_greater']:.1e} | {h2['oracle2_vs_hybrid']['delta_mean']:+.4f} |"
        )
    lines += [
        "",
        "## Hypothesis verdicts",
        "",
        "| Dataset | H1: degraded ≥ 15% (CI lower bound) | H2: headroom ≥ 0.02, p < 0.01 | H1 w/ in-sample α | H2 w/ in-sample α |",
        "| :--- | :---: | :---: | :---: | :---: |",
    ]
    yn = lambda b: "✅" if b else "❌"  # noqa: E731
    for s in summaries:
        lines.append(
            f"| {s['dataset']} | {yn(s['H1']['supported'])} ({pct(s['H1']['prevalence_ci95'][0])}) "
            f"| {yn(s['H2']['supported'])} | {yn(s['H1_insample_alpha']['supported'])} "
            f"| {yn(s['H2_insample_alpha']['supported'])} |"
        )
    lines += [
        "",
        "## H3 diagnostics (one-sided Mann-Whitney U, p-values)",
        "",
        "| Dataset | Max-IDF top quartile → larger Δ_BM25 | Low dense keyword coverage → larger Δ_Dense "
        "| Low BM25 top-gap → larger Δ_Dense | Jaccard@10 = 0 → larger Δ_MaxSingle |",
        "| :--- | :---: | :---: | :---: | :---: |",
    ]

    def cell(d):
        if d.get("p_value") is None:
            return "n/a"
        return f"{d['mean_a']:+.3f} vs {d['mean_b']:+.3f} (p={d['p_value']:.1e})"

    for s in summaries:
        h3 = s["H3"]
        lines.append(
            f"| {s['dataset']} | {cell(h3['H3a_maxidf_top_quartile_vs_rest__delta_bm25'])} "
            f"| {cell(h3['H3b_low_dense_keyword_coverage_vs_rest__delta_dense'])} "
            f"| {cell(h3['H3b_low_bm25_gap_vs_rest__delta_dense'])} "
            f"| {cell(h3['H3c_jaccard10_zero_vs_positive__delta_max_single'])} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    args = parse_args()
    exp_cfg = load_config(args.config_dir / "experiment.yaml")
    cfg = {
        "retrievers": load_config(args.config_dir / "retrievers.yaml"),
        "hybrid": load_config(args.config_dir / "hybrid.yaml"),
        "eps": args.epsilon if args.epsilon is not None else load_config(args.config_dir / "labels.yaml")["default_epsilon"],
        "seed": exp_cfg["seed"],
        "k_folds": exp_cfg["k_folds"],
    }
    seed_everything(cfg["seed"])
    datasets = args.datasets or exp_cfg["datasets"]

    all_pq, summaries = {}, []
    for ds in datasets:
        pq, summary = run_dataset(ds, args, cfg)
        all_pq[ds] = pq
        summaries.append(summary)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_tables(summaries, args.output_dir / "tables.md")
    if not args.no_plots and len(all_pq) > 1:
        make_pooled_cdf(all_pq, cfg["eps"], args.output_dir / "figures")
    log.info(f"Wrote {args.output_dir / 'tables.md'}")


if __name__ == "__main__":
    main()
