"""Offline fixtures for Track 2 tests: planted-signal route tables and a toy retrieval corpus.

Everything here runs in well under a second and needs no GPU, network or raw data.
"""

import numpy as np
import pandas as pd

from router_retrieval.data.schema import Document, Query
from router_retrieval.features.feature_builder import FEATURE_COLUMNS

ALPHAS = [round(0.1 * i, 1) for i in range(11)]


def make_route_table(n=240, seed=0, signal=True, name="toy"):
    """
    Route table in the Experiment 2 cache schema. With `signal`, a large BM25 top-1 gap means BM25 beats
    hybrid by 0.4 and a large dense gap means dense beats hybrid by 0.4 (so a router can profit); without
    it the per-route scores are pure noise around the hybrid score.
    """
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(rng.normal(size=(n, len(FEATURE_COLUMNS))), columns=FEATURE_COLUMNS)
    hyb = np.clip(rng.normal(0.5, 0.05, n), 0, 1)
    bm25 = np.clip(hyb + rng.normal(-0.05, 0.1, n), 0, 1)
    dense = np.clip(hyb + rng.normal(-0.05, 0.1, n), 0, 1)
    if signal:
        bm25 = np.clip(bm25 + 0.45 * (X["bm25_top1_top2_gap"] > 0.6).to_numpy(), 0, 1)
        dense = np.clip(dense + 0.45 * (X["dense_top1_top2_gap"] > 0.6).to_numpy(), 0, 1)
    df = pd.DataFrame({"dataset": name, "ndcg_bm25": bm25, "ndcg_dense": dense}, index=[f"q{i}" for i in range(n)])
    for a in ALPHAS:  # alpha=0.5 is best on average; others a bit worse
        df[f"ndcg_hyb_{a}"] = np.clip(hyb - 0.04 * abs(a - 0.5), 0, 1)
    X.index = df.index
    df = pd.concat([df, X], axis=1)
    df.index.name = "query_id"
    return df


TOY_DOCS = {
    "d1": ("Aspirin", "Acetylsalicylic acid inhibits cyclooxygenase and reduces platelet aggregation."),
    "d2": ("Metformin", "Metformin lowers hepatic glucose production by activating AMPK in type 2 diabetes."),
    "d3": ("BRCA1", "BRCA1 mutations raise lifetime risk of breast and ovarian cancer substantially."),
    "d4": ("Kubernetes", "Kubernetes error 137 means the container was OOMKilled after exceeding memory limits."),
    "d5": ("Rust", "Rust ownership and borrowing guarantee memory safety without a garbage collector."),
    "d6": ("Redis", "Redis append only file persistence trades write throughput for durability on crashes."),
    "d7": ("Quantum", "Quantum key distribution with BB84 uses photon polarization to exchange secret keys."),
    "d8": ("Graphene", "Graphene conducts heat exceptionally well because phonons travel long distances."),
    "d9": ("Stars", "Red giant stars fuse helium into carbon and oxygen through the triple alpha process."),
    "d10": ("Git", "A detached HEAD state appears when you checkout a commit instead of a branch."),
    "d11": ("Short", "too short"),
}


def toy_corpus():
    return {d: Document(d, text, title) for d, (title, text) in TOY_DOCS.items()}


def toy_queries():
    return {
        "t1": Query("t1", "aspirin platelet"),
        "t2": Query("t2", "metformin glucose AMPK diabetes"),
        "t3": Query("t3", "kubernetes OOMKilled memory limit"),
    }
