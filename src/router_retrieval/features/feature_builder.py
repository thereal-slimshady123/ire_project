"""Feature builder orchestrating query and retrieval features with caching."""

from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional

import numpy as np
import pandas as pd

from router_retrieval.data.schema import Candidate, Document, Query
from router_retrieval.features.query_features import extract_query_features, rare_idf_threshold
from router_retrieval.features.retrieval_signal_features import (
    compute_query_term_coverage,
    compute_top_ratio,
    extract_retrieval_signal_features,
)
from router_retrieval.retrieval.bm25 import tokenize

# Ordered columns of the cached feature matrix (the router's input space).
QUERY_FEATURES = [
    "query_length", "max_idf", "mean_idf", "rare_term_density", "entity_density", "oov_fraction",
]
RETRIEVAL_SIGNAL_FEATURES = [
    "jaccard_overlap_top10", "jaccard_overlap_top20", "rank_correlation_spearman",
    "bm25_top1_score", "bm25_top1_top2_gap", "bm25_top1_top2_ratio", "bm25_score_entropy",
    "dense_top1_score", "dense_top1_top2_gap", "dense_score_entropy",
    "bm25_top10_query_term_coverage", "dense_top10_query_term_coverage",
]
FEATURE_COLUMNS = QUERY_FEATURES + RETRIEVAL_SIGNAL_FEATURES

# plan.md Contract 2 names that differ from the native column names. Both are written to disk so
# Contract-2 consumers and the router share one file; routers select columns by explicit list.
CONTRACT_ALIASES = {
    "jaccard_overlap": "jaccard_overlap_top10",
    "rank_correlation": "rank_correlation_spearman",
}


def document_term_set(doc: Document) -> set:
    """Analyzed (stopped + stemmed) term set of title + body."""
    return set(tokenize(f"{doc.title} {doc.text}" if doc.title else doc.text))


class FeatureBuilder:
    """Orchestrates query and retrieval feature extraction and caching."""

    def __init__(
        self,
        cache_dir: Optional[Path] = None,
        idf_vocab: Optional[Dict[str, float]] = None,
        corpus_vocab: Optional[set] = None,
        rare_threshold: Optional[float] = None,
        corpus: Optional[Mapping[str, Document]] = None,
    ):
        """
        Initialize FeatureBuilder.

        Args:
            cache_dir: Directory to cache feature vectors to disk.
            idf_vocab: Corpus IDF vocabulary dictionary (analyzed term -> BM25 IDF).
            corpus_vocab: Set of corpus terms.
            rare_threshold: IDF above which a term counts as rare (needed for rare_term_density).
            corpus: doc_id -> Document; enables the top-10 query-term coverage features.
        """
        self.cache_dir = cache_dir
        self.idf_vocab = idf_vocab
        self.corpus_vocab = corpus_vocab
        self.rare_threshold = rare_threshold
        self.corpus = corpus
        self._doc_terms: Dict[str, set] = {}

    @classmethod
    def from_bm25(cls, bm25, corpus: Optional[Mapping[str, Document]] = None,
                  cache_dir: Optional[Path] = None) -> "FeatureBuilder":
        """Build from an indexed BM25Retriever: its IDFs, vocabulary and corpus size define the statistics."""
        idf_vocab = {t: float(bm25.idf[j]) for t, j in bm25.vocab.items()}
        return cls(cache_dir=cache_dir, idf_vocab=idf_vocab, corpus_vocab=set(bm25.vocab),
                   rare_threshold=rare_idf_threshold(bm25.num_docs), corpus=corpus)

    def _terms_of(self, doc_id: str) -> set:
        if doc_id not in self._doc_terms:
            self._doc_terms[doc_id] = document_term_set(self.corpus[doc_id])
        return self._doc_terms[doc_id]

    def _coverage(self, query: Query, candidates: List[Candidate]) -> float:
        ranked = sorted(candidates, key=lambda c: c.rank)[:10]
        if self.corpus is None:
            return 0.0
        return compute_query_term_coverage(
            tokenize(query.text), [self._terms_of(c.doc_id) for c in ranked if c.doc_id in self.corpus]
        )

    def extract_features(
        self,
        query: Query,
        bm25_candidates: List[Candidate],
        dense_candidates: List[Candidate],
    ) -> Dict[str, float]:
        """
        Extract complete combined feature vector for a single query.

        Args:
            query: Query object.
            bm25_candidates: BM25 candidate list.
            dense_candidates: Dense candidate list.

        Returns:
            Dictionary of all feature names (FEATURE_COLUMNS) to float values.
        """
        features = extract_query_features(query, self.idf_vocab, self.corpus_vocab, self.rare_threshold)
        features.update(extract_retrieval_signal_features(bm25_candidates, dense_candidates))
        features["bm25_top1_top2_ratio"] = compute_top_ratio(bm25_candidates)
        features["bm25_top10_query_term_coverage"] = self._coverage(query, bm25_candidates)
        features["dense_top10_query_term_coverage"] = self._coverage(query, dense_candidates)
        return {k: float(features[k]) for k in FEATURE_COLUMNS}

    def build_feature_matrix(
        self,
        queries: Iterable[Query],
        bm25_run: Dict[str, List[Candidate]],
        dense_run: Dict[str, List[Candidate]],
    ) -> pd.DataFrame:
        """
        Build feature matrix across a collection of queries.

        Args:
            queries: Queries to featurize.
            bm25_run: Map of query_id to BM25 candidate list.
            dense_run: Map of query_id to Dense candidate list.

        Returns:
            DataFrame indexed by query_id with one column per entry of FEATURE_COLUMNS.
        """
        rows = []
        for query in queries:
            row = self.extract_features(query, bm25_run.get(query.query_id, []),
                                        dense_run.get(query.query_id, []))
            row["query_id"] = query.query_id
            rows.append(row)
        if not rows:
            return pd.DataFrame(columns=FEATURE_COLUMNS, index=pd.Index([], name="query_id", dtype=object))
        return pd.DataFrame(rows).set_index("query_id")[FEATURE_COLUMNS]

    # ------------------------------------------------------------------ persistence (Contract 2)

    @staticmethod
    def to_contract_frame(df: pd.DataFrame) -> pd.DataFrame:
        """
        Contract 2 layout: `query_id` string column + float32 features + Contract-2 alias columns.

        Aliases (`jaccard_overlap`, `rank_correlation`) duplicate their native columns; note the
        rank correlation is Spearman (not Kendall) and top-1 scores are raw retriever scores.
        """
        out = df.copy()
        if "query_id" not in out.columns:
            out = out.rename_axis("query_id").reset_index()
        out["query_id"] = out["query_id"].astype(str)
        for alias, native in CONTRACT_ALIASES.items():
            if native in out.columns and alias not in out.columns:
                out[alias] = out[native]
        float_cols = [c for c in out.columns if c != "query_id" and pd.api.types.is_numeric_dtype(out[c])]
        out[float_cols] = out[float_cols].astype(np.float32)
        return out

    def save_features(self, df: pd.DataFrame, path: Path) -> None:
        """Save feature matrix to Parquet (Contract 2 schema)."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.to_contract_frame(df).to_parquet(path, index=False)

    def load_features(self, path: Path) -> pd.DataFrame:
        """Load a feature Parquet file as a DataFrame indexed by string query_id."""
        df = pd.read_parquet(path)
        if "query_id" in df.columns:
            df["query_id"] = df["query_id"].astype(str)
            df = df.set_index("query_id")
        return df

    def cached_matrix(self, name: str, build) -> pd.DataFrame:
        """Return `<cache_dir>/<name>.parquet` if present, else `build()` it and cache the result."""
        if self.cache_dir is None:
            return build()
        path = Path(self.cache_dir) / f"{name}.parquet"
        if path.exists():
            return self.load_features(path)
        df = build()
        self.save_features(df, path)
        return df
