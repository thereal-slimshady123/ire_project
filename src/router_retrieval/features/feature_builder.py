"""Feature builder orchestrating query and retrieval features with caching."""

from pathlib import Path
from typing import Dict, List, Optional
import pandas as pd
from router_retrieval.data.schema import Query, Candidate


class FeatureBuilder:
    """Orchestrates query and retrieval feature extraction and caching."""

    def __init__(
        self,
        cache_dir: Optional[Path] = None,
        idf_vocab: Optional[Dict[str, float]] = None,
        corpus_vocab: Optional[set] = None,
    ):
        """
        Initialize FeatureBuilder.

        Args:
            cache_dir: Directory to cache feature vectors to disk.
            idf_vocab: Corpus IDF vocabulary dictionary.
            corpus_vocab: Set of corpus terms.
        """
        self.cache_dir = cache_dir
        self.idf_vocab = idf_vocab
        self.corpus_vocab = corpus_vocab

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
            Dictionary of all feature names to float values.
        """
        raise NotImplementedError

    def build_feature_matrix(
        self,
        queries: List[Query],
        bm25_run: Dict[str, List[Candidate]],
        dense_run: Dict[str, List[Candidate]],
    ) -> pd.DataFrame:
        """
        Build feature matrix across a collection of queries.

        Args:
            queries: List of queries.
            bm25_run: Map of query_id to BM25 candidate list.
            dense_run: Map of query_id to Dense candidate list.

        Returns:
            DataFrame where index or column is query_id, containing all feature columns.
        """
        raise NotImplementedError

    def save_features(self, df: pd.DataFrame, path: Path) -> None:
        """Save feature matrix to Parquet file."""
        raise NotImplementedError

    def load_features(self, path: Path) -> pd.DataFrame:
        """Load feature matrix from Parquet file."""
        raise NotImplementedError
