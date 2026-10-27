"""Feature builder orchestrating query and retrieval features with caching."""

from pathlib import Path
from typing import Dict, List, Optional
import pandas as pd
from router_retrieval.data.schema import Query, Candidate
from router_retrieval.features.query_features import extract_query_features
from router_retrieval.features.retrieval_signal_features import extract_retrieval_signal_features


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
        features = extract_query_features(query, self.idf_vocab, self.corpus_vocab)
        features.update(extract_retrieval_signal_features(bm25_candidates, dense_candidates))
        return {k: float(v) for k, v in features.items()}

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
        rows = []
        for query in queries:
            row = self.extract_features(query, bm25_run.get(query.query_id, []),
                                        dense_run.get(query.query_id, []))
            row["query_id"] = query.query_id
            rows.append(row)
        return pd.DataFrame(rows).set_index("query_id") if rows else pd.DataFrame(index=pd.Index([], name="query_id"))

    def save_features(self, df: pd.DataFrame, path: Path) -> None:
        """Save feature matrix to Parquet file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(path)

    def load_features(self, path: Path) -> pd.DataFrame:
        """Load feature matrix from Parquet file."""
        return pd.read_parquet(path)
