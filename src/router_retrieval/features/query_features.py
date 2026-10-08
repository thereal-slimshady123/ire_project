"""Query-intrinsic feature extraction."""

from typing import Dict, Any, Optional
from router_retrieval.data.schema import Query


def extract_query_features(
    query: Query,
    idf_vocab: Optional[Dict[str, float]] = None,
    corpus_vocab: Optional[set] = None,
) -> Dict[str, float]:
    """
    Extract intrinsic features from the raw query string.

    Features extracted:
        - query_length: Token count of query.
        - max_idf: Maximum IDF across tokens in the query.
        - mean_idf: Mean IDF across tokens in the query.
        - rare_term_density: Fraction of tokens with IDF above a threshold.
        - entity_density: Density of capitalized / named entities in query.
        - oov_fraction: Fraction of tokens out of corpus vocabulary.

    Args:
        query: Query object.
        idf_vocab: Mapping from term to corpus-derived IDF value.
        corpus_vocab: Set of known corpus terms.

    Returns:
        Dictionary of feature name to numeric value.
    """
    raise NotImplementedError
