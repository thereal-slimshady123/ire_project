"""Query-intrinsic feature extraction."""

import math
import re
from typing import Dict, Optional

import numpy as np

from router_retrieval.data.schema import Query
from router_retrieval.retrieval.bm25 import LUCENE_STOPWORDS, tokenize

_RAW_TOKEN_RE = re.compile(r"\w+", re.UNICODE)


def rare_idf_threshold(num_docs: int, max_df_fraction: float = 0.005) -> float:
    """Lucene BM25 IDF of a term occurring in `max_df_fraction` of the corpus (min df = 1)."""
    df = max(1.0, max_df_fraction * num_docs)
    return math.log(1.0 + (num_docs - df + 0.5) / (df + 0.5))


def _is_entity_like(token: str, position: int) -> bool:
    """Heuristic named-entity / identifier test: digits, ALLCAPS acronyms, or mid-query Capitalization."""
    if any(ch.isdigit() for ch in token):
        return True
    if len(token) >= 2 and token.isupper():
        return True
    return position > 0 and token[:1].isupper()


def extract_query_features(
    query: Query,
    idf_vocab: Optional[Dict[str, float]] = None,
    corpus_vocab: Optional[set] = None,
    rare_threshold: Optional[float] = None,
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

    IDF statistics are taken over in-vocabulary analyzed (stopped + stemmed) terms; OOV terms
    contribute nothing to BM25 and are reported separately via oov_fraction.

    Args:
        query: Query object.
        idf_vocab: Mapping from analyzed term to corpus-derived IDF value.
        corpus_vocab: Set of known analyzed corpus terms (defaults to idf_vocab keys).
        rare_threshold: IDF above which a term counts as rare (see rare_idf_threshold).

    Returns:
        Dictionary of feature name to numeric value.
    """
    idf_vocab = idf_vocab or {}
    corpus_vocab = corpus_vocab if corpus_vocab is not None else set(idf_vocab)
    raw_tokens = _RAW_TOKEN_RE.findall(query.text)
    content_tokens = [(i, t) for i, t in enumerate(raw_tokens) if t.lower() not in LUCENE_STOPWORDS]
    terms = tokenize(query.text)

    idfs = np.array([idf_vocab[t] for t in terms if t in idf_vocab], dtype=np.float64)
    n_terms = len(terms)
    feats = {
        "query_length": float(len(raw_tokens)),
        "max_idf": float(idfs.max()) if idfs.size else 0.0,
        "mean_idf": float(idfs.mean()) if idfs.size else 0.0,
        "rare_term_density": (
            float((idfs > rare_threshold).sum() / n_terms) if (n_terms and rare_threshold is not None) else 0.0
        ),
        "entity_density": (
            sum(_is_entity_like(t, i) for i, t in content_tokens) / len(content_tokens) if content_tokens else 0.0
        ),
        "oov_fraction": (sum(t not in corpus_vocab for t in terms) / n_terms) if n_terms else 0.0,
    }
    return {k: float(v) for k, v in feats.items()}
