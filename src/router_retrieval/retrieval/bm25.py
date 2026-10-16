"""BM25 sparse retrieval implementation.

Scores are computed exactly with a precomputed sparse doc-term weight matrix (scipy),
using Lucene's BM25 IDF and an analyzer approximating Lucene's EnglishAnalyzer
(lowercase, Lucene English stopwords, Porter stemming) so results are comparable to
Pyserini/Elasticsearch BM25 baselines reported on BEIR.
"""

import pickle
import re
from collections import Counter
from functools import lru_cache
from pathlib import Path
from typing import List, Dict, Optional, Sequence

import numpy as np
from scipy import sparse

from router_retrieval.data.schema import Document, Candidate, Query

# Lucene EnglishAnalyzer default stopword set.
LUCENE_STOPWORDS = frozenset(
    "a an and are as at be but by for if in into is it no not of on or such that the "
    "their then there these they this to was will with".split()
)
_TOKEN_RE = re.compile(r"\w+", re.UNICODE)

try:
    from nltk.stem import PorterStemmer

    _stemmer = PorterStemmer()

    @lru_cache(maxsize=None)
    def _stem(token: str) -> str:
        return _stemmer.stem(token)

except ImportError:  # pragma: no cover - stemming is optional

    def _stem(token: str) -> str:
        return token


def tokenize(text: str, stem: bool = True) -> List[str]:
    """Lowercase, split on word characters, drop stopwords, and optionally Porter-stem."""
    tokens = [t for t in _TOKEN_RE.findall(text.lower()) if t not in LUCENE_STOPWORDS]
    return [_stem(t) for t in tokens] if stem else tokens


def document_text(doc: Document) -> str:
    """BEIR convention: index title and body together."""
    return f"{doc.title} {doc.text}" if doc.title else doc.text


class BM25Retriever:
    """BM25 retrieval engine."""

    def __init__(self, k1: float = 1.2, b: float = 0.75):
        """Initialize BM25 retriever with tuning parameters."""
        self.k1 = k1
        self.b = b
        self.corpus: Optional[Dict[str, Document]] = None
        self.doc_ids: List[str] = []
        self.vocab: Dict[str, int] = {}
        self.doc_freq: Optional[np.ndarray] = None
        self.idf: Optional[np.ndarray] = None
        self._weights: Optional[sparse.csc_matrix] = None

    @property
    def num_docs(self) -> int:
        return len(self.doc_ids)

    def index(self, corpus: Dict[str, Document]) -> None:
        """
        Build BM25 index from document collection.

        Args:
            corpus: Dict of doc_id to Document.
        """
        self.corpus = corpus
        self.doc_ids = list(corpus.keys())
        rows, cols, vals = [], [], []
        doc_lens = np.zeros(len(self.doc_ids), dtype=np.float64)
        for i, doc_id in enumerate(self.doc_ids):
            counts = Counter(tokenize(document_text(corpus[doc_id])))
            doc_lens[i] = sum(counts.values())
            for term, tf in counts.items():
                j = self.vocab.setdefault(term, len(self.vocab))
                rows.append(i)
                cols.append(j)
                vals.append(tf)

        n = len(self.doc_ids)
        tf = sparse.csr_matrix(
            (np.asarray(vals, dtype=np.float64), (rows, cols)), shape=(n, len(self.vocab))
        )
        self.doc_freq = np.bincount(tf.indices, minlength=len(self.vocab)).astype(np.float64)
        self.idf = np.log(1.0 + (n - self.doc_freq + 0.5) / (self.doc_freq + 0.5))

        avgdl = doc_lens.mean() if n else 0.0
        norm = self.k1 * (1.0 - self.b + self.b * doc_lens / max(avgdl, 1e-9))
        # Per-nonzero BM25 term weight: idf * tf * (k1 + 1) / (tf + norm_d)
        row_of_nnz = np.repeat(np.arange(n), np.diff(tf.indptr))
        tf.data = self.idf[tf.indices] * tf.data * (self.k1 + 1.0) / (tf.data + norm[row_of_nnz])
        self._weights = tf.tocsc()

    def term_idf(self, term: str) -> Optional[float]:
        """IDF of an analyzed term, or None if the term is out of vocabulary."""
        j = self.vocab.get(term)
        return None if j is None else float(self.idf[j])

    def score_all(self, query_text: str) -> np.ndarray:
        """Return BM25 scores of every document for the query."""
        counts = Counter(t for t in tokenize(query_text) if t in self.vocab)
        if not counts:
            return np.zeros(self.num_docs)
        cols = [self.vocab[t] for t in counts]
        qtf = np.asarray(list(counts.values()), dtype=np.float64)
        return np.asarray(self._weights[:, cols] @ qtf).ravel()

    def search(self, query: Query, k: int = 100) -> List[Candidate]:
        """
        Retrieve top-k candidates for a given query.

        Documents with zero score (no query-term match) are never returned, so the list
        may be shorter than k.

        Args:
            query: Query dataclass.
            k: Number of candidates to retrieve.

        Returns:
            List of Candidate objects sorted by score descending.
        """
        scores = self.score_all(query.text)
        positive = np.flatnonzero(scores > 0)
        if positive.size > k:
            positive = positive[np.argpartition(-scores[positive], k - 1)[:k]]
        # Sort by score desc, doc_id asc for deterministic tie-breaking.
        order = sorted(positive.tolist(), key=lambda i: (-scores[i], self.doc_ids[i]))
        return [
            Candidate(query_id=query.query_id, doc_id=self.doc_ids[i], score=float(scores[i]), rank=r + 1)
            for r, i in enumerate(order)
        ]

    def batch_search(self, queries: Sequence[Query], k: int = 100) -> Dict[str, List[Candidate]]:
        """Batch search for multiple queries."""
        return {q.query_id: self.search(q, k) for q in queries}

    def save(self, path: Path) -> None:
        """Persist BM25 index to disk."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        state = {k: v for k, v in self.__dict__.items() if k != "corpus"}
        with open(path, "wb") as f:
            pickle.dump(state, f)

    @classmethod
    def load(cls, path: Path) -> "BM25Retriever":
        """Load BM25 index from disk."""
        with open(path, "rb") as f:
            state = pickle.load(f)
        obj = cls(k1=state["k1"], b=state["b"])
        obj.__dict__.update(state)
        return obj
