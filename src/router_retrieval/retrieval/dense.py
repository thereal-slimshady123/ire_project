"""Dense embedding retrieval using SentenceTransformers and FAISS."""

import json
from pathlib import Path
from typing import List, Dict, Optional, Sequence

import numpy as np

from router_retrieval.data.schema import Document, Candidate, Query
from router_retrieval.retrieval.bm25 import document_text


def resolve_device(device: str) -> str:
    """Map 'auto' to 'cuda' if a CUDA device is visible to torch, else 'cpu'."""
    if device != "auto":
        return device
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


class DenseRetriever:
    """Dense retriever using bi-encoder representations and FAISS CPU index."""

    def __init__(
        self,
        model_name: str = "sentence-transformers/msmarco-distilbert-base-v4",
        batch_size: int = 64,
        device: str = "cpu",
        faiss_index_type: str = "IndexFlatIP",
        max_length: Optional[int] = None,
        query_prefix: str = "",
        passage_prefix: str = "",
    ):
        """
        Initialize dense retriever configuration ('auto' device picks CUDA when available).

        query_prefix / passage_prefix are prepended before encoding, for models trained with
        instruction prefixes (e.g. E5: "query: " / "passage: ").
        """
        self.model_name = model_name
        self.batch_size = batch_size
        self.device = resolve_device(device)
        self.faiss_index_type = faiss_index_type
        self.max_length = max_length
        self.query_prefix = query_prefix
        self.passage_prefix = passage_prefix
        self.doc_ids: List[str] = []
        self._model = None
        self._index = None

    @property
    def model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_name, device=self.device)
            if self.max_length:
                self._model.max_seq_length = min(self.max_length, self._model.max_seq_length)
        return self._model

    def encode(self, texts: Sequence[str], show_progress_bar: bool = False) -> np.ndarray:
        """Encode texts into L2-normalized float32 embeddings (inner product == cosine)."""
        emb = self.model.encode(
            list(texts),
            batch_size=self.batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=show_progress_bar,
        )
        return np.ascontiguousarray(emb, dtype=np.float32)

    def _build_index(self, embeddings: np.ndarray) -> None:
        import faiss

        if self.faiss_index_type != "IndexFlatIP":
            raise ValueError(f"Unsupported FAISS index type: {self.faiss_index_type}")
        self._index = faiss.IndexFlatIP(embeddings.shape[1])
        self._index.add(embeddings)

    def index(self, corpus: Dict[str, Document], show_progress_bar: bool = True) -> None:
        """
        Encode corpus passages and build FAISS index.

        Args:
            corpus: Dict of doc_id to Document.
        """
        self.doc_ids = list(corpus.keys())
        embeddings = self.encode(
            [self.passage_prefix + document_text(corpus[d]) for d in self.doc_ids], show_progress_bar
        )
        self._build_index(embeddings)

    def search(self, query: Query, k: int = 100) -> List[Candidate]:
        """
        Encode query and retrieve top-k nearest neighbor candidates from FAISS index.

        Args:
            query: Query dataclass.
            k: Number of candidates to retrieve.

        Returns:
            List of Candidate objects sorted by score descending.
        """
        return self.batch_search([query], k)[query.query_id]

    def batch_search(self, queries: Sequence[Query], k: int = 100) -> Dict[str, List[Candidate]]:
        """Batch search for multiple queries."""
        q_emb = self.encode([self.query_prefix + q.text for q in queries])
        scores, idx = self._index.search(q_emb, min(k, len(self.doc_ids)))
        out: Dict[str, List[Candidate]] = {}
        for q, row_s, row_i in zip(queries, scores, idx):
            pairs = sorted(
                ((float(s), self.doc_ids[i]) for s, i in zip(row_s, row_i) if i >= 0),
                key=lambda p: (-p[0], p[1]),
            )
            out[q.query_id] = [
                Candidate(query_id=q.query_id, doc_id=d, score=s, rank=r + 1) for r, (s, d) in enumerate(pairs)
            ]
        return out

    def save(self, path: Path) -> None:
        """Persist corpus embeddings and doc_id mapping to disk (path is a directory)."""
        import faiss

        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        embeddings = faiss.rev_swig_ptr(self._index.get_xb(), self._index.ntotal * self._index.d)
        np.save(path / "embeddings.npy", np.asarray(embeddings).reshape(self._index.ntotal, self._index.d))
        with open(path / "meta.json", "w", encoding="utf-8") as f:
            json.dump(
                {"model_name": self.model_name, "passage_prefix": self.passage_prefix, "doc_ids": self.doc_ids}, f
            )

    @classmethod
    def load(cls, path: Path, model_name: Optional[str] = None, **kwargs) -> "DenseRetriever":
        """Load FAISS index and doc_id mapping from disk."""
        path = Path(path)
        with open(path / "meta.json", encoding="utf-8") as f:
            meta = json.load(f)
        if model_name and model_name != meta["model_name"]:
            raise ValueError(f"Index at {path} was built with {meta['model_name']}, not {model_name}")
        kwargs.setdefault("passage_prefix", meta.get("passage_prefix", ""))
        if kwargs["passage_prefix"] != meta.get("passage_prefix", ""):
            raise ValueError(f"Index at {path} was built with passage prefix {meta.get('passage_prefix')!r}")
        obj = cls(model_name=meta["model_name"], **kwargs)
        obj.doc_ids = meta["doc_ids"]
        obj._build_index(np.load(path / "embeddings.npy"))
        return obj
