"""Dual-style synthetic query generators.

Styles (improvements.md section 3.2):
  lexical   specific entities, identifiers, technical names -> favours BM25
  semantic  conceptual / paraphrastic questions that avoid the passage's vocabulary -> favours dense

A generator is any object with `generate(passages: Sequence[str], style: str) -> List[str]`.
`ExtractiveLexicalGenerator` is model-free; `Seq2SeqGenerator` prompts an instruction-tuned seq2seq model
(default: flan-t5-base) and is the only component that needs `transformers` / `torch`.
"""

import re
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np

from router_retrieval.retrieval.bm25 import tokenize

STYLES = ("lexical", "semantic")

# Identifier-preserving surface tokens: "CRISPR-Cas9", "HTTP/3", "memory.max", "0.04".
_SURFACE_TOKEN_RE = re.compile(r"\w+(?:[-./]\w+)*", re.UNICODE)


@dataclass
class SyntheticQuery:
    """A generated query whose source passage is its single relevant document."""
    query_id: str
    text: str
    source_doc: str
    style: str
    dataset: str = ""


class ExtractiveLexicalGenerator:
    """
    Keyword / identifier-style query: the passage's rarest surface tokens, kept in passage order.

    Tokens are scored by the BM25 IDF of their analyzed form (`idf_fn`); tokens the corpus has never seen
    (error codes, gene symbols) get `oov_idf`. Digit-bearing and ALLCAPS tokens receive `entity_bonus`
    so identifiers outrank ordinary rare words. Deterministic given the seed.
    """

    def __init__(
        self,
        idf_fn: Callable[[str], Optional[float]],
        min_terms: int = 3,
        max_terms: int = 6,
        oov_idf: float = 12.0,
        entity_bonus: float = 1.0,
        seed: int = 42,
    ):
        if not 1 <= min_terms <= max_terms:
            raise ValueError("require 1 <= min_terms <= max_terms")
        self.idf_fn, self.min_terms, self.max_terms = idf_fn, min_terms, max_terms
        self.oov_idf, self.entity_bonus = oov_idf, entity_bonus
        self.rng = np.random.default_rng(seed)

    def _score(self, token: str) -> Optional[float]:
        analyzed = tokenize(token)
        if not analyzed:  # stopword / empty after analysis
            return None
        idf = self.idf_fn(analyzed[0])
        score = self.oov_idf if idf is None else idf
        if any(ch.isdigit() for ch in token) or (len(token) >= 2 and token.isupper()):
            score += self.entity_bonus
        return score

    def generate(self, passages: Sequence[str], style: str = "lexical") -> List[str]:
        if style != "lexical":
            raise ValueError("ExtractiveLexicalGenerator only produces the 'lexical' style")
        out = []
        for text in passages:
            scored, seen = [], set()
            for pos, tok in enumerate(_SURFACE_TOKEN_RE.findall(text)):
                key = tok.lower()
                s = self._score(tok)
                if s is not None and key not in seen:
                    seen.add(key)
                    scored.append((s, pos, tok))
            n = int(self.rng.integers(self.min_terms, self.max_terms + 1))
            best = sorted(scored, key=lambda t: (-t[0], t[1]))[:n]
            out.append(" ".join(tok for _, _, tok in sorted(best, key=lambda t: t[1])))
        return out


class Seq2SeqGenerator:
    """Prompted seq2seq query generator (flan-t5 family). Loaded lazily; needs `transformers` + `torch`."""

    PROMPTS: Dict[str, str] = {
        "semantic": (
            "Write a natural question that a person who has not read the passage might ask and that the "
            "passage answers. Paraphrase the concepts; do not copy the passage's distinctive words.\n\n"
            "Passage: {passage}\n\nQuestion:"
        ),
        "lexical": (
            "Write a short keyword search query of 3 to 6 specific terms (entities, identifiers, technical "
            "names) that the passage answers.\n\nPassage: {passage}\n\nQuery:"
        ),
    }

    def __init__(
        self,
        model_name: str = "google/flan-t5-base",
        device: str = "auto",
        batch_size: int = 32,
        max_input_tokens: int = 384,
        max_new_tokens: int = 32,
        temperature: float = 0.8,
        top_p: float = 0.95,
        seed: int = 42,
    ):
        self.model_name, self.device_spec = model_name, device
        self.batch_size, self.max_input_tokens, self.max_new_tokens = batch_size, max_input_tokens, max_new_tokens
        self.temperature, self.top_p, self.seed = temperature, top_p, seed
        self._model = self._tok = self._torch = None

    def _load(self):
        if self._model is not None:
            return
        try:
            import torch
            from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
        except ImportError as e:  # pragma: no cover - exercised only with the optional ML stack
            raise ImportError("Seq2SeqGenerator needs `torch` and `transformers` (pip install transformers)") from e
        dev = self.device_spec
        if dev == "auto":
            dev = "cuda" if torch.cuda.is_available() else "cpu"
        torch.manual_seed(self.seed)
        self._torch = torch
        self._tok = AutoTokenizer.from_pretrained(self.model_name)
        self._model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name).to(dev).eval()
        self.device = dev

    def generate(self, passages: Sequence[str], style: str) -> List[str]:  # pragma: no cover - needs model weights
        if style not in self.PROMPTS:
            raise ValueError(f"unknown style {style!r}")
        self._load()
        out: List[str] = []
        for i in range(0, len(passages), self.batch_size):
            prompts = [self.PROMPTS[style].format(passage=p) for p in passages[i:i + self.batch_size]]
            enc = self._tok(prompts, return_tensors="pt", padding=True, truncation=True,
                            max_length=self.max_input_tokens).to(self.device)
            with self._torch.no_grad():
                ids = self._model.generate(**enc, max_new_tokens=self.max_new_tokens, do_sample=True,
                                           temperature=self.temperature, top_p=self.top_p)
            out.extend(s.strip() for s in self._tok.batch_decode(ids, skip_special_tokens=True))
        return out


class MixedGenerator:
    """Routes each style to its own generator, so lexical can be model-free and semantic model-based."""

    def __init__(self, lexical, semantic):
        self.by_style = {"lexical": lexical, "semantic": semantic}

    def generate(self, passages: Sequence[str], style: str) -> List[str]:
        return self.by_style[style].generate(passages, style)
