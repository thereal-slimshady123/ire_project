"""Passage sampling, style assignment and leakage control for synthetic query generation."""

from typing import Dict, Iterable, List, Mapping, Sequence, Set, Tuple

import numpy as np

from router_retrieval.data.schema import Document, QRel
from router_retrieval.retrieval.bm25 import document_text

_WS = " ".join


def excluded_doc_ids(qrels: Iterable[QRel], min_relevance: int = 1) -> Set[str]:
    """Documents judged relevant for ANY natural evaluation query: forbidden as synthetic sources."""
    return {q.doc_id for q in qrels if q.relevance >= min_relevance}


def eligible_doc_ids(
    corpus: Mapping[str, Document],
    excluded: Set[str],
    min_words: int = 25,
    max_words: int = 400,
) -> List[str]:
    """Sorted ids of passages usable as synthetic sources: not excluded, long enough, not absurdly long."""
    return sorted(
        d for d, doc in corpus.items()
        if d not in excluded and min_words <= len(document_text(doc).split()) <= max_words
    )


def style_capacity(n_eligible: int, lexical_fraction: float = 0.5) -> int:
    """Largest total query count when each passage may serve each style at most once."""
    return int(n_eligible / max(lexical_fraction, 1.0 - lexical_fraction, 1e-9))


def plan_queries(
    eligible: Sequence[str],
    n_queries: int,
    lexical_fraction: float = 0.5,
    seed: int = 42,
) -> List[Tuple[str, str]]:
    """
    Choose (source_doc, style) pairs: `lexical_fraction` lexical, the rest semantic.

    Passages are drawn without replacement within a style, so a passage appears at most once per style
    (twice overall). Deterministic given the seed.
    """
    if n_queries > style_capacity(len(eligible), lexical_fraction):
        raise ValueError(f"requested {n_queries} queries but only {style_capacity(len(eligible), lexical_fraction)} "
                         f"fit {len(eligible)} eligible passages")
    n_lex = int(round(n_queries * lexical_fraction))
    counts = {"lexical": n_lex, "semantic": n_queries - n_lex}
    rng = np.random.default_rng(seed)
    plan: List[Tuple[str, str]] = []
    for style in ("lexical", "semantic"):
        picks = rng.choice(len(eligible), size=counts[style], replace=False)
        plan.extend((eligible[i], style) for i in sorted(picks))
    return plan


def allocate_budget(capacities: Mapping[str, int], total: int) -> Dict[str, int]:
    """Split `total` queries equally across datasets, capped at each capacity; leftovers are redistributed."""
    alloc = {k: 0 for k in capacities}
    remaining = min(total, sum(capacities.values()))
    open_keys = [k for k in capacities if capacities[k] > 0]
    while remaining > 0 and open_keys:
        share, extra = divmod(remaining, len(open_keys))
        progressed = 0
        for j, k in enumerate(list(open_keys)):
            want = share + (1 if j < extra else 0)
            give = min(want, capacities[k] - alloc[k])
            alloc[k] += give
            progressed += give
            if alloc[k] >= capacities[k]:
                open_keys.remove(k)
        remaining -= progressed
        if progressed == 0:
            break
    return alloc


def _norm(text: str) -> str:
    return _WS(text.lower().split())


def clean_queries(
    texts: Sequence[str],
    natural_texts: Iterable[str] = (),
    min_tokens: int = 2,
) -> List[bool]:
    """
    Keep-mask over generated texts: drop empty / too-short queries, queries identical to a natural
    evaluation query, and exact duplicates of an earlier generated query.
    """
    banned = {_norm(t) for t in natural_texts}
    seen: Set[str] = set()
    keep = []
    for t in texts:
        n = _norm(t)
        ok = len(n.split()) >= min_tokens and n not in banned and n not in seen
        if ok:
            seen.add(n)
        keep.append(ok)
    return keep


def assert_no_leakage(source_docs: Iterable[str], excluded: Set[str]) -> None:
    """Raise if any synthetic source passage is relevant to a natural evaluation query."""
    leaked = sorted(set(source_docs) & excluded)
    if leaked:
        raise ValueError(f"{len(leaked)} synthetic queries are sourced from evaluation-relevant passages, "
                         f"e.g. {leaked[:5]}")
