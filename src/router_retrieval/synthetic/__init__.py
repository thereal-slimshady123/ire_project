"""Synthetic training-query generation for the advantage router (improvements.md section 3)."""

from router_retrieval.synthetic.generator import (
    ExtractiveLexicalGenerator,
    MixedGenerator,
    Seq2SeqGenerator,
    STYLES,
    SyntheticQuery,
)
from router_retrieval.synthetic.sampler import (
    allocate_budget,
    assert_no_leakage,
    clean_queries,
    eligible_doc_ids,
    excluded_doc_ids,
    plan_queries,
    style_capacity,
)
from router_retrieval.synthetic.builder import (
    build_route_table,
    load_synthetic,
    save_synthetic,
    subsample_synthetic,
)

__all__ = [
    "STYLES", "SyntheticQuery", "ExtractiveLexicalGenerator", "Seq2SeqGenerator", "MixedGenerator",
    "excluded_doc_ids", "eligible_doc_ids", "plan_queries", "style_capacity", "allocate_budget",
    "clean_queries", "assert_no_leakage",
    "build_route_table", "save_synthetic", "load_synthetic", "subsample_synthetic",
]
