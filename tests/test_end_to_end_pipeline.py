"""Integration test running full pipeline on synthetic mini dataset."""

import pytest


def test_end_to_end_wiring():
    """Test full pipeline wiring on synthetic mini dataset from query to final candidates."""
    import numpy as np
    from router_retrieval.data.schema import Candidate, Document, Query
    from router_retrieval.pipeline.end_to_end import EndToEndPipeline
    from router_retrieval.router.escalation import EscalationDecider

    class Ret:
        def __init__(self, doc): self.doc = doc
        def search(self, q, k): return [Candidate(q.query_id, self.doc, 1.0, 1)]
    class Hybrid:
        def combine_candidates(self, b, d, k): return (b + d)[:k]
    class Features:
        def extract_features(self, q, b, d): return {"f": 1.0}
    class Router:
        def predict_proba(self, X): return np.array([[.9, .05, .05]])
    class CE:
        candidate_pool_size = 2
    p = EndToEndPipeline(Ret("b"), Ret("d"), Hybrid(), CE(), Features(), Router(),
                         EscalationDecider(.5), {"b": Document("b", "b"), "d": Document("d", "d")})
    rows, route, escalated = p.process_query(Query("q", "text"))
    assert route == "bm25" and not escalated and rows[0].doc_id == "b"


def test_end_to_end_escalation_path():
    """Test that low-confidence query correctly routes through cross-encoder escalation."""
    import numpy as np
    from router_retrieval.data.schema import Candidate, Document, Query
    from router_retrieval.pipeline.end_to_end import EndToEndPipeline
    from router_retrieval.router.escalation import EscalationDecider

    class Ret:
        def __init__(self, doc): self.doc = doc
        def search(self, q, k): return [Candidate(q.query_id, self.doc, 1.0, 1)]
    class Hybrid:
        def combine_candidates(self, b, d, k): return (b + d)[:k]
    class Features:
        def extract_features(self, q, b, d): return {"f": 1.0}
    class Router:
        def predict_proba(self, X): return np.array([[.4, .35, .25]])
    class CE:
        candidate_pool_size = 2
        def rerank(self, q, candidates, corpus, top_k): return candidates[:top_k]
    p = EndToEndPipeline(Ret("b"), Ret("d"), Hybrid(), CE(), Features(), Router(),
                         EscalationDecider(.5), {"b": Document("b", "b"), "d": Document("d", "d")})
    _, route, escalated = p.process_query(Query("q", "text"))
    assert route == "escalate" and escalated
