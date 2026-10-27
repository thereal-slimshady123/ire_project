"""Unit tests for escalation thresholding."""

import pytest


def test_escalation_boundary_conditions():
    """Test confidence exactly equal, below, and above tau."""
    import numpy as np
    from router_retrieval.router.escalation import EscalationDecider
    decider = EscalationDecider(tau=0.6)
    decisions, escalated, conf = decider.decide(np.array([[.6, .2, .2], [.59, .21, .20], [.8, .1, .1]]))
    assert decisions == ["bm25", "escalate", "bm25"]
    assert escalated == [False, True, False]
    assert np.allclose(conf, [.6, .59, .8])


def test_all_escalate_condition():
    """Test extreme high tau causes all queries to escalate."""
    import numpy as np
    from router_retrieval.router.escalation import EscalationDecider
    assert EscalationDecider(tau=1.01).decide(np.eye(3))[1] == [True, True, True]


def test_no_escalate_condition():
    """Test extreme low tau causes zero queries to escalate."""
    import numpy as np
    from router_retrieval.router.escalation import EscalationDecider
    assert EscalationDecider(tau=0).decide(np.eye(3))[1] == [False, False, False]


def test_calibrated_tau_is_deterministic_and_respects_ties():
    import numpy as np
    from router_retrieval.router.escalation import EscalationDecider
    p = np.array([[.8, .1, .1], [.8, .1, .1], [.4, .3, .3], [.5, .3, .2]])
    d = EscalationDecider()
    tau = d.find_tau_for_target_rate(p, .25)
    d.tau = tau
    _, escalated, _ = d.decide(p)
    assert sum(escalated) == 1  # closest achievable rate is 25%, without splitting ties
