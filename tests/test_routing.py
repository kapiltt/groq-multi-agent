from groq_multi_agent.graphs.supervisor import RouteDecision


def test_route_decision_accepts_finish():
    d = RouteDecision(next="FINISH")
    assert d.next == "FINISH"


def test_route_decision_members():
    for member in ("researcher", "analyst", "writer"):
        d = RouteDecision(next=member)
        assert d.next == member
