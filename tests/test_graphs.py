from langchain_core.messages import HumanMessage

from groq_multi_agent.graphs.hierarchy import build_hierarchy_graph
from groq_multi_agent.graphs.supervisor import build_supervisor_graph
from fakes import ScriptRouterChatModel


def test_supervisor_graph_reaches_finish():
    routes = ["researcher", "analyst", "writer", "FINISH"]
    llm = ScriptRouterChatModel(routes=routes, worker_reply="step complete")
    graph = build_supervisor_graph(llm)
    config = getattr(graph, "_groq_multi_agent_config", {"recursion_limit": 15})
    state = graph.invoke(
        {
            "messages": [HumanMessage(content="Test task")],
            "next": "researcher",
            "step_count": 0,
            "writer_verified": False,
            "writer_retries": 0,
        },
        config=config,
    )
    assert state["next"] == "FINISH"
    names = [getattr(m, "name", None) for m in state["messages"]]
    assert "researcher" in names
    assert "analyst" in names
    assert "writer" in names


def test_hierarchy_graph_reaches_finish():
    # Structured routing order: CEO → subgraph (lead×3) → CEO → writer → CEO
    combined = [
        "research_team",
        "researcher",
        "analyst",
        "FINISH",
        "writer",
        "FINISH",
    ]
    llm = ScriptRouterChatModel(routes=combined, worker_reply="hierarchy step")
    graph = build_hierarchy_graph(llm)
    config = getattr(graph, "_groq_multi_agent_config", {"recursion_limit": 25})
    state = graph.invoke(
        {
            "messages": [HumanMessage(content="Test hierarchy task")],
            "next": "research_team",
            "step_count": 0,
            "writer_verified": False,
            "writer_retries": 0,
        },
        config=config,
    )
    assert state["next"] == "FINISH"
    assert any(getattr(m, "name", None) == "research_team" for m in state["messages"])
    assert any(getattr(m, "name", None) == "writer" for m in state["messages"])
