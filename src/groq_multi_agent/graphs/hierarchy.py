"""CEO + research-team subgraph over shared workers."""

from __future__ import annotations

from typing import Literal

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field

from groq_multi_agent.agents import (
    make_analyst_node,
    make_researcher_node,
    make_writer_node,
)
from groq_multi_agent.config import get_llm_for_model, get_llm_or, get_worker_llm_or, groq_model_error_hint
from groq_multi_agent.patterns.circuit_breaker import circuit_open
from groq_multi_agent.patterns.observability import trace_event
from groq_multi_agent.patterns.routing import invoke_structured_route
from groq_multi_agent.prompts import CEO_SYSTEM, RESEARCH_LEAD_SYSTEM
from groq_multi_agent.state import TeamState

RESEARCH_MEMBERS = ("researcher", "analyst")
DEFAULT_RECURSION_LIMIT = 20


class ResearchRouteDecision(BaseModel):
    next: Literal["researcher", "analyst", "FINISH"] = Field(
        description="Next research worker or FINISH when research phase is done."
    )


class CEORouteDecision(BaseModel):
    next: Literal["research_team", "writer", "FINISH"] = Field(
        description="Delegate to research_team or writer, or FINISH."
    )


def _research_router(state: TeamState) -> str:
    if state.get("next") == "FINISH":
        return END
    return state.get("next", "researcher")


def _ceo_router(state: TeamState) -> str:
    nxt = state.get("next", "research_team")
    if nxt == "FINISH":
        return END
    return nxt


def make_research_lead_node(llm: BaseChatModel):
    def research_lead(state: TeamState) -> dict:
        step = int(state.get("step_count", 0)) + 1
        if circuit_open(step):
            return {"next": "FINISH", "step_count": step}
        context = _format_messages(state["messages"])
        try:
            decision: ResearchRouteDecision = invoke_structured_route(
                llm,
                ResearchRouteDecision,
                [
                    SystemMessage(content=RESEARCH_LEAD_SYSTEM),
                    HumanMessage(content=f"Conversation:\n{context}"),
                ],
            )
            trace_event("research_lead", "route", decision.next)
            return {"next": decision.next, "step_count": step}
        except Exception as exc:
            return {
                "next": "FINISH",
                "messages": [
                    AIMessage(
                        content=f"Research lead routing failed ({exc}).{groq_model_error_hint(exc)}",
                        name="research_lead",
                    )
                ],
            }

    return research_lead


def make_ceo_node(llm: BaseChatModel):
    def ceo(state: TeamState) -> dict:
        step = int(state.get("step_count", 0)) + 1
        if circuit_open(step):
            return {"next": "FINISH", "step_count": step}
        context = _format_messages(state["messages"])
        try:
            decision: CEORouteDecision = invoke_structured_route(
                llm,
                CEORouteDecision,
                [
                    SystemMessage(content=CEO_SYSTEM),
                    HumanMessage(content=f"Conversation:\n{context}"),
                ],
            )
            trace_event("ceo", "route", decision.next)
            return {"next": decision.next, "step_count": step}
        except Exception as exc:
            return {
                "next": "FINISH",
                "messages": [
                    AIMessage(
                        content=f"CEO routing failed ({exc}).{groq_model_error_hint(exc)}",
                        name="ceo",
                    ),
                ],
            }

    return ceo


def _format_messages(messages) -> str:
    lines: list[str] = []
    for msg in messages:
        role = getattr(msg, "name", None) or getattr(msg, "type", "msg")
        body = msg.content if isinstance(msg.content, str) else str(msg.content)
        lines.append(f"{role}: {body[:1200]}")
    return "\n".join(lines)


def _research_team_summary(state: TeamState) -> dict:
    """After subgraph, surface one message for the CEO."""
    chunks: list[str] = []
    for msg in state["messages"]:
        name = getattr(msg, "name", None)
        if name in ("researcher", "analyst", "research_lead"):
            body = msg.content if isinstance(msg.content, str) else str(msg.content)
            chunks.append(f"[{name}] {body[:800]}")
    summary = "\n".join(chunks) if chunks else "Research team produced no output."
    return {
        "messages": [
            AIMessage(content=f"Research team summary:\n{summary}", name="research_team")
        ],
    }


def build_research_subgraph(router: BaseChatModel, workers: BaseChatModel):
    builder = StateGraph(TeamState)
    builder.add_node("research_lead", make_research_lead_node(router))
    builder.add_node("researcher", make_researcher_node(workers))
    builder.add_node("analyst", make_analyst_node(workers))

    builder.add_edge(START, "research_lead")
    builder.add_conditional_edges(
        "research_lead",
        _research_router,
        {**{m: m for m in RESEARCH_MEMBERS}, END: END},
    )
    for member in RESEARCH_MEMBERS:
        builder.add_edge(member, "research_lead")
    return builder.compile()


def build_hierarchy_graph(
    llm: BaseChatModel | None = None,
    worker_llm: BaseChatModel | None = None,
    *,
    router_model: str | None = None,
    recursion_limit: int = DEFAULT_RECURSION_LIMIT,
):
    if router_model:
        router = get_llm_for_model(router_model)
    else:
        router = get_llm_or(llm)
    workers = get_worker_llm_or(llm, worker_llm)
    research_team = build_research_subgraph(router, workers)

    builder = StateGraph(TeamState)
    builder.add_node("ceo", make_ceo_node(router))
    builder.add_node("research_team", research_team)
    builder.add_node("research_team_wrap", _research_team_summary)
    builder.add_node("writer", make_writer_node(workers))

    def _enter_research(_state: TeamState) -> dict:
        return {"next": "researcher"}

    builder.add_node("enter_research", _enter_research)

    builder.add_edge(START, "ceo")
    builder.add_conditional_edges(
        "ceo",
        _ceo_router,
        {
            "research_team": "enter_research",
            "writer": "writer",
            END: END,
        },
    )
    builder.add_edge("enter_research", "research_team")
    builder.add_edge("research_team", "research_team_wrap")
    builder.add_edge("research_team_wrap", "ceo")
    builder.add_edge("writer", "ceo")

    compiled = builder.compile()
    compiled._groq_multi_agent_config = {"recursion_limit": recursion_limit}  # type: ignore[attr-defined]
    return compiled
