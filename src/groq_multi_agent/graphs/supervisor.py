"""Flat supervisor graph: routes researcher, analyst, writer, FINISH."""

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
from groq_multi_agent.patterns.verification import verify_writer_output
from groq_multi_agent.prompts import SUPERVISOR_SYSTEM
from groq_multi_agent.state import TeamState

MEMBERS = ("researcher", "analyst", "writer")
DEFAULT_RECURSION_LIMIT = 15


class RouteDecision(BaseModel):
    next: Literal["researcher", "analyst", "writer", "FINISH"] = Field(
        description="Next worker to run, or FINISH when done."
    )


def _supervisor_router(state: TeamState) -> str:
    n = state.get("next", "researcher")
    if n == "FINISH":
        return END
    return n


def _latest_task(messages) -> str:
    for msg in reversed(messages):
        if getattr(msg, "type", None) == "human":
            c = msg.content
            if isinstance(c, str) and c.strip():
                return c
    return ""


def make_supervisor_node(
    llm: BaseChatModel,
    worker_llm: BaseChatModel,
    system_prompt: str = SUPERVISOR_SYSTEM,
):
    def supervisor(state: TeamState) -> dict:
        step = int(state.get("step_count", 0)) + 1
        trace_event("supervisor", "tick", f"step={step}")

        if circuit_open(step):
            trace_event("supervisor", "circuit_open", f"step={step}")
            return {
                "next": "FINISH",
                "step_count": step,
                "messages": [
                    AIMessage(
                        content="Circuit breaker: max supervisor steps reached.",
                        name="supervisor",
                    )
                ],
            }

        for msg in reversed(state["messages"]):
            if getattr(msg, "name", None) == "writer":
                body = msg.content if isinstance(msg.content, str) else str(msg.content)
                if not body.strip():
                    break
                if state.get("writer_verified"):
                    return {"next": "FINISH", "step_count": step, "writer_verified": True}
                task = _latest_task(state["messages"])
                if verify_writer_output(task, body, worker_llm):
                    trace_event("supervisor", "verified", "writer_output")
                    return {"next": "FINISH", "step_count": step, "writer_verified": True}
                retries = int(state.get("writer_retries", 0))
                if retries < 1:
                    trace_event("supervisor", "verify_retry", "writer")
                    return {"next": "writer", "step_count": step, "writer_retries": retries + 1}
                return {"next": "FINISH", "step_count": step, "writer_verified": True}
                break

        context_lines: list[str] = []
        for msg in state["messages"]:
            role = getattr(msg, "name", None) or getattr(msg, "type", "msg")
            body = msg.content if isinstance(msg.content, str) else str(msg.content)
            context_lines.append(f"{role}: {body[:1500]}")
        context = "\n".join(context_lines)

        try:
            decision: RouteDecision = invoke_structured_route(
                llm,
                RouteDecision,
                [
                    SystemMessage(content=system_prompt),
                    HumanMessage(
                        content=(
                            "Given the conversation below, choose the next step.\n\n"
                            f"{context}"
                        )
                    ),
                ],
            )
            nxt = decision.next
            trace_event("supervisor", "route", nxt)
        except Exception as exc:
            hint = groq_model_error_hint(exc)
            return {
                "next": "FINISH",
                "step_count": step,
                "messages": [
                    AIMessage(
                        content=f"Supervisor routing failed ({exc}).{hint} Ending run.",
                        name="supervisor",
                    )
                ],
            }

        return {"next": nxt, "step_count": step}

    return supervisor


def build_supervisor_graph(
    llm: BaseChatModel | None = None,
    worker_llm: BaseChatModel | None = None,
    *,
    router_model: str | None = None,
    recursion_limit: int = DEFAULT_RECURSION_LIMIT,
):
    """Compile supervisor workflow."""
    if router_model:
        router = get_llm_for_model(router_model)
    else:
        router = get_llm_or(llm)
    workers = get_worker_llm_or(llm, worker_llm)
    builder = StateGraph(TeamState)

    builder.add_node("supervisor", make_supervisor_node(router, workers))
    builder.add_node("researcher", make_researcher_node(workers))
    builder.add_node("analyst", make_analyst_node(workers))
    builder.add_node("writer", make_writer_node(workers))

    builder.add_edge(START, "supervisor")
    builder.add_conditional_edges(
        "supervisor",
        _supervisor_router,
        {**{m: m for m in MEMBERS}, END: END},
    )
    for member in MEMBERS:
        builder.add_edge(member, "supervisor")

    compiled = builder.compile()
    compiled._groq_multi_agent_config = {"recursion_limit": recursion_limit}  # type: ignore[attr-defined]
    return compiled
