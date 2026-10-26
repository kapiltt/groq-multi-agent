"""CLI entry point for multi-agent workflows."""

from __future__ import annotations

import argparse
import sys

from langchain_core.messages import HumanMessage

from groq_multi_agent.config import (
    get_llm,
    get_model_name,
    get_text_llm,
    get_text_model_name,
    require_groq_api_key,
)
from groq_multi_agent.graphs.hierarchy import DEFAULT_RECURSION_LIMIT as HIERARCHY_LIMIT
from groq_multi_agent.graphs.hierarchy import build_hierarchy_graph
from groq_multi_agent.graphs.supervisor import DEFAULT_RECURSION_LIMIT as SUPERVISOR_LIMIT
from groq_multi_agent.graphs.supervisor import build_supervisor_graph
from groq_multi_agent.graphs.supervisor import RouteDecision
from groq_multi_agent.llm_chat import invoke_chat_text
from groq_multi_agent.patterns.human import confirm_report
from groq_multi_agent.patterns.observability import trace_event
from groq_multi_agent.patterns.routing import invoke_structured_route, pick_router_model_for_task
from groq_multi_agent.triage import assess_task, triage_enabled

DEFAULT_TASK = (
    "Research the tradeoffs of using Groq for a startup's customer-support "
    "assistant and write a one-page recommendation."
)

EXAMPLE_TASKS = [
    "Compare FastAPI vs Flask for a small SaaS API and recommend one.",
    "Summarize pros and cons of LangGraph for a hackathon multi-agent demo.",
    "What should a startup know about Groq pricing and rate limits in 2026?",
]


def _graph_config(graph, fallback: int) -> dict:
    return getattr(graph, "_groq_multi_agent_config", {"recursion_limit": fallback})


def _print_transcript(messages) -> None:
    for msg in messages:
        name = getattr(msg, "name", None) or getattr(msg, "type", "message")
        content = msg.content if isinstance(msg.content, str) else str(msg.content)
        print(f"\n--- {name} ---\n{content}\n")


def _final_report(messages) -> str:
    for msg in reversed(messages):
        if getattr(msg, "name", None) == "writer":
            return msg.content if isinstance(msg.content, str) else str(msg.content)
    last = messages[-1]
    return last.content if isinstance(last.content, str) else str(last.content)


def run_check() -> int:
    """Verify API key, router model, and text worker model."""
    require_groq_api_key()
    router = get_llm()
    workers = get_text_llm()
    print(f"Router model (supervisor): {get_model_name()}")
    print(f"Worker model (research/write): {get_text_model_name()}")
    decision = invoke_structured_route(
        router,
        RouteDecision,
        [HumanMessage(content="Need research first; pick next agent.")],
    )
    print(f"Structured routing OK -> {decision.next}")
    ping = invoke_chat_text(workers, [HumanMessage(content='Say "ok" only.')])
    print(f"Text worker OK -> {ping[:80]}")
    print("Environment check passed.")
    return 0


def run_task(
    task: str,
    *,
    hierarchy: bool = False,
    auto_approve: bool = False,
    skip_triage: bool = False,
) -> str:
    if triage_enabled() and not skip_triage:
        gate = assess_task(task)
        trace_event("triage", "decision", f"eligible={gate.eligible} {gate.reason}")
        if not gate.eligible:
            print(gate.user_message)
            return gate.user_message

    router_model = pick_router_model_for_task(task)
    trace_event("run", "start", f"router={router_model} hierarchy={hierarchy}")
    if hierarchy:
        graph = build_hierarchy_graph(router_model=router_model)
        config = _graph_config(graph, HIERARCHY_LIMIT)
        initial_next = "research_team"
    else:
        graph = build_supervisor_graph(router_model=router_model)
        config = _graph_config(graph, SUPERVISOR_LIMIT)
        initial_next = "researcher"

    state = graph.invoke(
        {
            "messages": [HumanMessage(content=task)],
            "next": initial_next,
            "step_count": 0,
            "writer_verified": False,
            "writer_retries": 0,
        },
        config=config,
    )
    _print_transcript(state["messages"])
    report = _final_report(state["messages"])
    if not confirm_report(report, auto_approve=auto_approve):
        print("Report rejected by reviewer.", file=sys.stderr)
        return report
    print("=== Final report ===\n")
    print(report)
    trace_event("run", "done", "")
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run Groq multi-agent workflows.",
        epilog="Examples:\n  %(prog)s --check\n  %(prog)s \"Your question here\"\n",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("task", nargs="?", default=DEFAULT_TASK, help="Task for the team")
    parser.add_argument(
        "--hierarchy",
        action="store_true",
        help="Use CEO + research-team hierarchy instead of flat supervisor",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Verify GROQ_API_KEY and models without running the full pipeline",
    )
    parser.add_argument(
        "--list-examples",
        action="store_true",
        help="Print sample tasks you can pass as the task argument",
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        help="Skip human approval prompt (or set GROQ_REQUIRE_APPROVAL=0)",
    )
    parser.add_argument(
        "--skip-triage",
        action="store_true",
        help="Run pipeline without research-worthiness check",
    )
    args = parser.parse_args(argv)

    if args.list_examples:
        for i, t in enumerate(EXAMPLE_TASKS, 1):
            print(f"{i}. {t}")
        print(f"\nDefault:\n{DEFAULT_TASK}")
        return 0

    if args.check:
        try:
            return run_check()
        except Exception as exc:
            print(f"Check failed: {exc}", file=sys.stderr)
            return 1

    try:
        run_task(
            args.task,
            hierarchy=args.hierarchy,
            auto_approve=args.yes,
            skip_triage=args.skip_triage,
        )
    except RuntimeError as exc:
        print(exc, file=sys.stderr)
        return 1
    except Exception as exc:
        print(f"Run failed: {exc}", file=sys.stderr)
        if "429" in str(exc):
            print("Groq rate limit — wait a minute or set GROQ_MODEL=openai/gpt-oss-20b", file=sys.stderr)
        if "Recursion limit" in str(exc):
            print(
                "Supervisor did not reach FINISH in time — retry or use a shorter task.",
                file=sys.stderr,
            )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
