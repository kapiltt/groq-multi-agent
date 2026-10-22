"""Worker agent nodes (researcher, analyst, writer)."""

from __future__ import annotations

import os

from langchain.agents import create_agent
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage

from groq_multi_agent.config import researcher_uses_tool_agent
from groq_multi_agent.llm_chat import invoke_chat_text
from groq_multi_agent.patterns.context import compact_conversation_context
from groq_multi_agent.patterns.feedback import log_incident
from groq_multi_agent.patterns.observability import trace_event
from groq_multi_agent.prompts import (
    ANALYST_REFLECTION_SYSTEM,
    ANALYST_SYSTEM,
    RESEARCHER_SYNTHESIS_SYSTEM,
    RESEARCHER_SYSTEM,
    WRITER_SYSTEM,
)
from groq_multi_agent.state import TeamState
from groq_multi_agent.tools.search import web_search
from groq_multi_agent.tools.summarize import summarize_text


def _latest_user_task(messages: list[BaseMessage]) -> str:
    for msg in reversed(messages):
        if getattr(msg, "type", None) == "human" or msg.__class__.__name__ == "HumanMessage":
            content = msg.content
            if isinstance(content, str) and content.strip():
                return content
    if messages:
        content = messages[0].content
        return content if isinstance(content, str) else str(content)
    return "Complete the assigned task."


def _search_fallback_report(task: str, raw: str, brief: str) -> str:
    return (
        f"Research findings for: {task}\n\n"
        f"Summary:\n{brief}\n\n"
        f"Sources (raw search):\n{raw[:6000]}"
    )


def _reflection_enabled() -> bool:
    return os.getenv("GROQ_REFLECTION", "1").strip().lower() not in ("0", "false", "no")


def scripted_research(task: str, llm: BaseChatModel) -> str:
    """Deterministic search → summarize → synthesize (no tool-calling loop)."""
    raw = web_search.invoke({"query": task, "max_results": 5})
    brief = summarize_text.invoke({"text": raw, "max_sentences": 8})
    messages = [
        SystemMessage(content=RESEARCHER_SYNTHESIS_SYSTEM),
        HumanMessage(
            content=(
                f"User task:\n{task}\n\n"
                f"Summarized search results:\n{brief}\n\n"
                f"Full search payload:\n{raw[:8000]}\n\n"
                "Synthesize research findings as bullet points with source URLs."
            )
        ),
    ]
    try:
        return invoke_chat_text(llm, messages)
    except Exception as exc:
        log_incident("research_fallback", str(exc))
        return _search_fallback_report(task, raw, brief)


def _tool_agent_research(task: str, llm: BaseChatModel) -> str:
    prompt = (
        f"User task:\n{task}\n\n"
        "Produce research findings for the team. Use web_search and summarize_text only."
    )
    agent = create_agent(
        llm,
        tools=[web_search, summarize_text],
        system_prompt=RESEARCHER_SYSTEM,
    )
    result = agent.invoke({"messages": [HumanMessage(content=prompt)]})
    last = result["messages"][-1]
    return last.content if isinstance(last.content, str) else str(last.content)


def make_researcher_node(llm: BaseChatModel):
    """Researcher gathers web data; avoids brittle tool loops on GPT-OSS by default."""

    def researcher(state: TeamState) -> dict:
        trace_event("researcher", "start", "")
        task = _latest_user_task(state["messages"])
        content: str
        if researcher_uses_tool_agent():
            try:
                content = _tool_agent_research(task, llm)
            except NotImplementedError:
                content = scripted_research(task, llm)
            except Exception as exc:
                err = str(exc)
                log_incident("researcher_tool_fail", err)
                if "tool_use_failed" in err or "tool call validation" in err.lower():
                    content = scripted_research(task, llm)
                else:
                    raise
        else:
            content = scripted_research(task, llm)

        return {"messages": [AIMessage(content=content, name="researcher")]}

    return researcher


def make_analyst_node(llm: BaseChatModel):
    def analyst(state: TeamState) -> dict:
        trace_event("analyst", "start", "")
        task = _latest_user_task(state["messages"])
        context = compact_conversation_context(state["messages"])
        content = invoke_chat_text(
            llm,
            [
                SystemMessage(content=ANALYST_SYSTEM),
                HumanMessage(
                    content=f"User task:\n{task}\n\nConversation so far:\n{context}"
                ),
            ],
        )
        if _reflection_enabled():
            reflection = invoke_chat_text(
                llm,
                [
                    SystemMessage(content=ANALYST_REFLECTION_SYSTEM),
                    HumanMessage(content=f"Analysis to review:\n{content[:5000]}"),
                ],
            )
            content = f"{content}\n\n### Reflection\n{reflection}"
        return {"messages": [AIMessage(content=content, name="analyst")]}

    return analyst


def make_writer_node(llm: BaseChatModel):
    def writer(state: TeamState) -> dict:
        trace_event("writer", "start", "")
        task = _latest_user_task(state["messages"])
        context = compact_conversation_context(state["messages"])
        content = invoke_chat_text(
            llm,
            [
                SystemMessage(content=WRITER_SYSTEM),
                HumanMessage(
                    content=f"User task:\n{task}\n\nConversation so far:\n{context}"
                ),
            ],
        )
        return {"messages": [AIMessage(content=content, name="writer")]}

    return writer
