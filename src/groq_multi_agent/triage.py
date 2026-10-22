"""Classify whether a user task needs the research pipeline (Classify-Then-Act)."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from groq_multi_agent.config import get_llm
from groq_multi_agent.patterns.routing import invoke_structured_route

REJECT_REPLY = (
    "This system runs **web research + analysis + report** for substantive topics "
    "(e.g. comparing tools, business tradeoffs, technical write-ups).\n\n"
    "Your message doesn't look research-worthy — try a clear question like:\n"
    "“Compare X vs Y for a startup” or “What are pros/cons of Groq for customer support?”"
)


@dataclass(frozen=True)
class TriageResult:
    eligible: bool
    reason: str
    user_message: str


_GREETING = re.compile(
    r"^(hi|hello|hey|hola|namaste|good\s+(morning|evening|night)|"
    r"how\s+are\s+you|how\s+r\s+u|what'?s\s+up|sup)[\s!.?]*$",
    re.I,
)

_PERSONAL_OR_SYSTEM = re.compile(
    r"(what\s+is\s+(the\s+)?current\s+time|what\s+time\s+is\s+it|"
    r"what'?s\s+the\s+time|time\s+now|"
    r"what\s+is\s+my\s+name|what'?s\s+my\s+name|who\s+am\s+i|"
    r"where\s+am\s+i|what\s+day\s+is\s+it|today'?s\s+date)",
    re.I,
)

_VAGUE_SHORT = re.compile(
    r"^(what\s+is\s+this|what'?s\s+this|explain\s+this|help|test|ok|yes|no)[\s!.?]*$",
    re.I,
)

_RESEARCH_HINTS = re.compile(
    r"\b(research|compare|comparison|vs\.?|versus|pros?\s+and\s+cons?|tradeoff|"
    r"recommend|analy[sz]e|report|summar(y|ize)|startup|pricing|benchmark|"
    r"architecture|guide|overview|evaluate|write\s+a|one-?page|documentation|"
    r"how\s+does|why\s+should|best\s+practice|implementation)\b",
    re.I,
)


class SuitabilityDecision(BaseModel):
    research_worthy: bool = Field(
        description="True if the task needs web research and a multi-step report."
    )
    reason: str = Field(description="Brief reason for the decision.")


def triage_heuristic(task: str) -> TriageResult | None:
    """Fast local rules; returns None if an LLM check is needed."""
    t = task.strip()
    if not t:
        return TriageResult(False, "empty", "Please enter a research question.")

    if _GREETING.match(t):
        return TriageResult(
            False,
            "greeting",
            "Hi! I'm a research team agent — ask a topic you want researched and reported.",
        )

    if _PERSONAL_OR_SYSTEM.search(t):
        return TriageResult(
            False,
            "personal_or_system",
            "I can't answer personal info or live clock/time — I only do topic research from the web.",
        )

    if _VAGUE_SHORT.match(t):
        return TriageResult(
            False,
            "vague",
            "“What is this?” is too vague without context. Describe the topic to research.",
        )

    if len(t) < 12 and not _RESEARCH_HINTS.search(t):
        return TriageResult(False, "too_short", REJECT_REPLY)

    if _RESEARCH_HINTS.search(t) or len(t) >= 40:
        return TriageResult(True, "heuristic_ok", "")

    return None


def triage_with_llm(task: str, llm: BaseChatModel | None = None) -> TriageResult:
    model = llm or get_llm()
    decision: SuitabilityDecision = invoke_structured_route(
        model,
        SuitabilityDecision,
        [
            SystemMessage(
                content=(
                    "Decide if the user message needs a research team (web search, "
                    "analysis, written report). NOT worthy: greetings, chit-chat, "
                    "current time/date, user's name, vague 'what is this' without topic, "
                    "questions answerable without research. WORTHY: business/tech topics, "
                    "comparisons, recommendations, explainers needing sources."
                )
            ),
            HumanMessage(content=task),
        ],
        max_attempts=2,
    )
    if decision.research_worthy:
        return TriageResult(True, f"llm:{decision.reason}", "")
    return TriageResult(
        False,
        f"llm:{decision.reason}",
        f"Not a research task: {decision.reason}\n\n{REJECT_REPLY}",
    )


def triage_enabled() -> bool:
    return os.getenv("GROQ_TRIAGE", "1").strip().lower() not in ("0", "false", "no")


def assess_task(task: str, llm: BaseChatModel | None = None) -> TriageResult:
    """Run heuristics first, then optional LLM for borderline cases."""
    fast = triage_heuristic(task)
    if fast is not None:
        if fast.eligible:
            return fast
        return fast

    use_llm = os.getenv("GROQ_TRIAGE_LLM", "1").strip().lower() not in ("0", "false", "no")
    if use_llm:
        return triage_with_llm(task, llm)
    return TriageResult(False, "borderline_no_llm", REJECT_REPLY)