"""Schema validation retry + budget-aware model picks."""

from __future__ import annotations

import os

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage
from pydantic import BaseModel

from groq_multi_agent.patterns.failover import invoke_with_failover
from groq_multi_agent.patterns.feedback import log_incident


def pick_router_model_for_task(task: str) -> str:
    """Budget-Aware Model Routing: shorter tasks use smaller router model."""
    mode = os.getenv("GROQ_BUDGET_MODE", "auto").strip().lower()
    configured = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b").strip()
    if mode == "low":
        return "openai/gpt-oss-20b"
    if mode == "high":
        return configured or "openai/gpt-oss-120b"
    if len(task.strip()) < 160:
        return "openai/gpt-oss-20b"
    return configured or "openai/gpt-oss-120b"


def invoke_structured_route(
    llm: BaseChatModel,
    schema: type[BaseModel],
    messages: list[BaseMessage],
    *,
    max_attempts: int = 3,
) -> BaseModel:
    """Schema Validation Retry — re-ask when structured routing fails."""

    last_exc: Exception | None = None
    for attempt in range(1, max_attempts + 1):
        try:

            def _call(client: BaseChatModel) -> BaseModel:
                structured = client.with_structured_output(schema)
                return structured.invoke(messages)

            return invoke_with_failover(llm, _call)
        except Exception as exc:
            last_exc = exc
            log_incident("schema_retry", f"attempt={attempt} {exc}")
    assert last_exc is not None
    raise last_exc
