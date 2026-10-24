"""Failover-Aware Model Fallback for Groq calls."""

from __future__ import annotations

import os
from typing import Callable, TypeVar

from langchain_core.language_models.chat_models import BaseChatModel

from groq_multi_agent.config import require_groq_api_key
from groq_multi_agent.patterns.feedback import log_incident
from langchain_groq import ChatGroq

T = TypeVar("T")

FALLBACK_CHAIN = (
    "openai/gpt-oss-120b",
    "openai/gpt-oss-20b",
    "qwen/qwen3.8-27b",
)


def _fallback_models(primary: str) -> list[str]:
    extra = [
        m.strip()
        for m in os.getenv("GROQ_FALLBACK_MODELS", "").split(",")
        if m.strip()
    ]
    chain = [primary, *FALLBACK_CHAIN, *extra]
    seen: set[str] = set()
    out: list[str] = []
    for m in chain:
        if m not in seen:
            seen.add(m)
            out.append(m)
    return out


def _chat(model: str) -> ChatGroq:
    return ChatGroq(model=model, temperature=0, api_key=require_groq_api_key())


def invoke_with_failover(
    primary: BaseChatModel,
    operation: Callable[[BaseChatModel], T],
) -> T:
    if not isinstance(primary, ChatGroq):
        return operation(primary)

    primary_id = str(getattr(primary, "model_name", "") or getattr(primary, "model", ""))
    errors: list[str] = []
    for model_id in _fallback_models(primary_id):
        client = _chat(model_id) if model_id != primary_id else primary
        try:
            return operation(client)
        except Exception as exc:
            err = str(exc)
            errors.append(f"{model_id}: {err[:200]}")
            if not any(
                x in err
                for x in ("429", "model_not_found", "503", "rate", "tool_use_failed")
            ):
                raise
            log_incident("failover", err[:500])
    raise RuntimeError("All model fallbacks failed: " + " | ".join(errors))
