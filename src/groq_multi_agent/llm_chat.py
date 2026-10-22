"""Safe text-only Groq chat calls (avoids GPT-OSS bogus tool calls)."""

from __future__ import annotations

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage

from groq_multi_agent.patterns.failover import invoke_with_failover


def _model_id(llm: BaseChatModel) -> str:
    return str(getattr(llm, "model_name", "") or getattr(llm, "model", "")).lower()


def _groq_text_kwargs(llm: BaseChatModel) -> dict:
    if "gpt-oss" in _model_id(llm):
        return {"tool_choice": "none", "reasoning_effort": "low"}
    return {}


def invoke_chat_text(llm: BaseChatModel, messages: list[BaseMessage]) -> str:
    """Invoke model for plain text with failover and one tool-call retry."""

    def _once(client: BaseChatModel) -> str:
        kwargs = _groq_text_kwargs(client)
        try:
            response = client.invoke(messages, **kwargs)
        except Exception as exc:
            if "tool_use_failed" not in str(exc) and "called a tool" not in str(exc):
                raise
            retry_msgs = [
                SystemMessage(
                    content=(
                        "Reply in plain text only. Do not use JSON, functions, or tools. "
                        "No markdown code fences."
                    )
                ),
                *messages,
                HumanMessage(content="Output plain text bullets only."),
            ]
            response = client.invoke(retry_msgs, **kwargs)
        content = response.content
        return content if isinstance(content, str) else str(content)

    return invoke_with_failover(llm, _once)
