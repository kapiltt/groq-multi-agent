"""Context Window Auto-Compaction for worker prompts."""

from __future__ import annotations

from langchain_core.messages import BaseMessage

from groq_multi_agent.tools.summarize import summarize_text


def compact_conversation_context(
    messages: list[BaseMessage],
    *,
    max_chars: int = 6000,
    per_message_cap: int = 800,
) -> str:
    lines: list[str] = []
    for msg in messages:
        name = getattr(msg, "name", None) or getattr(msg, "type", "message")
        content = msg.content if isinstance(msg.content, str) else str(msg.content)
        lines.append(f"[{name}]: {content[:per_message_cap]}")
    blob = "\n".join(lines)
    if len(blob) <= max_chars:
        return blob
    compacted = summarize_text.invoke({"text": blob, "max_sentences": 12})
    if len(compacted) > max_chars:
        return compacted[: max_chars - 4] + " ..."
    return compacted
