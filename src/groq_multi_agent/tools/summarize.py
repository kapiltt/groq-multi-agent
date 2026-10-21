"""Extractive text summarization tool."""

from __future__ import annotations

import re

from langchain_core.tools import tool


def _split_sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [p.strip() for p in parts if p.strip()]


@tool
def summarize_text(text: str, max_sentences: int = 5) -> str:
    """Return the first max_sentences sentences of text (extractive summary)."""
    if not text or not text.strip():
        return "(empty input)"
    max_sentences = max(1, min(int(max_sentences), 20))
    sentences = _split_sentences(text)
    if not sentences:
        return text[:500]
    summary = " ".join(sentences[:max_sentences])
    if len(sentences) > max_sentences:
        summary += " [...]"
    return summary
