"""Web search tool using DuckDuckGo (no extra API key)."""

from __future__ import annotations

from langchain_core.tools import tool


@tool
def web_search(query: str, max_results: int = 5) -> str:
    """Search the web for the query and return title, URL, and snippet for top results."""
    try:
        from ddgs import DDGS
    except ImportError as exc:
        return f"Search failed: ddgs package not installed ({exc})"

    max_results = max(1, min(int(max_results), 10))
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=max_results))
    except Exception as exc:
        return f"Search failed: {exc}"

    if not results:
        return f"No results found for: {query}"

    lines: list[str] = []
    for i, hit in enumerate(results, start=1):
        title = hit.get("title") or "(no title)"
        url = hit.get("href") or hit.get("link") or ""
        body = hit.get("body") or hit.get("snippet") or ""
        lines.append(f"{i}. {title}\n   URL: {url}\n   {body}")
    return "\n\n".join(lines)
