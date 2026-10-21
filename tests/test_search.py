from unittest.mock import MagicMock, patch

from groq_multi_agent.tools.search import web_search


def test_web_search_formats_results():
    fake_hits = [
        {"title": "A", "href": "https://a", "body": "snippet a"},
        {"title": "B", "href": "https://b", "body": "snippet b"},
    ]
    mock_ddgs = MagicMock()
    mock_ddgs.__enter__.return_value = mock_ddgs
    mock_ddgs.text.return_value = fake_hits

    with patch("ddgs.DDGS", return_value=mock_ddgs):
        out = web_search.invoke({"query": "groq", "max_results": 2})

    assert "https://a" in out
    assert "snippet b" in out


def test_web_search_failure_returns_message():
    with patch("ddgs.DDGS", side_effect=RuntimeError("network")):
        out = web_search.invoke({"query": "x"})
    assert out.startswith("Search failed:")
