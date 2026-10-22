from unittest.mock import patch

from groq_multi_agent.agents import scripted_research
from fakes import ScriptRouterChatModel


def test_scripted_research_uses_search_tools():
    llm = ScriptRouterChatModel(routes=["FINISH"], worker_reply="Research bullets.")
    with patch("groq_multi_agent.agents.web_search") as mock_search:
        mock_search.invoke.return_value = "1. Example\n   URL: https://example.com"
        with patch("groq_multi_agent.agents.summarize_text") as mock_sum:
            mock_sum.invoke.return_value = "Example snippet."
            out = scripted_research("groq pricing", llm)
    assert "Research bullets" in out
    mock_search.invoke.assert_called_once()
    mock_sum.invoke.assert_called_once()
