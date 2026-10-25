from groq_multi_agent.patterns.circuit_breaker import circuit_open, max_supervisor_steps
from groq_multi_agent.patterns.context import compact_conversation_context
from groq_multi_agent.patterns.routing import pick_router_model_for_task
from langchain_core.messages import HumanMessage


def test_circuit_breaker():
    assert not circuit_open(max_supervisor_steps() - 1)
    assert circuit_open(max_supervisor_steps())


def test_budget_router_short_task():
    assert pick_router_model_for_task("hi") == "openai/gpt-oss-20b"


def test_context_compaction():
    msgs = [HumanMessage(content="Sentence one. " * 300)]
    out = compact_conversation_context(msgs, max_chars=400)
    assert len(out) <= 420
