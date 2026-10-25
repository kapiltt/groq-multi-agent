# Agentic patterns used in groq-multi-agent

Reference catalogue: [awesome-agentic-patterns](https://github.com/nibzard/awesome-agentic-patterns)

## All 25 patterns (this project)

| # | Pattern | Where |
|---|---------|--------|
| 1 | Autonomous Workflow Agent Architecture | `graphs/supervisor.py`, `graphs/hierarchy.py` |
| 2 | Declarative Multi-Agent Topology Definition | LangGraph `StateGraph` |
| 3 | Discrete Phase Separation | researcher → analyst → writer |
| 4 | Plan-Then-Execute Pattern | supervisor routes, workers execute |
| 5 | Sub-Agent Spawning | `--hierarchy` / CEO + research subgraph |
| 6 | Structured Output Specification | `RouteDecision`, Pydantic routing |
| 7 | Dual LLM Pattern | `GROQ_MODEL` + `GROQ_TEXT_MODEL` |
| 8 | Oracle and Worker Multi-Model Approach | router vs worker models in `config.py` |
| 9 | Hybrid LLM/Code Workflow Coordinator | scripted `web_search` + LLM synthesis |
| 10 | AI Web Search Agent Loop | `tools/search.py`, researcher |
| 11 | Tool Capability Compartmentalization | search tools on researcher only |
| 12 | Surface-Scoped System Prompt | `prompts.py` per role |
| 13 | Context-Minimization Pattern | `patterns/context.py` compaction |
| 14 | CLI-Native Agent Orchestration | `python -m groq_multi_agent.run` |
| 15 | Workflow Evals with Mocked Tools | `tests/` + fake router |
| 16 | Failover-Aware Model Fallback | `patterns/failover.py` |
| 17 | Schema Validation Retry with Cross-Step Learning | `patterns/routing.py` |
| 18 | Agent Circuit Breaker | `patterns/circuit_breaker.py` |
| 19 | Output Verification Loop | `patterns/verification.py` |
| 20 | Context Window Auto-Compaction | `patterns/context.py` |
| 21 | Budget-Aware Model Routing with Hard Cost Caps | `pick_router_model_for_task` |
| 22 | LLM Observability | `patterns/observability.py` → `logs/agent-trace.jsonl` |
| 23 | Reflection Loop | analyst reflection in `agents.py` |
| 24 | Human-in-the-Loop Approval Framework | `patterns/human.py`, `--yes` |
| 25 | Rich Feedback Loops > Perfect Prompts | `patterns/feedback.py` → `logs/incidents.jsonl` |

## Env knobs

| Variable | Pattern |
|----------|---------|
| `GROQ_FALLBACK_MODELS` | Failover |
| `GROQ_MAX_SUPERVISOR_STEPS` | Circuit breaker (default 12) |
| `GROQ_BUDGET_MODE` | `auto` / `low` / `high` |
| `GROQ_TRACE` | Observability (`0` to disable) |
| `GROQ_REFLECTION` | Reflection (`0` to disable) |
| `GROQ_REQUIRE_APPROVAL` | Human-in-the-loop (`1` to enable) |
