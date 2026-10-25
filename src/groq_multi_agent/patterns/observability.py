"""LLM Observability — lightweight step tracing."""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path

_TRACE = os.getenv("GROQ_TRACE", "1").strip() not in ("0", "false", "no")
_LOG = Path(__file__).resolve().parents[3] / "logs" / "agent-trace.jsonl"


def trace_event(agent: str, event: str, detail: str = "") -> None:
    if not _TRACE:
        return
    line = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "agent": agent,
        "event": event,
        "detail": detail[:500],
    }
    msg = f"[trace] {agent}.{event}: {detail[:120]}"
    print(msg, flush=True)
    try:
        _LOG.parent.mkdir(parents=True, exist_ok=True)
        with _LOG.open("a", encoding="utf-8") as f:
            f.write(json.dumps(line) + "\n")
    except OSError:
        pass
