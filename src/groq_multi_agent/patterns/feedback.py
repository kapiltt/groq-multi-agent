"""Rich Feedback Loops — persist runtime incidents for prompt/code fixes."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

_INCIDENTS = Path(__file__).resolve().parents[3] / "logs" / "incidents.jsonl"


def log_incident(code: str, detail: str) -> None:
    row = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "code": code,
        "detail": detail[:2000],
    }
    try:
        _INCIDENTS.parent.mkdir(parents=True, exist_ok=True)
        with _INCIDENTS.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")
    except OSError:
        pass
