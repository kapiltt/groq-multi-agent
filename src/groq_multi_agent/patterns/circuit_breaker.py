"""Agent Circuit Breaker — cap supervisor iterations."""

from __future__ import annotations

import os


def max_supervisor_steps() -> int:
    return int(os.getenv("GROQ_MAX_SUPERVISOR_STEPS", "12"))


def circuit_open(step_count: int) -> bool:
    return step_count >= max_supervisor_steps()
