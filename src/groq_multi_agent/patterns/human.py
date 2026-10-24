"""Human-in-the-Loop approval before final deliverable."""

from __future__ import annotations

import sys


def approval_required() -> bool:
    import os

    return os.getenv("GROQ_REQUIRE_APPROVAL", "").strip().lower() in (
        "1",
        "true",
        "yes",
    )


def confirm_report(report: str, *, auto_approve: bool = False) -> bool:
    if auto_approve or not approval_required():
        return True
    print("\n=== Pending approval ===\n")
    print(report[:8000])
    if len(report) > 8000:
        print("\n... [truncated for review] ...\n")
    answer = input("Approve this report? [y/N]: ").strip().lower()
    return answer in ("y", "yes")
