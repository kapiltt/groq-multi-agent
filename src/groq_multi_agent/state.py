"""Shared graph state."""

from __future__ import annotations

from typing import Annotated, TypedDict

from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage


class TeamState(TypedDict, total=False):
    messages: Annotated[list[BaseMessage], add_messages]
    next: str
    step_count: int
    writer_verified: bool
    writer_retries: int
