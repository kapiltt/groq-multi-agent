"""Test doubles for LLM routing and worker outputs."""

from __future__ import annotations

from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.runnables import Runnable
from pydantic import BaseModel


class ScriptRouterChatModel(BaseChatModel):
    """Returns canned text from invoke; structured routes from a script."""

    routes: list[str]
    worker_reply: str = "stub worker output"
    _route_index: int = 0

    @property
    def _llm_type(self) -> str:
        return "script-router-fake"

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        return ChatResult(
            generations=[ChatGeneration(message=AIMessage(content=self.worker_reply))]
        )

    def with_structured_output(self, schema: type[BaseModel], **kwargs: Any) -> Runnable:
        model = self

        class _Structured(Runnable):
            def invoke(self, input: Any, config=None, **kw: Any) -> BaseModel:
                idx = min(model._route_index, len(model.routes) - 1)
                value = model.routes[idx]
                model._route_index += 1
                return schema(next=value)

        return _Structured()
