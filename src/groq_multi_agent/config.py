"""Environment and LLM configuration."""

from __future__ import annotations

import os
from functools import lru_cache

from pathlib import Path

from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.language_models.chat_models import BaseChatModel

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(_PROJECT_ROOT / ".env", override=True)
load_dotenv()

# Groq retired llama-3.3-70b-versatile for free/developer tiers (2026-08-16).
# See https://console.groq.com/docs/models — override with GROQ_MODEL in .env.
DEFAULT_MODEL = "openai/gpt-oss-120b"
# GPT-OSS often emits tool calls on plain chat; use Qwen for worker text by default.
DEFAULT_TEXT_MODEL = "qwen/qwen3.8-27b"


def require_groq_api_key() -> str:
    key = os.getenv("GROQ_API_KEY", "").strip()
    if not key:
        raise RuntimeError(
            "GROQ_API_KEY is not set. Copy .env.example to .env and add your key."
        )
    return key


def get_model_name() -> str:
    return os.getenv("GROQ_MODEL", DEFAULT_MODEL).strip() or DEFAULT_MODEL


def researcher_uses_tool_agent() -> bool:
    """Whether the researcher uses LangChain tool-calling agents vs a scripted search flow."""
    mode = os.getenv("GROQ_RESEARCHER_MODE", "auto").strip().lower()
    if mode in ("scripted", "deterministic"):
        return False
    if mode in ("tools", "agent"):
        return True
    # GPT-OSS on Groq often calls hallucinated tools (e.g. open_file).
    if get_model_name().lower().startswith("openai/gpt-oss"):
        return False
    return True


def get_text_model_name() -> str:
    explicit = os.getenv("GROQ_TEXT_MODEL", "").strip()
    if explicit:
        return explicit
    main = get_model_name().lower()
    if main.startswith("openai/gpt-oss"):
        return DEFAULT_TEXT_MODEL
    return get_model_name()


@lru_cache(maxsize=8)
def _cached_chat(model: str) -> ChatGroq:
    return ChatGroq(
        model=model,
        temperature=0,
        api_key=require_groq_api_key(),
    )


def get_llm_for_model(model_name: str) -> ChatGroq:
    return _cached_chat(model_name)


@lru_cache(maxsize=1)
def get_llm() -> ChatGroq:
    return _cached_chat(get_model_name())


@lru_cache(maxsize=1)
def get_text_llm() -> ChatGroq:
    return _cached_chat(get_text_model_name())


def groq_model_error_hint(exc: BaseException) -> str:
    err = str(exc)
    if "model_not_found" in err or "does not exist" in err:
        return (
            " Check GROQ_MODEL in .env (Groq retired llama-3.3-70b-versatile for "
            "free/developer keys; try openai/gpt-oss-120b or qwen/qwen3.8-27b)."
        )
    return ""


def get_llm_or(model: BaseChatModel | None) -> BaseChatModel:
    """Return injected model for tests, else the configured Groq client."""
    if model is not None:
        return model
    return get_llm()


def get_worker_llm_or(
    router_llm: BaseChatModel | None,
    worker_llm: BaseChatModel | None = None,
) -> BaseChatModel:
    """Workers need reliable plain-text generation (often a different model than router)."""
    if worker_llm is not None:
        return worker_llm
    if router_llm is not None:
        return router_llm
    return get_text_llm()
