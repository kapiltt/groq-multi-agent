"""Output Verification Loop — writer output vs user task."""

from __future__ import annotations

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from langchain_groq import ChatGroq

from groq_multi_agent.llm_chat import invoke_chat_text

VERIFY_SYSTEM = (
    "You verify whether the draft answers the user task. "
    "Reply with exactly YES or NO."
)


def verify_writer_output(task: str, draft: str, llm: BaseChatModel) -> bool:
    if not isinstance(llm, ChatGroq):
        return bool(draft.strip())
    if not draft.strip():
        return False
    if len(task) > 400 and len(draft) < 80:
        return False
    answer = invoke_chat_text(
        llm,
        [
            SystemMessage(content=VERIFY_SYSTEM),
            HumanMessage(
                content=f"Task:\n{task}\n\nDraft:\n{draft[:6000]}\n\nDoes draft answer the task?"
            ),
        ],
    ).strip().upper()
    return answer.startswith("YES")
