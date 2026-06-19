from __future__ import annotations

from typing import Any

from langchain_core.callbacks import BaseCallbackHandler

from .log_utils import dump_prompt, prompt_dump_enabled


_ROLE_BY_TYPE = {
    "human": "user",
    "ai": "assistant",
    "system": "system",
    "tool": "tool",
    "function": "tool",
}


def _content_to_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, dict):
                parts.append(str(block.get("text", block)))
            else:
                parts.append(str(block))
        return "\n".join(parts)
    return str(content)


class PromptDumpCallbackHandler(BaseCallbackHandler):
    """Persists every ReAct LLM turn (seed prompt plus each tool-loop step) so the
    prompt record for a run is complete, not just the direct chat_completion calls.

    repo + phase come from the run metadata each ReAct invocation threads through
    its config (dump_metadata); using metadata rather than a contextvar keeps this
    correct even when LangChain dispatches callbacks on a worker thread. No-ops
    unless a prompt-dump directory was configured for the run."""

    def on_chat_model_start(self, serialized, messages, *, metadata=None, **kwargs) -> None:
        if not prompt_dump_enabled():
            return
        meta = metadata or {}
        repo_url = meta.get("dump_repo", "")
        phase = meta.get("dump_phase", "react")
        # messages is a batch: list of prompts, each a list of BaseMessage.
        for prompt_messages in messages or []:
            converted = [
                {
                    "role": _ROLE_BY_TYPE.get(getattr(m, "type", ""), getattr(m, "type", "unknown")),
                    "content": _content_to_text(getattr(m, "content", "")),
                }
                for m in prompt_messages
            ]
            if converted:
                dump_prompt(repo_url, phase, converted)
