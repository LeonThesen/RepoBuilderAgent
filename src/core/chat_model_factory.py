from __future__ import annotations

from typing import Any, Callable

from langchain_openai import ChatOpenAI

from .prompt_dump_callback import PromptDumpCallbackHandler
from .token_usage_callback import TokenUsageCallbackHandler

# One handler instance is enough: it is stateless and reads per-invocation repo/
# phase from run metadata. It no-ops unless a prompt-dump dir is configured.
_PROMPT_DUMP_HANDLER = PromptDumpCallbackHandler()
# Counts real ReAct LLM-turn tokens (TODO 68) and emits [TOKENS] lines.
_TOKEN_USAGE_HANDLER = TokenUsageCallbackHandler()


def make_prebuilt_chat_model_factory(
    *,
    model: str,
    temperature: float,
    api_key: str,
    base_url: str,
    max_retries: int,
    http_async_client: Any,
) -> Callable[[int], ChatOpenAI]:
    """Create a timeout-aware ChatOpenAI builder for ReAct loops."""

    def _new_prebuilt_chat_model(timeout_seconds: int) -> ChatOpenAI:
        kwargs = {
            "model": model,
            "temperature": temperature,
            "api_key": api_key,
            "base_url": base_url,
            "timeout": timeout_seconds,
            "max_retries": max_retries,
            "http_async_client": http_async_client,
            "callbacks": [_PROMPT_DUMP_HANDLER, _TOKEN_USAGE_HANDLER],
        }
        if model == "gpt-5-codex":
            kwargs["use_responses_api"] = True
        return ChatOpenAI(**kwargs)

    return _new_prebuilt_chat_model
