from __future__ import annotations

from typing import Any, Callable

from langchain_openai import ChatOpenAI


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
        }
        return ChatOpenAI(**kwargs)

    return _new_prebuilt_chat_model
