"""Token-usage callback for ReAct/agent LLM calls (TODO 68).

Direct chat_completion calls log their own `[TOKENS]` lines, but the LangChain
ReAct loops (L1 retrieval, L2 generator/reviewer, L3 repair) previously only
*estimated* the seed prompt — their actual multi-turn LLM usage was uncounted, so
tokens-per-repo and cost-per-build (a primary ablation axis) under-reported the
react-heavy variants. This handler records the real usage of every agent LLM turn
and emits a `[TOKENS]` line in the same format, so parse_tokens_from_log aggregates
it with everything else.

repo + phase come from the per-invocation run metadata (dump_repo / dump_phase),
matched to the LLM run by run_id (callbacks may fire on worker threads).
"""

from __future__ import annotations

import json
from typing import Any

from langchain_core.callbacks import BaseCallbackHandler

from .log_utils import log_info


def _extract_usage(response: Any) -> tuple[int, int, int]:
    """Pull (prompt, completion, total) tokens from an LLMResult, tolerant of shape."""
    llm_output = getattr(response, "llm_output", None) or {}
    usage = llm_output.get("token_usage") or llm_output.get("usage") or {}
    pt = int(usage.get("prompt_tokens", 0) or 0)
    ct = int(usage.get("completion_tokens", 0) or 0)
    tt = int(usage.get("total_tokens", 0) or 0)
    if not (pt or ct or tt):
        # Fallback: sum per-generation usage_metadata (newer LangChain shape).
        for gen_list in getattr(response, "generations", None) or []:
            for gen in gen_list:
                msg = getattr(gen, "message", None)
                um = getattr(msg, "usage_metadata", None) or {}
                pt += int(um.get("input_tokens", 0) or 0)
                ct += int(um.get("output_tokens", 0) or 0)
                tt += int(um.get("total_tokens", 0) or 0)
    if not tt:
        tt = pt + ct
    return pt, ct, tt


class TokenUsageCallbackHandler(BaseCallbackHandler):
    """Emit a [TOKENS] line per agent LLM turn so ReAct usage is counted."""

    def __init__(self) -> None:
        self._runs: dict[Any, tuple[str, str]] = {}

    def on_chat_model_start(self, serialized, messages, *, run_id=None, metadata=None, **kwargs) -> None:
        meta = metadata or {}
        self._runs[run_id] = (meta.get("dump_repo", ""), meta.get("dump_phase", "react"))

    def on_llm_end(self, response, *, run_id=None, **kwargs) -> None:
        repo, phase = self._runs.pop(run_id, ("", "react"))
        pt, ct, tt = _extract_usage(response)
        if tt:
            log_info("[TOKENS] " + json.dumps({
                "phase": phase, "repo": repo,
                "prompt_tokens": pt, "completion_tokens": ct, "total_tokens": tt,
            }))

    def on_llm_error(self, error, *, run_id=None, **kwargs) -> None:
        self._runs.pop(run_id, None)
