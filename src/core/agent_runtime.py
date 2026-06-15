"""Shared runtime dependencies for the Stage-1 agent loops.

The L1 retrieval, L2 synthesis, L3 validation and architecture-state-graph loops
all need the same handful of cross-cutting helpers: the timeout-aware chat-model
factory plus the payload/trace extractors and the list/checks normalizers. These
are constant for a classification run and were previously threaded as five-to-six
separate parameters through every signature (run_l2_synthesis_loop had 17 params,
run_architecture_state_graph 20). Bundling them into one frozen object keeps the
signatures about the actual work, not the plumbing.

They live as callables (rather than imported directly) because their concrete
implementations live in agent_classify, which imports these loop modules — so a
direct import would be circular. Passing one object instead of six is the cheap
fix for that constraint.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable


@dataclass(frozen=True)
class ClassifyRuntime:
    model_name: str
    new_prebuilt_chat_model: Callable[[int], Any]
    extract_agent_payload: Callable[[dict[str, Any]], Any]
    extract_agent_trace: Callable[[dict[str, Any]], list[dict[str, Any]]]
    normalize_text_list: Callable[[Any], list[str]]
    normalize_validation_checks: Callable[[Any], Any]
