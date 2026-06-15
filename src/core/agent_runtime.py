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
class RepoRef:
    """Identity of the repository under analysis. These three always travel
    together (url + clone name + on-disk path), so they move as one value instead
    of as three parameters on every Stage-1 signature."""
    url: str
    name: str
    path: Any  # str | pathlib.Path, depending on caller


@dataclass(frozen=True)
class ClassifyConfig:
    """Run-constant Stage-1 tuning knobs (sourced once from argv). Each loop reads
    only the subset it needs; bundling them keeps the per-loop knobs out of the
    signatures, which otherwise carried five-to-seven of these each."""
    classification_timeout: int
    selection_timeout: int
    react_max_steps: int
    react_max_total_files: int
    react_final_cap: int
    synthesis_react_max_steps: int
    synthesis_review_rounds: int
    validation_react_max_steps: int
    synthesis_subagents_enabled: bool
    snippet_tools_enabled: bool
    run_validation: bool


@dataclass(frozen=True)
class ClassifyRuntime:
    model_name: str
    new_prebuilt_chat_model: Callable[[int], Any]
    extract_agent_payload: Callable[[dict[str, Any]], Any]
    extract_agent_trace: Callable[[dict[str, Any]], list[dict[str, Any]]]
    normalize_text_list: Callable[[Any], list[str]]
    normalize_validation_checks: Callable[[Any], Any]


@dataclass(frozen=True)
class RepairRuntime:
    """Stage-3 (L3 repair) analogue of ClassifyRuntime: the model factory plus the
    tool-builder callables the repair/verify ReAct loops need. Passed as one object
    because l3_react_loop can't import these directly from agent_dockerfile_repair
    (circular). build_snippet_tool / repo_tools stay per-call (config-dependent)."""
    model_name: str
    new_prebuilt_chat_model: Callable[[int], Any]
    build_think_tool: Callable[[], Any]
    build_hadolint_snippet_tool: Callable[[], Any]
    extract_dockerfile: Callable[[str], str]
