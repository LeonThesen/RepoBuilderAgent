"""Architecture manifest — the single declarative source of truth for the agent
components and ReAct tools the pipeline claims to provide.

For every subpart and tool it records:
  * the symbol that implements it          → used by the *wiring* audit
  * the config that activates it           → so disabled parts aren't expected
  * the per-repo artifact it should write  → used by the *output* audit

Two distinct failure states are derived from this, and they must never be
conflated (see audit helpers in agent_pipeline.py):

  NOT_WIRED  — the implementing symbol/tool factory does not import or resolve.
               The architecture is missing or not connected. Detected statically
               at preflight, before any stage runs.
  NO_OUTPUT  — the part is wired and was enabled by config, yet produced no
               runtime artifact. Detected post-stage, and only for parts whose
               wiring already passed.

Keeping this declarative means the runtime preflight and the test-suite audit
share one definition: a renamed, deleted, or un-wired component is caught the
same way in both places.
"""
from __future__ import annotations

import importlib
import sys
from dataclasses import dataclass, field

# Phase identifiers — must match the run_step() names used in agent_pipeline.py.
CLASSIFY = "classification"
DOCKERFILE = "dockerfile generation"
VALIDATION_GATE = "post-generation validation gate"
REPAIR = "dockerfile repair"
INSTALL_GUIDE = "install guide generation"

# Module-path prefixes tried in order when resolving a "module:attr" symbol, so
# the manifest works whether the package is imported as RepoBuilderAgent.src.*,
# src.*, or with src/ already on sys.path (the three ways the agent runs).
_IMPORT_PREFIXES = ("RepoBuilderAgent.src.", "src.", "")


@dataclass(frozen=True)
class Component:
    """A wireable subpart of the architecture.

    kind: "primary" (the stage's main deliverable) | "subpart" (an optional,
    flag-gated layer) | "tool" (a ReAct tool factory).
    artifact: template with a ``{repo}`` placeholder, relative to ``dir_key``;
    empty string means the component has no per-repo artifact (tools, traces
    that are conditional on runtime work) and is wiring-checked only.
    """

    key: str
    label: str
    kind: str
    phase: str
    symbol: str
    dir_key: str = ""
    artifact: str = ""

    @property
    def checks_output(self) -> bool:
        return bool(self.artifact and self.dir_key)


# ── Retrieval strategy → implementing symbol ─────────────────────────────────
# L1 file discovery is one subpart whose implementation depends on the selected
# strategy, so the wiring audit checks the symbol that will actually run.
RETRIEVAL_SYMBOLS = {
    "bm25": "retrieval.repo_fingerprint:select_files_by_bm25",
    "one_shot_fingerprint_budgeted": "retrieval.repo_fingerprint:select_files_by_bm25_budgeted",
    "one_shot_fingerprint": "retrieval.repo_fingerprint:fingerprint",
    "neural_embedding": "stages.stage_1_repository_installation_analysis.agent_classify:select_files_by_neural_embedding",
    "iterative_react": "stages.stage_1_repository_installation_analysis.l1_relevant_file_discovery:select_files_by_iterative_react",
}

_S1 = "stages.stage_1_repository_installation_analysis"
_S2 = "stages.stage_2_dockerfile_generation"
_S3 = "stages.stage_3_iterative_dockerfile_repair"
_S4 = "stages.stage_4_install_guide"
_RT = "agent_tools.react_loop_tools"


# ── ReAct tool factories ─────────────────────────────────────────────────────
# Every tool the architecture exposes. `consumers` is informational: which agent
# wires the tool, used to decide whether the tool is active under a given config.
@dataclass(frozen=True)
class Tool:
    key: str
    label: str
    factory: str  # attribute name in react_loop_tools
    consumers: tuple = field(default_factory=tuple)
    kind: str = "tool"

    @property
    def symbol(self) -> str:
        return f"{_RT}:{self.factory}"


TOOLS: tuple[Tool, ...] = (
    Tool("tool.read_file", "read_file", "build_read_file_tool", ("l1_react", "repair_repo")),
    Tool("tool.list_tree", "list_tree", "build_list_tree_tool", ("l1_react", "repair_repo")),
    Tool("tool.search_pattern", "search_pattern", "build_search_pattern_tool", ("l1_react", "repair_repo")),
    Tool("tool.read_gitlog", "read_gitlog", "build_read_gitlog_tool", ("l1_react",)),
    Tool("tool.search_commits", "search_commits", "build_search_commits_tool", ("l1_react",)),
    Tool("tool.search_structure_paths", "search_structure_paths", "build_search_structure_paths_tool", ("l1_react",)),
    Tool("tool.select_default_files", "select_default_files", "build_select_default_files_tool", ("l1_react",)),
    Tool("tool.fetch_file_context", "fetch_file_context", "build_fetch_file_context_tool", ("l3_validation",)),
    Tool("tool.list_selected_files", "list_selected_files", "build_list_selected_files_tool", ("l3_validation",)),
    Tool("tool.search_selected_files", "search_selected_files", "build_search_selected_files_tool", ("l3_validation",)),
    Tool("tool.think", "think", "build_think_tool", ("l1_react", "l3_validation", "repair")),
    Tool("tool.get_dockerfile_snippet", "get_dockerfile_snippet", "build_get_dockerfile_snippet_tool", ("repair_snippet",)),
    Tool("tool.hadolint_snippet", "hadolint_snippet", "build_hadolint_snippet_tool", ("repair",)),
)


def resolve_symbol(dotted: str) -> tuple[bool, str]:
    """Resolve a 'module.path:attr' symbol, trying each import prefix.

    Returns (ok, detail). ok=False means the architecture is NOT WIRED here:
    the module won't import or the attribute is absent.
    """
    if ":" not in dotted:
        return False, f"malformed symbol '{dotted}' (expected 'module:attr')"
    module_path, attr = dotted.split(":", 1)
    last_error = "no import prefix succeeded"
    # Some stage modules call parser.parse_args() at import time. Importing them
    # here to verify wiring would otherwise parse the *pipeline's* argv with the
    # *stage's* parser and sys.exit. Neutralize argv during import (the test
    # harness does the same) and tolerate a module that exits at import.
    saved_argv = sys.argv
    sys.argv = [saved_argv[0] if saved_argv else "arch_wiring_probe"]
    try:
        for prefix in _IMPORT_PREFIXES:
            try:
                module = importlib.import_module(prefix + module_path)
            except SystemExit as exc:  # module-level parse_args() exited
                last_error = f"import {prefix + module_path} called sys.exit({exc.code}) at import time"
                continue
            except Exception as exc:  # noqa: BLE001 — any import failure means not wired
                last_error = f"import {prefix + module_path} failed: {exc}"
                continue
            if not hasattr(module, attr):
                return False, f"module {prefix + module_path} has no attribute '{attr}'"
            return True, f"{prefix + module_path}:{attr}"
        return False, last_error
    finally:
        sys.argv = saved_argv


def active_components(flags: dict) -> list[Component]:
    """Return the subparts/primaries active under ``flags``.

    flags keys: phase_skips(dict), retrieval_strategy(str), exploration,
    synthesis, validation, scratchpads(bool).
    """
    skips = flags.get("phase_skips", {})
    out: list[Component] = []

    if not skips.get("classify"):
        strategy = flags.get("retrieval_strategy", "one_shot_fingerprint")
        retrieval_symbol = RETRIEVAL_SYMBOLS.get(strategy, RETRIEVAL_SYMBOLS["one_shot_fingerprint"])
        out.append(Component("stage1.classification", "classification result", "primary", CLASSIFY,
                             f"{_S1}.agent_classify:analyze_repository", "results_dir", "{repo}.yaml"))
        out.append(Component("stage1.l1_file_discovery", f"L1 file discovery ({strategy})", "subpart", CLASSIFY,
                             retrieval_symbol, "summaries_dir", "{repo}.selected-files.yaml"))
        if strategy == "iterative_react":
            out.append(Component("stage1.l1_react_trace", "L1 ReAct trace", "subpart", CLASSIFY,
                                 RETRIEVAL_SYMBOLS["iterative_react"], "summaries_dir", "{repo}.react-trace.yaml"))
        if flags.get("exploration"):
            out.append(Component("stage1.exploration", "L1 exploration", "subpart", CLASSIFY,
                                 f"{_S1}.architecture_state_graph:run_architecture_state_graph",
                                 "summaries_dir", "{repo}.exploration.yaml"))
        if flags.get("synthesis"):
            out.append(Component("stage1.l2_synthesis", "L2 synthesis", "subpart", CLASSIFY,
                                 f"{_S1}.l2_install_command_extraction:run_l2_synthesis_loop",
                                 "summaries_dir", "{repo}.synthesis.yaml"))
        if flags.get("validation"):
            out.append(Component("stage1.l3_validation", "L3 validation", "subpart", CLASSIFY,
                                 f"{_S1}.classify_validation_loop:run_classify_validation_loop",
                                 "summaries_dir", "{repo}.validation.yaml"))
        if flags.get("scratchpads"):
            out.append(Component("stage1.scratchpad", "architecture scratchpad", "subpart", CLASSIFY,
                                 f"{_S1}.scratchpad_payloads:build_architecture_scratchpad_payload",
                                 "summaries_dir", "{repo}.architecture-scratchpad.yaml"))

    if not skips.get("dockerfile"):
        out.append(Component("stage2.dockerfile", "Dockerfile", "primary", DOCKERFILE,
                             f"{_S2}.agent_dockerfile:generate_dockerfile", "dockerfiles_dir", "{repo}.Dockerfile"))

    if not skips.get("validation_gate"):
        out.append(Component("stage2.validation_gate", "validation gate", "primary", VALIDATION_GATE,
                             f"{_S2}.agent_validation_gate:validate_repository",
                             "summaries_dir", "{repo}.postgen-validation.yaml"))

    if not skips.get("repair"):
        out.append(Component("stage3.repair", "repair report", "primary", REPAIR,
                             f"{_S3}.agent_dockerfile_repair:repair_repository", "reports_dir", "{repo}/report.yaml"))
        # The L3 ReAct repair loop is a gated component: the baseline repair is a
        # single-shot LLM call, so the ReAct loop only runs when react_repair is on.
        # Wiring-checked only — its per-attempt trace is written *only* when a build
        # actually fails, so absence is not NO_OUTPUT.
        if flags.get("react_repair"):
            out.append(Component("stage3.l3_react_loop", "L3 repair ReAct loop", "subpart", REPAIR,
                                 f"{_S3}.l3_react_loop:run_l3_dockerfile_repair_react"))

    if not skips.get("install_guide"):
        out.append(Component("stage4.install_guide", "INSTALL.md", "primary", INSTALL_GUIDE,
                             f"{_S4}.agent_install_guide:generate_install_guide", "install_guides_dir", "{repo}/INSTALL.md"))

    return out


def active_tools(flags: dict) -> list[Tool]:
    """Return the ReAct tools wired into the agents active under ``flags``."""
    skips = flags.get("phase_skips", {})
    consumers: set[str] = set()

    if not skips.get("classify"):
        if flags.get("retrieval_strategy") == "iterative_react":
            consumers.add("l1_react")
        if flags.get("validation"):
            consumers.add("l3_validation")
    # The repair ReAct tools (think/hadolint, snippet, repo tools) exist only when
    # the ReAct repair agent is active; the baseline single-shot repair uses none.
    if not skips.get("repair") and flags.get("react_repair"):
        consumers.add("repair")
        if flags.get("snippet_tools"):
            consumers.add("repair_snippet")
        if flags.get("repair_repo_tools"):
            consumers.add("repair_repo")

    return [t for t in TOOLS if consumers.intersection(t.consumers)]
