from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class LoopOutcome:
    """One Stage-1 loop's result: where its artifact was written (if at all), the
    artifact dict, its ReAct loop trace, and why the loop stopped. The three loops
    (exploration, synthesis, validation) share this shape, so the scratchpad builder
    takes three of these instead of twelve scalar params."""
    path: Path | None
    artifact: dict[str, Any]
    loop_trace: list[dict[str, Any]]
    stop_reason: str


@dataclass(frozen=True)
class TokenCounts:
    step1_selection: int
    step2_classification: int
    two_step_total: int


@dataclass(frozen=True)
class SummaryPaths:
    structure_summary: Path
    selected_summary: Path
    selected_files: Path


def build_architecture_scratchpad_payload(
    *,
    repo_url: str,
    retrieval_strategy: str,
    selected_files: list[str],
    exploration: LoopOutcome,
    synthesis: LoopOutcome,
    validation: LoopOutcome,
    tokens: TokenCounts,
    paths: SummaryPaths,
    subagents_enabled: bool,
    budget_behavior: dict[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": "1.0",
        "repo": repo_url,
        "generated_by": "agent_classify.py",
        "contract": {
            "required_sections": ["exploration", "synthesis", "validation"],
            "optional_sections": ["react_trace"],
        },
        "architecture": {
            "retrieval_strategy": retrieval_strategy,
            "exploration": {
                "selected_files_count": len(selected_files),
                "selected_files": selected_files,
                "react_steps": len(exploration.loop_trace),
                "react_stop_reason": exploration.stop_reason,
                "exploration_artifact_path": str(exploration.path) if exploration.path else None,
                "evidence_gaps": exploration.artifact["evidence_gaps"],
                "files_read_by_l1": exploration.artifact.get("files_read_by_l1", []),
                "react_trace": exploration.loop_trace,
            },
            "synthesis": {
                "selected_summary_path": str(paths.selected_summary),
                "structure_summary_path": str(paths.structure_summary),
                "selected_files_path": str(paths.selected_files),
                "synthesis_artifact_path": str(synthesis.path) if synthesis.path else None,
                "react_steps": len(synthesis.loop_trace),
                "react_stop_reason": synthesis.stop_reason,
                "subagents_enabled": subagents_enabled,
                "subagent_outputs": synthesis.artifact["subagent_outputs"],
                "abstraction": synthesis.artifact.get("abstraction_l2", {}),
                "per_source_confidence": synthesis.artifact.get("per_source_confidence", {}),
                "unknown_markers": synthesis.artifact.get("unknown_markers", []),
                "transition_policy": synthesis.artifact.get("transition_policy", {}),
                "loop_checkpoint": synthesis.artifact.get("loop_checkpoint", {}),
                "react_trace": synthesis.loop_trace,
                "build_strategy_hypotheses": synthesis.artifact["build_strategy_hypotheses"],
                "risk_notes": synthesis.artifact["risk_notes"],
                "prompt_tokens": {
                    "step1_selection": tokens.step1_selection,
                    "step2_classification": tokens.step2_classification,
                    "two_step_total": tokens.two_step_total,
                },
                "budget_behavior": budget_behavior,
            },
            "validation": {
                "validation_artifact_path": str(validation.path) if validation.path else None,
                "react_steps": len(validation.loop_trace),
                "react_stop_reason": validation.stop_reason,
                "abstraction": validation.artifact.get("abstraction_classify_validation", {}),
                "outcome_state": validation.artifact.get("outcome_state", "unknown"),
                "loop_checkpoint": validation.artifact.get("loop_checkpoint", {}),
                "react_trace": validation.loop_trace,
                "checks": validation.artifact["checks"],
                "warnings": validation.artifact["warnings"],
            },
        },
    }
