from typing import Any, Callable

import yaml
from langgraph.prebuilt import create_react_agent

try:
    from RepoBuilderAgent.src.loops.tools import (
        build_fetch_file_context_tool,
        build_list_selected_files_tool,
        build_search_selected_files_tool,
        build_think_tool,
    )
except ImportError:
    from loops.tools import (
        build_fetch_file_context_tool,
        build_list_selected_files_tool,
        build_search_selected_files_tool,
        build_think_tool,
    )


def _truncate_for_prompt(text: str, max_chars: int = 12000) -> str:
    if len(text) <= max_chars:
        return text
    keep_each = max(1000, max_chars // 2)
    return text[:keep_each] + "\n... [truncated] ...\n" + text[-keep_each:]


def _compact_synthesis_for_validation(synthesis_artifact: dict[str, Any]) -> dict[str, Any]:
    return {
        "repo": synthesis_artifact.get("repo"),
        "stage": synthesis_artifact.get("stage"),
        "react": synthesis_artifact.get("react"),
        "build_strategy_hypotheses": (synthesis_artifact.get("build_strategy_hypotheses") or [])[:12],
        "dependency_assumptions": synthesis_artifact.get("dependency_assumptions") or {},
        "risk_notes": (synthesis_artifact.get("risk_notes") or [])[:12],
        "subagent_outputs": (synthesis_artifact.get("subagent_outputs") or [])[:3],
    }


async def run_l3_validation_loop(
    *,
    repo_url: str,
    summary: str,
    synthesis_artifact: dict[str, Any],
    selected_files: list[str],
    file_context_by_path: dict[str, str],
    classification_timeout: int,
    validation_react_max_steps: int,
    new_prebuilt_chat_model: Callable[[int], Any],
    extract_agent_payload: Callable[[dict[str, Any]], Any],
    extract_agent_trace: Callable[[dict[str, Any]], list[dict[str, Any]]],
    normalize_text_list: Callable[[Any], list[str]],
    normalize_validation_checks: Callable[[Any], dict[str, dict[str, str]]],
) -> tuple[dict[str, Any], list[dict[str, Any]], str]:
    selected_lower = [path.lower() for path in selected_files]
    has_manifest = any(
        any(marker in path for marker in ("package.json", "requirements.txt", "pyproject.toml", "go.mod", "cargo.toml", "pom.xml", "build.gradle"))
        for path in selected_lower
    )
    has_ci = any(path.startswith(".github/workflows/") for path in selected_lower)
    has_docker = any("dockerfile" in path or "docker-compose" in path for path in selected_lower)
    has_tests = any("test" in path or "spec" in path for path in selected_lower)

    checks: dict[str, dict[str, str]] = {
        "manifest_evidence": {
            "status": "pass" if has_manifest else "warn",
            "detail": "Manifest/build metadata detected in selected evidence." if has_manifest else "No manifest/build metadata detected in selected evidence.",
        },
        "ci_workflow_evidence": {
            "status": "pass" if has_ci else "warn",
            "detail": "CI workflow files detected." if has_ci else "No CI workflow files detected in selected evidence.",
        },
        "docker_evidence": {
            "status": "pass" if has_docker else "warn",
            "detail": "Docker-related files detected." if has_docker else "No Docker-related files detected in selected evidence.",
        },
        "test_evidence": {
            "status": "pass" if has_tests else "warn",
            "detail": "Test-related files detected." if has_tests else "No test-related files detected in selected evidence.",
        },
        "selected_summary_non_empty": {
            "status": "pass" if bool(summary.strip()) else "fail",
            "detail": "Reduced selected-files summary is non-empty." if bool(summary.strip()) else "Reduced selected-files summary is empty.",
        },
    }

    fetch_file_context = build_fetch_file_context_tool(file_context_by_path)
    list_selected_files = build_list_selected_files_tool(selected_files)
    search_selected_files = build_search_selected_files_tool(selected_files)
    think = build_think_tool()

    validation_agent = create_react_agent(
        model=new_prebuilt_chat_model(classification_timeout),
        tools=[think, list_selected_files, search_selected_files, fetch_file_context],
        prompt=(
            "You are the L3 validation ReAct agent. Use tools whenever evidence is missing. "
            "Use list_selected_files/search_selected_files to inspect available evidence before fetching file content. "
            "Use think for brief intent notes before/after key tool decisions. "
            "Return YAML-compatible fields only."
        ),
    )

    compact_synthesis = _compact_synthesis_for_validation(synthesis_artifact)
    compact_summary = _truncate_for_prompt(summary, max_chars=14000)

    validation_prompt = (
        f"Repository: {repo_url}\n"
        "Validate installation/build evidence coverage and highlight risky gaps.\n"
        "Use think between major tool decisions.\n"
        "Return keys: thought, checks (map), warnings (list), selected_files (list), done (bool).\n"
        "Each checks entry must include status (pass|warn|fail) and detail.\n\n"
        "CURRENT_CHECKS:\n"
        + yaml.dump(checks, sort_keys=False, allow_unicode=True)
        + "\nSYNTHESIS_ARTIFACT:\n"
        + yaml.dump(compact_synthesis, sort_keys=False, allow_unicode=True)
        + "\nSUMMARY_EVIDENCE:\n"
        + compact_summary
    )

    result = await validation_agent.ainvoke(
        {"messages": [{"role": "user", "content": validation_prompt}]},
        config={"configurable": {"thread_id": f"{repo_url}:l3"}, "recursion_limit": max(8, int(validation_react_max_steps) * 4)},
    )
    payload = extract_agent_payload(result)
    parsed_checks = normalize_validation_checks(payload.get("checks") if isinstance(payload, dict) else {})
    warning_updates = normalize_text_list(payload.get("warnings") if isinstance(payload, dict) else [])
    done_flag = bool(payload.get("done", False)) if isinstance(payload, dict) else False

    checks.update(parsed_checks)
    if warning_updates:
        checks["agent_validation_warnings"] = {
            "status": "warn",
            "detail": " | ".join(warning_updates[:6]),
        }

    loop_trace = extract_agent_trace(result)
    stop_reason = "model_done" if done_flag else "agent_converged"

    validation_warnings = [
        key
        for key, value in checks.items()
        if value.get("status") in {"warn", "fail"}
    ]
    fail_count = sum(1 for value in checks.values() if value.get("status") == "fail")
    warn_count = sum(1 for value in checks.values() if value.get("status") == "warn")
    if fail_count == 0 and warn_count == 0:
        outcome_state = "validated"
    elif fail_count == 0:
        outcome_state = "partial"
    else:
        outcome_state = "failure"
    confidence = max(0.0, min(1.0, 1.0 - fail_count * 0.25 - warn_count * 0.08))

    validation_artifact = {
        "repo": repo_url,
        "stage": "validation",
        "react": {
            "steps": len(loop_trace),
            "max_steps": max(1, int(validation_react_max_steps)),
            "stop_reason": stop_reason,
        },
        "loop_checkpoint": {
            "stage": "l3_validation",
            "completed": True,
            "terminal_state": outcome_state,
        },
        "abstraction_l3": {
            "confidence": round(confidence, 4),
            "outcome_state": outcome_state,
            "fail_count": fail_count,
            "warn_count": warn_count,
        },
        "outcome_state": outcome_state,
        "checks": checks,
        "warnings": validation_warnings,
        "loop_trace": loop_trace,
    }
    return validation_artifact, loop_trace, stop_reason
