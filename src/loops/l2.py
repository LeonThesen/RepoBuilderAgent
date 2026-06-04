from typing import Any, Callable

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


def _run_synthesis_subagents(selected_files: list[str]) -> list[dict[str, Any]]:
    selected_lower = [path.lower() for path in selected_files]
    return [
        {
            "name": "build-signal-scan",
            "signal": [
                path
                for path in selected_files
                if any(marker in path.lower() for marker in ("package.json", "pyproject.toml", "requirements", "go.mod", "cargo.toml", "pom.xml", "build.gradle"))
            ][:8],
        },
        {
            "name": "runtime-signal-scan",
            "signal": [
                path
                for path in selected_files
                if any(marker in path.lower() for marker in ("dockerfile", "docker-compose", "runtime", "launch", "entrypoint"))
            ][:8],
        },
        {
            "name": "verification-signal-scan",
            "signal": [path for path in selected_files if any(marker in path.lower() for marker in ("test", "spec", ".github/workflows", "ci"))][:8],
        },
        {
            "name": "gap-check",
            "signal": {
                "has_manifest": any(
                    any(marker in path for marker in ("package.json", "pyproject.toml", "requirements", "go.mod", "cargo.toml", "pom.xml", "build.gradle"))
                    for path in selected_lower
                ),
                "has_docker": any("dockerfile" in path or "docker-compose" in path for path in selected_lower),
                "has_tests": any("test" in path or "spec" in path for path in selected_lower),
            },
        },
    ]


async def run_l2_synthesis_loop(
    *,
    repo_url: str,
    repo_name: str,
    selected_files: list[str],
    summary: str,
    exploration_artifact: dict[str, Any],
    file_context_by_path: dict[str, str],
    classification_timeout: int,
    synthesis_react_max_steps: int,
    synthesis_subagents_enabled: bool,
    new_prebuilt_chat_model: Callable[[int], Any],
    extract_agent_payload: Callable[[dict[str, Any]], Any],
    extract_agent_trace: Callable[[dict[str, Any]], list[dict[str, Any]]],
    normalize_text_list: Callable[[Any], list[str]],
) -> tuple[dict[str, Any], list[dict[str, Any]], str]:
    selected_snapshot = selected_files[:]
    selected_lower = [path.lower() for path in selected_snapshot]

    base_hypotheses: list[str] = []
    if any("package.json" in path for path in selected_lower):
        base_hypotheses.append("Use Node.js package-manager driven install/build flow inferred from package.json.")
    if any("pyproject.toml" in path or "requirements.txt" in path for path in selected_lower):
        base_hypotheses.append("Use Python dependency installation before running project-specific build/test steps.")
    if any("go.mod" in path for path in selected_lower):
        base_hypotheses.append("Use Go module restore and compile/test verification path.")
    if any("cargo.toml" in path for path in selected_lower):
        base_hypotheses.append("Use Cargo dependency fetch and build workflow with Rust toolchain in image.")
    if not base_hypotheses:
        base_hypotheses.append("Fallback to minimal deterministic build strategy based on discovered repository structure.")

    risk_notes = [
        "manifest_evidence_missing" if exploration_artifact["evidence_gaps"]["manifest_evidence_missing"] else None,
        "docker_evidence_missing" if exploration_artifact["evidence_gaps"]["docker_evidence_missing"] else None,
        "test_evidence_missing" if exploration_artifact["evidence_gaps"]["test_evidence_missing"] else None,
    ]
    risk_notes = [item for item in risk_notes if item]
    subagent_outputs = _run_synthesis_subagents(selected_snapshot) if synthesis_subagents_enabled else []

    fetch_file_context = build_fetch_file_context_tool(file_context_by_path)
    list_selected_files = build_list_selected_files_tool(selected_snapshot)
    search_selected_files = build_search_selected_files_tool(selected_snapshot)
    think = build_think_tool()

    synthesis_agent = create_react_agent(
        model=new_prebuilt_chat_model(classification_timeout),
        tools=[think, list_selected_files, search_selected_files, fetch_file_context],
        prompt=(
            "You are the L2 synthesis ReAct agent. Use tools whenever evidence is missing. "
            "Use list_selected_files/search_selected_files to inspect available evidence before fetching file content. "
            "Use think for brief intent notes before/after key tool decisions. "
            "Return YAML-compatible fields only."
        ),
    )

    synthesis_prompt = (
        f"Repository: {repo_url}\n"
        "Improve build strategy hypotheses and risk notes using evidence.\n"
        "Use think between major tool decisions.\n"
        "Return keys: thought, hypothesis_updates (list), risk_updates (list), selected_files (list), done (bool).\n\n"
        "CURRENT_HYPOTHESES:\n"
        + "\n".join(f"- {item}" for item in base_hypotheses[:20])
        + "\n\nCURRENT_RISKS:\n"
        + ("\n".join(f"- {item}" for item in risk_notes[:20]) if risk_notes else "- (none)")
        + "\n\nSUBAGENT_SIGNALS:\n"
        + str(subagent_outputs[:4])
        + "\n\nSUMMARY_EVIDENCE:\n"
        + summary
    )

    result = await synthesis_agent.ainvoke(
        {"messages": [{"role": "user", "content": synthesis_prompt}]},
        config={"configurable": {"thread_id": f"{repo_name}:l2"}, "recursion_limit": max(8, int(synthesis_react_max_steps) * 4)},
    )
    payload = extract_agent_payload(result)
    updates = normalize_text_list(payload.get("hypothesis_updates") if isinstance(payload, dict) else [])
    risk_updates = normalize_text_list(payload.get("risk_updates") if isinstance(payload, dict) else [])
    done_flag = bool(payload.get("done", False)) if isinstance(payload, dict) else False

    dedup_hypotheses = list(dict.fromkeys(item.strip() for item in (base_hypotheses + updates) if item and item.strip()))
    dedup_risks = list(dict.fromkeys(item.strip() for item in (risk_notes + risk_updates) if item and item.strip()))
    loop_trace = extract_agent_trace(result)
    stop_reason = "model_done" if done_flag else "agent_converged"

    has_manifest = any(
        any(marker in path for marker in ("package.json", "pyproject.toml", "requirements", "go.mod", "cargo.toml", "pom.xml", "build.gradle"))
        for path in selected_lower
    )
    has_docker = any("dockerfile" in path or "docker-compose" in path for path in selected_lower)
    has_tests = any("test" in path or "spec" in path for path in selected_lower)
    has_ci = any(path.startswith(".github/workflows/") for path in selected_lower)
    coverage_hits = sum(1 for hit in (has_manifest, has_docker, has_tests, has_ci) if hit)
    gap_penalty = sum(1 for flag in exploration_artifact.get("evidence_gaps", {}).values() if flag)
    confidence = max(0.0, min(1.0, 0.35 + coverage_hits * 0.15 - gap_penalty * 0.08))
    run_l3 = bool(gap_penalty > 0 or confidence < 0.78 or dedup_risks)
    transition_reasons: list[str] = []
    if gap_penalty > 0:
        transition_reasons.append("unresolved_evidence_gaps")
    if confidence < 0.78:
        transition_reasons.append("confidence_below_threshold")
    if dedup_risks:
        transition_reasons.append("risk_notes_present")
    if not transition_reasons:
        transition_reasons.append("synthesis_sufficient")

    synthesis_artifact = {
        "repo": repo_url,
        "stage": "synthesis",
        "react": {
            "steps": len(loop_trace),
            "max_steps": max(1, int(synthesis_react_max_steps)),
            "stop_reason": stop_reason,
            "subagents_enabled": synthesis_subagents_enabled,
            "subagent_count": len(subagent_outputs),
        },
        "subagent_outputs": subagent_outputs,
        "loop_trace": loop_trace,
        "loop_checkpoint": {
            "stage": "l2_synthesis",
            "completed": True,
            "next_stage": "l3_validation" if run_l3 else "terminal",
        },
        "abstraction_l2": {
            "confidence": round(confidence, 4),
            "complete": not run_l3,
            "run_l3": run_l3,
            "reasons": transition_reasons,
        },
        "transition_policy": {
            "run_l3": run_l3,
            "threshold": 0.78,
            "confidence": round(confidence, 4),
            "reasons": transition_reasons,
        },
        "build_strategy_hypotheses": dedup_hypotheses[:40],
        "dependency_assumptions": {
            "system_build_tools_required": any(marker in path for marker in ("binding.gyp", "cmake", "makefile") for path in selected_lower),
            "network_install_steps_likely": any(
                marker in path
                for marker in ("package.json", "requirements.txt", "pyproject.toml", "go.mod", "cargo.toml")
                for path in selected_lower
            ),
        },
        "risk_notes": dedup_risks[:40],
    }
    return synthesis_artifact, loop_trace, stop_reason
