from typing import Any, Callable

from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import create_react_agent

try:
    from RepoBuilderAgent.src.agent_tools.react_loop_tools import (
        build_fetch_file_context_tool,
        build_hadolint_snippet_tool,
        build_list_selected_files_tool,
        build_search_selected_files_tool,
        build_think_tool,
    )
except ImportError:
    from agent_tools.react_loop_tools import (
        build_fetch_file_context_tool,
        build_hadolint_snippet_tool,
        build_list_selected_files_tool,
        build_search_selected_files_tool,
        build_think_tool,
    )


def _deterministic_signal_scan(selected_files: list[str], markers: tuple[str, ...], limit: int = 8) -> list[str]:
    return [path for path in selected_files if any(marker in path.lower() for marker in markers)][: max(1, int(limit))]


async def _invoke_signal_subagent(
    *,
    name: str,
    repo_name: str,
    summary: str,
    prompt_body: str,
    deterministic_fallback: list[str],
    classification_timeout: int,
    new_prebuilt_chat_model: Callable[[int], Any],
    extract_agent_payload: Callable[[dict[str, Any]], Any],
    extract_agent_trace: Callable[[dict[str, Any]], list[dict[str, Any]]],
    normalize_text_list: Callable[[Any], list[str]],
    think,
    list_selected_files,
    search_selected_files,
    fetch_file_context,
) -> dict[str, Any]:
    signal_agent = create_react_agent(
        model=new_prebuilt_chat_model(classification_timeout),
        tools=[think, list_selected_files, search_selected_files, fetch_file_context],
        prompt=(
            "You are a focused L2 signal sub-agent. "
            "Use tools to find high-value evidence for your assigned signal type. "
            "Return YAML-compatible fields only."
        ),
    )


    subagent_prompt = (
        f"Repository: {repo_name}\n"
        f"Sub-agent: {name}\n"
        + prompt_body
        + "\n"
        "Use think between major tool decisions.\n"
        "Return keys: thought, signal (list), notes (list), done (bool).\n\n"
        "SUMMARY_EVIDENCE:\n"
        + summary
    )

    result = await signal_agent.ainvoke(
        {"messages": [{"role": "user", "content": subagent_prompt}]},
        config={"configurable": {"thread_id": f"{repo_name}:l2-signal:{name}"}, "recursion_limit": 16},
    )
    payload = extract_agent_payload(result)
    signal = normalize_text_list(payload.get("signal") if isinstance(payload, dict) else [])
    notes = normalize_text_list(payload.get("notes") if isinstance(payload, dict) else [])
    done_flag = bool(payload.get("done", False)) if isinstance(payload, dict) else False
    trace = extract_agent_trace(result)

    if not signal:
        signal = deterministic_fallback

    return {
        "name": name,
        "mode": "llm_subagent",
        "signal": signal[:8],
        "notes": notes[:8],
        "steps": len(trace),
        "stop_reason": "model_done" if done_flag else "agent_converged",
    }


async def _invoke_gap_subagent(
    *,
    repo_name: str,
    summary: str,
    selected_lower: list[str],
    classification_timeout: int,
    new_prebuilt_chat_model: Callable[[int], Any],
    extract_agent_payload: Callable[[dict[str, Any]], Any],
    extract_agent_trace: Callable[[dict[str, Any]], list[dict[str, Any]]],
    normalize_text_list: Callable[[Any], list[str]],
    think,
    list_selected_files,
    search_selected_files,
    fetch_file_context,
) -> dict[str, Any]:
    gap_agent = create_react_agent(
        model=new_prebuilt_chat_model(classification_timeout),
        tools=[think, list_selected_files, search_selected_files, fetch_file_context],
        prompt=(
            "You are the L2 gap-check sub-agent. "
            "Detect whether manifest, docker, and test evidence are present. "
            "Return YAML-compatible fields only."
        ),
    )

    gap_prompt = (
        f"Repository: {repo_name}\n"
        "Check evidence presence and return booleans.\n"
        "Use think between major tool decisions.\n"
        "Return keys: thought, has_manifest (bool), has_docker (bool), has_tests (bool), notes (list), done (bool).\n\n"
        "SUMMARY_EVIDENCE:\n"
        + summary
    )

    result = await gap_agent.ainvoke(
        {"messages": [{"role": "user", "content": gap_prompt}]},
        config={"configurable": {"thread_id": f"{repo_name}:l2-signal:gap-check"}, "recursion_limit": 16},
    )
    payload = extract_agent_payload(result)
    trace = extract_agent_trace(result)
    notes = normalize_text_list(payload.get("notes") if isinstance(payload, dict) else [])
    done_flag = bool(payload.get("done", False)) if isinstance(payload, dict) else False

    def _as_bool(field: str, fallback: bool) -> bool:
        if not isinstance(payload, dict):
            return fallback
        value = payload.get(field)
        return bool(value) if isinstance(value, bool) else fallback

    fallback_manifest = any(
        any(marker in path for marker in ("package.json", "pyproject.toml", "requirements", "go.mod", "cargo.toml", "pom.xml", "build.gradle"))
        for path in selected_lower
    )
    fallback_docker = any("dockerfile" in path or "docker-compose" in path for path in selected_lower)
    fallback_tests = any("test" in path or "spec" in path for path in selected_lower)

    return {
        "name": "gap-check",
        "mode": "llm_subagent",
        "signal": {
            "has_manifest": _as_bool("has_manifest", fallback_manifest),
            "has_docker": _as_bool("has_docker", fallback_docker),
            "has_tests": _as_bool("has_tests", fallback_tests),
        },
        "notes": notes[:8],
        "steps": len(trace),
        "stop_reason": "model_done" if done_flag else "agent_converged",
    }


async def _run_synthesis_subagents(
    *,
    repo_name: str,
    selected_files: list[str],
    summary: str,
    classification_timeout: int,
    new_prebuilt_chat_model: Callable[[int], Any],
    extract_agent_payload: Callable[[dict[str, Any]], Any],
    extract_agent_trace: Callable[[dict[str, Any]], list[dict[str, Any]]],
    normalize_text_list: Callable[[Any], list[str]],
    think,
    list_selected_files,
    search_selected_files,
    fetch_file_context,
) -> list[dict[str, Any]]:
    class SignalSubagentState(dict):
        pass

    selected_lower = [path.lower() for path in selected_files]

    build_fallback = _deterministic_signal_scan(
        selected_files,
        ("package.json", "pyproject.toml", "requirements", "go.mod", "cargo.toml", "pom.xml", "build.gradle"),
    )
    runtime_fallback = _deterministic_signal_scan(
        selected_files,
        ("dockerfile", "docker-compose", "runtime", "launch", "entrypoint"),
    )
    verification_fallback = _deterministic_signal_scan(
        selected_files,
        ("test", "spec", ".github/workflows", "ci"),
    )

    async def build_node(_: SignalSubagentState) -> dict[str, Any]:
        return {
            "build": await _invoke_signal_subagent(
                name="build-signal-scan",
                repo_name=repo_name,
                summary=summary,
                prompt_body="Identify top manifest/build-signal files relevant for installation/build planning.",
                deterministic_fallback=build_fallback,
                classification_timeout=classification_timeout,
                new_prebuilt_chat_model=new_prebuilt_chat_model,
                extract_agent_payload=extract_agent_payload,
                extract_agent_trace=extract_agent_trace,
                normalize_text_list=normalize_text_list,
                think=think,
                list_selected_files=list_selected_files,
                search_selected_files=search_selected_files,
                fetch_file_context=fetch_file_context,
            )
        }

    async def runtime_node(_: SignalSubagentState) -> dict[str, Any]:
        return {
            "runtime": await _invoke_signal_subagent(
                name="runtime-signal-scan",
                repo_name=repo_name,
                summary=summary,
                prompt_body="Identify top runtime/containerization-signal files relevant for execution behavior.",
                deterministic_fallback=runtime_fallback,
                classification_timeout=classification_timeout,
                new_prebuilt_chat_model=new_prebuilt_chat_model,
                extract_agent_payload=extract_agent_payload,
                extract_agent_trace=extract_agent_trace,
                normalize_text_list=normalize_text_list,
                think=think,
                list_selected_files=list_selected_files,
                search_selected_files=search_selected_files,
                fetch_file_context=fetch_file_context,
            )
        }

    async def verification_node(_: SignalSubagentState) -> dict[str, Any]:
        return {
            "verification": await _invoke_signal_subagent(
                name="verification-signal-scan",
                repo_name=repo_name,
                summary=summary,
                prompt_body="Identify top verification/test/CI signal files relevant for confidence and validation.",
                deterministic_fallback=verification_fallback,
                classification_timeout=classification_timeout,
                new_prebuilt_chat_model=new_prebuilt_chat_model,
                extract_agent_payload=extract_agent_payload,
                extract_agent_trace=extract_agent_trace,
                normalize_text_list=normalize_text_list,
                think=think,
                list_selected_files=list_selected_files,
                search_selected_files=search_selected_files,
                fetch_file_context=fetch_file_context,
            )
        }

    async def gap_node(_: SignalSubagentState) -> dict[str, Any]:
        return {
            "gap": await _invoke_gap_subagent(
                repo_name=repo_name,
                summary=summary,
                selected_lower=selected_lower,
                classification_timeout=classification_timeout,
                new_prebuilt_chat_model=new_prebuilt_chat_model,
                extract_agent_payload=extract_agent_payload,
                extract_agent_trace=extract_agent_trace,
                normalize_text_list=normalize_text_list,
                think=think,
                list_selected_files=list_selected_files,
                search_selected_files=search_selected_files,
                fetch_file_context=fetch_file_context,
            )
        }

    subgraph = StateGraph(SignalSubagentState)
    subgraph.add_node("build", build_node)
    subgraph.add_node("runtime", runtime_node)
    subgraph.add_node("verification", verification_node)
    subgraph.add_node("gap", gap_node)
    subgraph.add_edge(START, "build")
    subgraph.add_edge(START, "runtime")
    subgraph.add_edge(START, "verification")
    subgraph.add_edge(START, "gap")
    subgraph.add_edge("build", END)
    subgraph.add_edge("runtime", END)
    subgraph.add_edge("verification", END)
    subgraph.add_edge("gap", END)
    compiled = subgraph.compile()

    results = await compiled.ainvoke(
        {},
        config={"configurable": {"thread_id": f"{repo_name}:l2-signals"}},
    )
    return [results["build"], results["runtime"], results["verification"], results["gap"]]


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
    synthesis_review_rounds: int,
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
    fetch_file_context = build_fetch_file_context_tool(file_context_by_path)
    list_selected_files = build_list_selected_files_tool(selected_snapshot)
    search_selected_files = build_search_selected_files_tool(selected_snapshot)
    think = build_think_tool()
    hadolint_check = build_hadolint_snippet_tool()

    subagent_outputs = (
        await _run_synthesis_subagents(
            repo_name=repo_name,
            selected_files=selected_snapshot,
            summary=summary,
            classification_timeout=classification_timeout,
            new_prebuilt_chat_model=new_prebuilt_chat_model,
            extract_agent_payload=extract_agent_payload,
            extract_agent_trace=extract_agent_trace,
            normalize_text_list=normalize_text_list,
            think=think,
            list_selected_files=list_selected_files,
            search_selected_files=search_selected_files,
            fetch_file_context=fetch_file_context,
        )
        if synthesis_subagents_enabled
        else []
    )

    synthesis_generator = create_react_agent(
        model=new_prebuilt_chat_model(classification_timeout),
        tools=[think, list_selected_files, search_selected_files, fetch_file_context, hadolint_check],
        prompt=(
            "You are the L2 synthesis generator agent. Use tools whenever evidence is missing. "
            "Use list_selected_files/search_selected_files to inspect available evidence before fetching file content. "
            "If you draft a Dockerfile snippet as part of a hypothesis, call run_hadolint_on_snippet to lint it before finalizing. "
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

    generation_result = await synthesis_generator.ainvoke(
        {"messages": [{"role": "user", "content": synthesis_prompt}]},
        config={"configurable": {"thread_id": f"{repo_name}:l2"}, "recursion_limit": max(8, int(synthesis_react_max_steps) * 4)},
    )
    generation_payload = extract_agent_payload(generation_result)
    updates = normalize_text_list(generation_payload.get("hypothesis_updates") if isinstance(generation_payload, dict) else [])
    risk_updates = normalize_text_list(generation_payload.get("risk_updates") if isinstance(generation_payload, dict) else [])
    done_flag = bool(generation_payload.get("done", False)) if isinstance(generation_payload, dict) else False

    generated_hypotheses = list(dict.fromkeys(item.strip() for item in (base_hypotheses + updates) if item and item.strip()))
    generated_risks = list(dict.fromkeys(item.strip() for item in (risk_notes + risk_updates) if item and item.strip()))
    generator_trace = extract_agent_trace(generation_result)
    generator_stop_reason = "model_done" if done_flag else "agent_converged"

    synthesis_reviewer = create_react_agent(
        model=new_prebuilt_chat_model(classification_timeout),
        tools=[think, list_selected_files, search_selected_files, fetch_file_context, hadolint_check],
        prompt=(
            "You are the L2 synthesis reviewer agent. Critique the generator output for missing evidence and weak assumptions. "
            "Use tools to verify claims, and use run_hadolint_on_snippet when Dockerfile snippets appear in hypotheses. "
            "Return YAML-compatible fields only."
        ),
    )

    review_hypotheses = generated_hypotheses[:]
    review_risks = generated_risks[:]
    critique_notes: list[str] = []
    reviewer_trace: list[dict[str, Any]] = []
    review_accepted = False
    review_done_flag = False
    review_rounds_executed = 0

    for round_index in range(max(1, int(synthesis_review_rounds))):
        reviewer_prompt = (
            f"Repository: {repo_url}\n"
            f"Review and refine generated synthesis output (round {round_index + 1}/{max(1, int(synthesis_review_rounds))}).\n"
            "Return keys: thought, accepted (bool), revised_hypotheses (list), revised_risks (list), critique_notes (list), done (bool).\n\n"
            "GENERATOR_HYPOTHESES:\n"
            + "\n".join(f"- {item}" for item in review_hypotheses[:30])
            + "\n\nGENERATOR_RISKS:\n"
            + ("\n".join(f"- {item}" for item in review_risks[:30]) if review_risks else "- (none)")
            + "\n\nSUBAGENT_SIGNALS:\n"
            + str(subagent_outputs[:4])
            + "\n\nSUMMARY_EVIDENCE:\n"
            + summary
        )

        review_result = await synthesis_reviewer.ainvoke(
            {"messages": [{"role": "user", "content": reviewer_prompt}]},
            config={
                "configurable": {"thread_id": f"{repo_name}:l2-review:r{round_index + 1}"},
                "recursion_limit": max(8, int(synthesis_react_max_steps) * 4),
            },
        )
        review_rounds_executed += 1
        review_payload = extract_agent_payload(review_result)
        revised_hypotheses = normalize_text_list(review_payload.get("revised_hypotheses") if isinstance(review_payload, dict) else [])
        revised_risks = normalize_text_list(review_payload.get("revised_risks") if isinstance(review_payload, dict) else [])
        round_critique_notes = normalize_text_list(review_payload.get("critique_notes") if isinstance(review_payload, dict) else [])
        critique_notes.extend(round_critique_notes)
        review_done_flag = bool(review_payload.get("done", False)) if isinstance(review_payload, dict) else False
        review_accepted = bool(review_payload.get("accepted", False)) if isinstance(review_payload, dict) else False

        if revised_hypotheses:
            review_hypotheses = revised_hypotheses
        if revised_risks:
            review_risks = revised_risks

        round_trace = extract_agent_trace(review_result)
        for event in round_trace:
            if isinstance(event, dict):
                reviewer_trace.append({"round": round_index + 1, **event})

        if review_done_flag or review_accepted:
            break

    dedup_hypotheses = list(dict.fromkeys(item.strip() for item in review_hypotheses if item and item.strip()))
    dedup_risks = list(dict.fromkeys(item.strip() for item in review_risks if item and item.strip()))
    loop_trace: list[dict[str, Any]] = []
    for event in generator_trace:
        if isinstance(event, dict):
            loop_trace.append({"phase": "l2_generator", **event})
    for event in reviewer_trace:
        if isinstance(event, dict):
            loop_trace.append({"phase": "l2_reviewer", **event})

    stop_reason = "reviewer_done" if review_done_flag else "reviewer_converged"

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
    run_classify_validation = bool(gap_penalty > 0 or confidence < 0.78 or dedup_risks)
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
            "generator_stop_reason": generator_stop_reason,
            "generator_steps": len(generator_trace),
            "reviewer_steps": len(reviewer_trace),
            "review_rounds_requested": max(1, int(synthesis_review_rounds)),
            "review_rounds_executed": review_rounds_executed,
        },
        "subagent_outputs": subagent_outputs,
        "review": {
            "enabled": True,
            "accepted": review_accepted,
            "rounds": review_rounds_executed,
            "critique_notes": critique_notes[:30],
        },
        "loop_trace": loop_trace,
        "loop_checkpoint": {
            "stage": "l2_synthesis",
            "completed": True,
            "next_stage": "classify_validation" if run_classify_validation else "terminal",
        },
        "abstraction_l2": {
            "confidence": round(confidence, 4),
            "complete": not run_classify_validation,
            "run_classify_validation": run_classify_validation,
            "reasons": transition_reasons,
        },
        "transition_policy": {
            "run_classify_validation": run_classify_validation,
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
