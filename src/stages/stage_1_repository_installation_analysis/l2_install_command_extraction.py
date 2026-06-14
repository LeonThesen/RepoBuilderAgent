from typing import Any, Callable, Optional, TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import create_react_agent

try:
    from RepoBuilderAgent.src.agent_tools.react_loop_tools import (
        build_fetch_file_context_tool,
        build_finalize_tool,
        build_get_dockerfile_snippet_tool,
        build_hadolint_snippet_tool,
        build_list_selected_files_tool,
        build_read_gitlog_tool,
        build_search_commits_tool,
        build_search_selected_files_tool,
        build_think_tool,
        extract_finalize_answer,
        hit_step_limit,
    )
    from RepoBuilderAgent.src.core.common import prompt_path
    from RepoBuilderAgent.src.core.log_utils import log_warn
except ImportError:
    from agent_tools.react_loop_tools import (
        build_fetch_file_context_tool,
        build_finalize_tool,
        build_get_dockerfile_snippet_tool,
        build_hadolint_snippet_tool,
        build_list_selected_files_tool,
        build_read_gitlog_tool,
        build_search_commits_tool,
        build_search_selected_files_tool,
        build_think_tool,
        extract_finalize_answer,
        hit_step_limit,
    )
    from core.common import prompt_path
    from core.log_utils import log_warn


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
        tools=[think, list_selected_files, search_selected_files, fetch_file_context, build_finalize_tool()],
        prompt=prompt_path("PROMPT_L2_SIGNAL_SYSTEM.md").read_text(encoding="utf-8").strip(),
    )

    subagent_prompt = (
        prompt_path("PROMPT_L2_SIGNAL_TASK.md").read_text(encoding="utf-8")
        .replace("{{REPO_NAME}}", repo_name)
        .replace("{{SUBAGENT_NAME}}", name)
        .replace("{{PROMPT_BODY}}", prompt_body)
        .replace("{{SUMMARY_EVIDENCE}}", summary)
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
    build_signals: dict[str, Any],
    runtime_signals: dict[str, Any],
    scripts_signals: dict[str, Any],
    source_signals: dict[str, Any],
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
        tools=[think, list_selected_files, search_selected_files, fetch_file_context, build_finalize_tool()],
        prompt=prompt_path("PROMPT_L2_GAPCHECK_SYSTEM.md").read_text(encoding="utf-8").strip(),
    )

    def _format_signal(sig: dict) -> str:
        if not sig:
            return "  (no output)"
        files = sig.get("signal", [])
        notes = sig.get("notes", [])
        lines = [f"  files: {files}"]
        if notes:
            lines.append(f"  notes: {notes}")
        return "\n".join(lines)

    gap_prompt = (
        prompt_path("PROMPT_L2_GAPCHECK_TASK.md").read_text(encoding="utf-8")
        .replace("{{REPO_NAME}}", repo_name)
        .replace("{{BUILD_SIGNALS}}", _format_signal(build_signals))
        .replace("{{RUNTIME_SIGNALS}}", _format_signal(runtime_signals))
        .replace("{{SCRIPTS_SIGNALS}}", _format_signal(scripts_signals))
        .replace("{{SOURCE_SIGNALS}}", _format_signal(source_signals))
        .replace("{{SUMMARY_EVIDENCE}}", summary)
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

    def _as_list(field: str) -> list:
        if not isinstance(payload, dict):
            return []
        value = payload.get(field)
        return value if isinstance(value, list) else []

    fallback_manifest = any(
        any(marker in path for marker in ("package.json", "pyproject.toml", "requirements", "go.mod", "cargo.toml", "pom.xml", "build.gradle"))
        for path in selected_lower
    )
    fallback_docker = any("dockerfile" in path or "docker-compose" in path for path in selected_lower)
    fallback_scripts = any(
        any(marker in path for marker in ("makefile", "setup.sh", "install.sh", "build.sh", "/bin/", "/scripts/"))
        for path in selected_lower
    )
    fallback_source = any(
        any(marker in path for marker in (".env", "config.", "settings.", "src/", "lib/"))
        for path in selected_lower
    )

    return {
        "name": "gap-check",
        "mode": "llm_subagent",
        "signal": {
            "has_manifest": _as_bool("has_manifest", fallback_manifest),
            "has_docker": _as_bool("has_docker", fallback_docker),
            "has_scripts": _as_bool("has_scripts", fallback_scripts),
            "has_source_deps": _as_bool("has_source_deps", fallback_source),
            "conflicts": _as_list("conflicts"),
            "gaps": _as_list("gaps"),
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
    class SignalSubagentState(TypedDict, total=False):
        build: Optional[dict]
        runtime: Optional[dict]
        scripts: Optional[dict]
        source: Optional[dict]
        gap: Optional[dict]

    selected_lower = [path.lower() for path in selected_files]

    build_fallback = _deterministic_signal_scan(
        selected_files,
        ("package.json", "pyproject.toml", "requirements", "go.mod", "cargo.toml", "pom.xml", "build.gradle"),
    )
    runtime_fallback = _deterministic_signal_scan(
        selected_files,
        ("dockerfile", "docker-compose", "nix", "vagrantfile", "entrypoint"),
    )
    scripts_fallback = _deterministic_signal_scan(
        selected_files,
        ("makefile", "setup.sh", "install.sh", "build.sh", "/bin/", "scripts/"),
    )
    source_fallback = _deterministic_signal_scan(
        selected_files,
        (".env", ".env.example", "config.", "settings.", "src/", "lib/"),
    )

    async def build_node(_: SignalSubagentState) -> dict[str, Any]:
        return {
            "build": await _invoke_signal_subagent(
                name="build-signal-scan",
                repo_name=repo_name,
                summary=summary,
                prompt_body="Identify top manifest/build-declaration files (e.g. package.json, pyproject.toml, Cargo.toml, pom.xml, go.mod) relevant for build and dependency planning.",
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
                prompt_body="Identify containerization and runtime files (e.g. Dockerfile, docker-compose.yml, Nix flakes, Vagrantfile) that describe the full execution stack.",
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

    async def scripts_node(_: SignalSubagentState) -> dict[str, Any]:
        return {
            "scripts": await _invoke_signal_subagent(
                name="scripts-signal-scan",
                repo_name=repo_name,
                summary=summary,
                prompt_body="Identify build/install helper scripts (e.g. Makefile, setup.sh, install.sh, files in /bin or /scripts) that document manual installation steps.",
                deterministic_fallback=scripts_fallback,
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

    async def source_node(_: SignalSubagentState) -> dict[str, Any]:
        return {
            "source": await _invoke_signal_subagent(
                name="source-signal-scan",
                repo_name=repo_name,
                summary=summary,
                prompt_body="Identify source files that reveal implicit dependencies: files with import statements, .env / config files listing required ENV variables, or source files containing error strings that hint at missing system dependencies.",
                deterministic_fallback=source_fallback,
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

    async def gap_node(state: SignalSubagentState) -> dict[str, Any]:
        # Runs after all four scanners complete — can see their full outputs
        return {
            "gap": await _invoke_gap_subagent(
                repo_name=repo_name,
                summary=summary,
                selected_lower=selected_lower,
                build_signals=state.get("build") or {},
                runtime_signals=state.get("runtime") or {},
                scripts_signals=state.get("scripts") or {},
                source_signals=state.get("source") or {},
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
    subgraph.add_node("scripts", scripts_node)
    subgraph.add_node("source", source_node)
    subgraph.add_node("gap", gap_node)
    # Fan-out: 4 scanners run in parallel
    subgraph.add_edge(START, "build")
    subgraph.add_edge(START, "runtime")
    subgraph.add_edge(START, "scripts")
    subgraph.add_edge(START, "source")
    # Fan-in: gap runs after all four complete, receives merged state
    subgraph.add_edge("build", "gap")
    subgraph.add_edge("runtime", "gap")
    subgraph.add_edge("scripts", "gap")
    subgraph.add_edge("source", "gap")
    subgraph.add_edge("gap", END)
    compiled = subgraph.compile()

    results = await compiled.ainvoke(
        {},
        config={"configurable": {"thread_id": f"{repo_name}:l2-signals"}},
    )
    return [results["build"], results["runtime"], results["scripts"], results["source"], results["gap"]]


async def run_l2_synthesis_loop(
    *,
    repo_url: str,
    repo_name: str,
    repo_path: "Path",
    selected_files: list[str],
    summary: str,
    exploration_artifact: dict[str, Any],
    file_context_by_path: dict[str, str],
    classification_timeout: int,
    synthesis_react_max_steps: int,
    synthesis_subagents_enabled: bool,
    synthesis_review_rounds: int,
    snippet_tools_enabled: bool = False,
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
    from pathlib import Path as _Path
    _repo_path = _Path(repo_path).resolve()

    fetch_file_context = build_fetch_file_context_tool(file_context_by_path)
    list_selected_files = build_list_selected_files_tool(selected_snapshot)
    search_selected_files = build_search_selected_files_tool(selected_snapshot)
    think = build_think_tool()
    hadolint_check = build_hadolint_snippet_tool()
    read_gitlog = build_read_gitlog_tool(_repo_path)
    search_commits = build_search_commits_tool(_repo_path)
    get_dockerfile_snippet = build_get_dockerfile_snippet_tool() if snippet_tools_enabled else None

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

    _generator_tools = [think, list_selected_files, search_selected_files, fetch_file_context, read_gitlog, search_commits, hadolint_check, build_finalize_tool()]
    if get_dockerfile_snippet is not None:
        _generator_tools.append(get_dockerfile_snippet)
    _snippet_hint_generator = (
        "Use get_dockerfile_snippet to retrieve validated RUN-block snippets for common toolchains "
        "(call with action='list_actions' to see all options). "
        if get_dockerfile_snippet is not None else ""
    )
    synthesis_generator = create_react_agent(
        model=new_prebuilt_chat_model(classification_timeout),
        tools=_generator_tools,
        prompt=(
            prompt_path("PROMPT_L2_SYNTHESIS_GENERATOR_SYSTEM.md").read_text(encoding="utf-8")
            .replace("{{SNIPPET_TOOL_HINT}}", _snippet_hint_generator)
            .strip()
        ),
    )

    synthesis_prompt = (
        prompt_path("PROMPT_L2_SYNTHESIS_TASK.md").read_text(encoding="utf-8")
        .replace("{{REPO_URL}}", repo_url)
        .replace("{{CURRENT_HYPOTHESES}}", "\n".join(f"- {item}" for item in base_hypotheses[:20]))
        .replace("{{CURRENT_RISKS}}", "\n".join(f"- {item}" for item in risk_notes[:20]) if risk_notes else "- (none)")
        .replace("{{SUBAGENT_SIGNALS}}", str(subagent_outputs[:4]))
        .replace("{{SUMMARY_EVIDENCE}}", summary)
    )

    generation_result = await synthesis_generator.ainvoke(
        {"messages": [{"role": "user", "content": synthesis_prompt}]},
        config={"configurable": {"thread_id": f"{repo_name}:l2"}, "recursion_limit": max(20, int(synthesis_react_max_steps) * 6)},
    )
    generation_payload = extract_agent_payload(generation_result)
    updates = normalize_text_list(generation_payload.get("hypothesis_updates") if isinstance(generation_payload, dict) else [])
    risk_updates = normalize_text_list(generation_payload.get("risk_updates") if isinstance(generation_payload, dict) else [])
    done_flag = bool(generation_payload.get("done", False)) if isinstance(generation_payload, dict) else False

    generated_hypotheses = list(dict.fromkeys(item.strip() for item in (base_hypotheses + updates) if item and item.strip()))
    generated_risks = list(dict.fromkeys(item.strip() for item in (risk_notes + risk_updates) if item and item.strip()))
    generator_trace = extract_agent_trace(generation_result)
    if not generation_payload and hit_step_limit(generation_result):
        log_warn(
            f"[l2-synthesis {repo_name}] generator hit the recursion_limit without "
            f"finalizing; generation payload empty."
        )
        generator_stop_reason = "recursion_limit_hit"
    else:
        generator_stop_reason = "model_done" if done_flag else "agent_converged"

    _reviewer_tools = [think, list_selected_files, search_selected_files, fetch_file_context, read_gitlog, search_commits, hadolint_check, build_finalize_tool()]
    if get_dockerfile_snippet is not None:
        _reviewer_tools.append(get_dockerfile_snippet)
    _snippet_hint_reviewer = (
        "Use get_dockerfile_snippet to retrieve validated RUN-block snippets for toolchains you are reviewing. "
        if get_dockerfile_snippet is not None else ""
    )
    synthesis_reviewer = create_react_agent(
        model=new_prebuilt_chat_model(classification_timeout),
        tools=_reviewer_tools,
        prompt=(
            prompt_path("PROMPT_L2_SYNTHESIS_REVIEWER_SYSTEM.md").read_text(encoding="utf-8")
            .replace("{{SNIPPET_TOOL_HINT}}", _snippet_hint_reviewer)
            .strip()
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
            prompt_path("PROMPT_L2_REVIEW_TASK.md").read_text(encoding="utf-8")
            .replace("{{REPO_URL}}", repo_url)
            .replace("{{ROUND_INDEX}}", str(round_index + 1))
            .replace("{{ROUND_TOTAL}}", str(max(1, int(synthesis_review_rounds))))
            .replace("{{GENERATOR_HYPOTHESES}}", "\n".join(f"- {item}" for item in review_hypotheses[:30]))
            .replace("{{GENERATOR_RISKS}}", "\n".join(f"- {item}" for item in review_risks[:30]) if review_risks else "- (none)")
            .replace("{{SUBAGENT_SIGNALS}}", str(subagent_outputs[:4]))
            .replace("{{SUMMARY_EVIDENCE}}", summary)
        )

        review_result = await synthesis_reviewer.ainvoke(
            {"messages": [{"role": "user", "content": reviewer_prompt}]},
            config={
                "configurable": {"thread_id": f"{repo_name}:l2-review:r{round_index + 1}"},
                "recursion_limit": max(20, int(synthesis_react_max_steps) * 6),
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
    # Weighted signal hierarchy per diagram: CI/CD (0.20) > Dockerfile (0.18) > Manifest (0.15) > Tests (0.10)
    per_source_confidence = {
        "ci": 0.20 if has_ci else 0.0,
        "docker": 0.18 if has_docker else 0.0,
        "manifest": 0.15 if has_manifest else 0.0,
        "tests": 0.10 if has_tests else 0.0,
    }
    weighted_coverage = sum(per_source_confidence.values())
    gap_penalty = sum(1 for flag in exploration_artifact.get("evidence_gaps", {}).values() if flag)
    # Sub-agents with empty signal lists had no evidence to return.
    unknown_markers = [
        out["name"] for out in subagent_outputs
        if isinstance(out.get("signal"), list) and not out["signal"]
    ]
    stack_unknown = weighted_coverage == 0.0 and not dedup_hypotheses
    confidence = max(0.0, min(1.0, 0.35 + weighted_coverage - gap_penalty * 0.08))
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
        "stack_unknown": stack_unknown,
        "per_source_confidence": per_source_confidence,
        "unknown_markers": unknown_markers,
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
