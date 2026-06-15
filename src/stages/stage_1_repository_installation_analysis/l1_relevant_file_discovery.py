from typing import Any, Callable

from langchain.agents import create_agent
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.store.memory import InMemoryStore

try:
    from RepoBuilderAgent.src.agent_tools.react_loop_tools import (
        build_finalize_tool,
        build_list_tree_tool,
        build_read_file_tool,
        build_read_gitlog_tool,
        build_search_commits_tool,
        build_search_pattern_tool,
        build_search_structure_paths_tool,
        build_select_default_files_tool,
        build_think_tool,
        tool_call_budget,
    )
    from RepoBuilderAgent.src.core.common import prompt_path
    from RepoBuilderAgent.src.core.agent_runtime import ClassifyConfig, ClassifyRuntime, RepoRef
except ImportError:
    from agent_tools.react_loop_tools import (
        build_finalize_tool,
        build_list_tree_tool,
        build_read_file_tool,
        build_read_gitlog_tool,
        build_search_commits_tool,
        build_search_pattern_tool,
        build_search_structure_paths_tool,
        build_select_default_files_tool,
        build_think_tool,
        tool_call_budget,
    )
    from core.common import prompt_path
    from core.agent_runtime import ClassifyConfig, ClassifyRuntime, RepoRef

try:
    from RepoBuilderAgent.src.core.log_utils import log_warn
except ImportError:
    from core.log_utils import log_warn

try:
    from langgraph.errors import GraphRecursionError
except ImportError:  # langgraph version without a dedicated error class
    GraphRecursionError = RecursionError


def _hard_cap_selected_files(candidates: list[str], default_selected_files: list[str], cap: int) -> list[str]:
    """Enforce a strict final cap while keeping high-signal evidence first."""
    safe_cap = max(1, int(cap))
    high_signal_markers = (
        "package.json",
        "pyproject.toml",
        "requirements",
        "go.mod",
        "cargo.toml",
        "pom.xml",
        "build.gradle",
        "dockerfile",
        "docker-compose",
        ".github/workflows",
        "readme",
    )

    ordered_unique = list(dict.fromkeys(item.strip() for item in candidates if item and item.strip()))

    scored: list[tuple[int, str]] = []
    for idx, path in enumerate(ordered_unique):
        lower = path.lower()
        score = 0
        if any(marker in lower for marker in high_signal_markers):
            score += 100
        if lower.endswith((".yml", ".yaml", ".toml", ".json", ".md")):
            score += 10
        score += max(0, 20 - idx)
        scored.append((score, path))

    scored.sort(key=lambda item: (-item[0], item[1]))
    picked = [path for _, path in scored[:safe_cap]]

    if picked:
        return picked
    return default_selected_files[:safe_cap]


async def select_files_by_iterative_react(
    *,
    repo: RepoRef,
    structure_summary: str,
    default_selected_files: list[str],
    config: ClassifyConfig,
    runtime: ClassifyRuntime,
    estimate_tokens: Callable[[str, str], int],
) -> tuple[list[str], int, list[dict[str, Any]], str]:
    repo_url = repo.url
    repo_name = repo.name
    repo_path = repo.path
    selection_timeout = config.selection_timeout
    react_max_steps = config.react_max_steps
    react_max_total_files = config.react_max_total_files
    react_final_cap = config.react_final_cap
    model_name = runtime.model_name
    new_prebuilt_chat_model = runtime.new_prebuilt_chat_model
    extract_agent_payload = runtime.extract_agent_payload
    extract_agent_trace = runtime.extract_agent_trace
    normalize_text_list = runtime.normalize_text_list

    from pathlib import Path as _Path
    _repo_path = _Path(repo_path).resolve()

    max_total = max(1, int(react_max_total_files))
    # Keep the recursion_limit in one place so the prompt's tool-call budget and the
    # graph config can never drift apart.
    recursion_limit = max(30, int(react_max_steps) * 8)
    step1_prompt = (
        prompt_path("PROMPT_L1_TASK.md").read_text(encoding="utf-8")
        .replace("{{REPO_URL}}", repo_url)
        .replace("{{MAX_FILES}}", str(max_total))
        .replace("{{MAX_TOOL_CALLS}}", str(tool_call_budget(recursion_limit)))
        .replace("{{STRUCTURE_SUMMARY}}", structure_summary)
    )
    step1_tokens_total = estimate_tokens(step1_prompt, model_name)

    search_structure_paths = build_search_structure_paths_tool(structure_summary)
    select_default_files = build_select_default_files_tool(default_selected_files)
    think = build_think_tool()
    read_file = build_read_file_tool(_repo_path)
    list_tree = build_list_tree_tool(_repo_path)
    search_pattern = build_search_pattern_tool(_repo_path)
    read_gitlog = build_read_gitlog_tool(_repo_path)
    search_commits = build_search_commits_tool(_repo_path)

    # NOTE: L1 uses langchain.agents.create_agent (middleware API), not
    # create_react_agent (pre_model_hook API), so the history-trim hook does not
    # apply here. History trimming for L1 is intentionally deferred: its input is
    # already bounded (a token-capped structure summary plus a small tool-call
    # budget of small observations), so it does not overflow the input cap.
    retrieval_agent = create_agent(
        model=new_prebuilt_chat_model(selection_timeout),
        tools=[think, list_tree, search_pattern, read_file, read_gitlog, search_commits, search_structure_paths, select_default_files, build_finalize_tool()],
        system_prompt=prompt_path("PROMPT_L1_SYSTEM.md").read_text(encoding="utf-8").strip(),
        checkpointer=InMemorySaver(),
        store=InMemoryStore(),
        name="l1_retrieval_agent",
    )

    final_cap = min(max_total, max(1, int(react_final_cap)))
    try:
        result = await retrieval_agent.ainvoke(
            {"messages": [{"role": "user", "content": step1_prompt}]},
            config={"configurable": {"thread_id": f"{repo_name}:l1"}, "recursion_limit": recursion_limit},
        )
    except GraphRecursionError:
        # The ReAct loop never emitted a finalize action within the step budget
        # (common when the repo surfaces no obvious manifest). Don't crash the repo
        # — fall back to the heuristic default selection so classification can still
        # proceed. Previously this propagated and the repo produced nothing.
        log_warn(
            f"[l1 {repo_name}] iterative_react hit the recursion limit without "
            f"converging; falling back to default file selection."
        )
        selected_files = _hard_cap_selected_files(default_selected_files, default_selected_files, cap=final_cap)
        if not selected_files:
            selected_files = default_selected_files.copy()
        return selected_files, step1_tokens_total, [], "recursion_limit_fallback"
    payload = extract_agent_payload(result)
    raw_selected = payload.get("selected_files") if isinstance(payload, dict) else []
    prelim_selected = normalize_text_list(raw_selected)[:max_total]
    selected_files = _hard_cap_selected_files(
        prelim_selected,
        default_selected_files,
        cap=final_cap,
    )
    stop_reason = "model_done" if bool(payload.get("done", False)) else "agent_converged"
    react_trace = extract_agent_trace(result)

    if not selected_files:
        selected_files = default_selected_files.copy()
        stop_reason = "fallback_defaults"

    return selected_files, step1_tokens_total, react_trace, stop_reason
