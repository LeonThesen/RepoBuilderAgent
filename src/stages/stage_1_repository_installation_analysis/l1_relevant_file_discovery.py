from typing import Any, Callable

from langchain.agents import create_agent
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.store.memory import InMemoryStore

try:
    from RepoBuilderAgent.src.agent_tools.react_loop_tools import (
        build_list_tree_tool,
        build_read_file_tool,
        build_read_gitlog_tool,
        build_search_commits_tool,
        build_search_pattern_tool,
        build_search_structure_paths_tool,
        build_select_default_files_tool,
        build_think_tool,
    )
    from RepoBuilderAgent.src.core.common import prompt_path
except ImportError:
    from agent_tools.react_loop_tools import (
        build_list_tree_tool,
        build_read_file_tool,
        build_read_gitlog_tool,
        build_search_commits_tool,
        build_search_pattern_tool,
        build_search_structure_paths_tool,
        build_select_default_files_tool,
        build_think_tool,
    )
    from core.common import prompt_path


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
    repo_url: str,
    repo_name: str,
    repo_path: "Path",
    structure_summary: str,
    default_selected_files: list[str],
    model_name: str,
    selection_timeout: int,
    react_max_steps: int,
    react_max_total_files: int,
    react_final_cap: int,
    new_prebuilt_chat_model: Callable[[int], Any],
    extract_agent_payload: Callable[[dict[str, Any]], Any],
    extract_agent_trace: Callable[[dict[str, Any]], list[dict[str, Any]]],
    normalize_text_list: Callable[[Any], list[str]],
    estimate_tokens: Callable[[str, str], int],
) -> tuple[list[str], int, list[dict[str, Any]], str]:
    from pathlib import Path as _Path
    _repo_path = _Path(repo_path).resolve()

    max_total = max(1, int(react_max_total_files))
    step1_prompt = (
        prompt_path("PROMPT_L1_TASK.md").read_text(encoding="utf-8")
        .replace("{{REPO_URL}}", repo_url)
        .replace("{{MAX_FILES}}", str(max_total))
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

    retrieval_agent = create_agent(
        model=new_prebuilt_chat_model(selection_timeout),
        tools=[think, list_tree, search_pattern, read_file, read_gitlog, search_commits, search_structure_paths, select_default_files],
        system_prompt=prompt_path("PROMPT_L1_SYSTEM.md").read_text(encoding="utf-8").strip(),
        checkpointer=InMemorySaver(),
        store=InMemoryStore(),
        name="l1_retrieval_agent",
    )

    result = await retrieval_agent.ainvoke(
        {"messages": [{"role": "user", "content": step1_prompt}]},
        config={"configurable": {"thread_id": f"{repo_name}:l1"}, "recursion_limit": max(12, int(react_max_steps) * 6)},
    )
    payload = extract_agent_payload(result)
    raw_selected = payload.get("selected_files") if isinstance(payload, dict) else []
    prelim_selected = normalize_text_list(raw_selected)[:max_total]
    selected_files = _hard_cap_selected_files(
        prelim_selected,
        default_selected_files,
        cap=min(max_total, max(1, int(react_final_cap))),
    )
    stop_reason = "model_done" if bool(payload.get("done", False)) else "agent_converged"
    react_trace = extract_agent_trace(result)

    if not selected_files:
        selected_files = default_selected_files.copy()
        stop_reason = "fallback_defaults"

    return selected_files, step1_tokens_total, react_trace, stop_reason
