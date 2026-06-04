from typing import Any, Callable

from langchain.agents import create_agent
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.store.memory import InMemoryStore

try:
    from RepoBuilderAgent.src.loops.tools import build_search_structure_paths_tool, build_select_default_files_tool, build_think_tool
except ImportError:
    from loops.tools import build_search_structure_paths_tool, build_select_default_files_tool, build_think_tool


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
    max_total = max(1, int(react_max_total_files))
    step1_prompt = (
        "Repository: " + repo_url + "\n"
        "Use tools to inspect likely high-signal files and return YAML keys: thought, selected_files, done.\n"
        "Before each decisive tool call, use the think tool with a brief intent note.\n"
        "selected_files must contain only repository-relative paths.\n"
        "Keep at most " + str(max_total) + " files and set done=true when sufficient evidence exists for install/build/verify coverage.\n\n"
        "STRUCTURE_SUMMARY:\n" + structure_summary
    )
    step1_tokens_total = estimate_tokens(step1_prompt, model_name)

    search_structure_paths = build_search_structure_paths_tool(structure_summary)
    select_default_files = build_select_default_files_tool(default_selected_files)
    think = build_think_tool()

    retrieval_agent = create_agent(
        model=new_prebuilt_chat_model(selection_timeout),
        tools=[think, search_structure_paths, select_default_files],
        system_prompt=(
            "You are the repository exploration agent. Use tools before answering. "
            "Use think for concise planning notes between tool decisions. "
            "Return only YAML-compatible fields in your final answer."
        ),
        checkpointer=InMemorySaver(),
        store=InMemoryStore(),
        name="l1_retrieval_agent",
    )

    result = await retrieval_agent.ainvoke(
        {"messages": [{"role": "user", "content": step1_prompt}]},
        config={"configurable": {"thread_id": f"{repo_name}:l1"}, "recursion_limit": max(8, int(react_max_steps) * 4)},
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
