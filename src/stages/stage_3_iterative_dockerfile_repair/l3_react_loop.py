import re
from typing import Any, Callable

from langgraph.prebuilt import create_react_agent

try:
    from RepoBuilderAgent.src.core.common import prompt_path
    from RepoBuilderAgent.src.core.agent_runtime import RepairRuntime
    from RepoBuilderAgent.src.core.llm_yaml import parse_llm_yaml_dict
    from RepoBuilderAgent.src.core.log_utils import log_warn
    from RepoBuilderAgent.src.agent_tools.react_loop_tools import (
        HISTORY_BUDGET,
        ainvoke_with_recursion_guard,
        build_finalize_tool,
        extract_finalize_answer,
        hit_step_limit,
        make_history_trim_hook,
        hooked_tool_call_budget,
    )
except ImportError:
    from core.common import prompt_path
    from core.agent_runtime import RepairRuntime
    from core.llm_yaml import parse_llm_yaml_dict
    from core.log_utils import log_warn
    from agent_tools.react_loop_tools import (
        HISTORY_BUDGET,
        ainvoke_with_recursion_guard,
        build_finalize_tool,
        extract_finalize_answer,
        hit_step_limit,
        make_history_trim_hook,
        hooked_tool_call_budget,
    )


def _extract_repair_trace(result: dict, *, max_content_chars: int = 2000) -> list[dict]:
    """Build a compact, serializable trace of the repair agent's ReAct steps.

    Captures each message's role, truncated content and any tool calls (name + args)
    so the viewer can show which tools the agent used — including the optional
    read-only repository tools. This is the L3 analogue of the L1 react-trace.
    """
    trace: list[dict] = []
    for idx, message in enumerate(result.get("messages") or [], start=1):
        content = getattr(message, "content", "")
        text = content if isinstance(content, str) else str(content)
        if len(text) > max_content_chars:
            text = text[:max_content_chars] + "\n... [truncated]"
        tool_calls = []
        for tc in (getattr(message, "tool_calls", None) or []):
            if isinstance(tc, dict):
                tool_calls.append({"name": tc.get("name"), "args": tc.get("args")})
        trace.append(
            {
                "step": idx,
                "role": getattr(message, "type", "unknown"),
                "content": text,
                "tool_calls": tool_calls,
            }
        )
    return trace


def _extract_react_payload(result: dict) -> dict:
    # (1) structured_response if dict.
    structured = result.get("structured_response")
    if isinstance(structured, dict):
        return structured

    messages = result.get("messages") or []

    # (2) prefer the finalize tool-call's answer when present.
    finalize_answer = extract_finalize_answer(messages)
    if finalize_answer is not None:
        parsed = parse_llm_yaml_dict(finalize_answer)
        if isinstance(parsed, dict) and parsed:
            return parsed

    # (3) trailing-message YAML fallback.
    if messages:
        content = getattr(messages[-1], "content", "")
        parsed = parse_llm_yaml_dict(content)
        if isinstance(parsed, dict) and parsed:
            return parsed

    # (4) {} — surface the silent recursion-limit failure.
    if hit_step_limit(result):
        log_warn(
            "[l3] ReAct repair/verify agent yielded empty payload after hitting the "
            "LangGraph recursion_limit (placeholder 'Sorry, need more steps...'); returning {}."
        )
    return {}


def _extract_react_command(result: dict, candidate_keys: list[str]) -> str:
    payload = _extract_react_payload(result)
    text = ""
    if isinstance(payload, dict):
        for key in candidate_keys:
            value = payload.get(key)
            if isinstance(value, str) and value.strip():
                text = value.strip()
                break

    if not text:
        messages = result.get("messages") or []
        if messages:
            last_content = getattr(messages[-1], "content", "")
            if isinstance(last_content, str) and last_content.strip():
                text = last_content.strip()

    if not text:
        return ""

    match = re.search(r"```(?:bash|sh)?\n(.*?)```", text, re.DOTALL | re.IGNORECASE)
    if match:
        text = match.group(1).strip()

    return text.strip().strip("`")


async def run_l3_dockerfile_repair_react(
    *,
    repo_url: str,
    attempt_number: int,
    prompt: str,
    repair_timeout: int,
    l3_react_max_steps: int,
    runtime: RepairRuntime,
    build_snippet_tool: "Callable[[], Any] | None" = None,
    repo_tools: "list[Any] | None" = None,
) -> tuple[str, int, list[dict]]:
    model_name = runtime.model_name
    new_prebuilt_chat_model = runtime.new_prebuilt_chat_model
    build_think_tool = runtime.build_think_tool
    build_hadolint_snippet_tool = runtime.build_hadolint_snippet_tool
    extract_dockerfile = runtime.extract_dockerfile

    think = build_think_tool()
    hadolint_tool = build_hadolint_snippet_tool()
    tools: list[Any] = [think, hadolint_tool, build_finalize_tool()]
    if build_snippet_tool is not None:
        tools.append(build_snippet_tool())
    # Optional read-only repository tools (read_file / list_tree / search_pattern) let the
    # repair agent inspect the actual source it is fixing. They are evidence-gathering only;
    # building/verifying/rollback stay in the deterministic outer loop.
    if repo_tools:
        tools.extend(repo_tools)
    _snippet_hint = (
        "Use get_dockerfile_snippet to retrieve validated RUN-block snippets for common toolchains "
        "(call with action='list_actions' to see all available snippets). "
        if build_snippet_tool is not None else ""
    )
    recursion_limit = max(8, int(l3_react_max_steps) * 4)
    repair_agent = create_react_agent(
        model=new_prebuilt_chat_model(repair_timeout),
        tools=tools,
        prompt=(
            prompt_path("PROMPT_L3_REPAIR_SYSTEM.md").read_text(encoding="utf-8")
            .replace("{{SNIPPET_TOOL_HINT}}", _snippet_hint)
            .replace("{{MAX_TOOL_CALLS}}", str(hooked_tool_call_budget(recursion_limit)))
            .strip()
        ),
        pre_model_hook=make_history_trim_hook(model_name, HISTORY_BUDGET),
    )

    result = await ainvoke_with_recursion_guard(
        repair_agent,
        {"messages": [{"role": "user", "content": prompt}]},
        {
            "configurable": {"thread_id": f"{repo_url}:l3-repair:{attempt_number}"},
            "recursion_limit": recursion_limit,
        },
    )

    payload = _extract_react_payload(result)
    repaired = ""
    if isinstance(payload, dict):
        candidate = payload.get("repaired_dockerfile")
        if isinstance(candidate, str) and candidate.strip():
            repaired = extract_dockerfile(candidate.strip())

    if not repaired:
        messages = result.get("messages") or []
        if messages:
            last_content = getattr(messages[-1], "content", "")
            if isinstance(last_content, str) and last_content.strip():
                repaired = extract_dockerfile(last_content.strip())

    trace = _extract_repair_trace(result)
    if not repaired and hit_step_limit(result):
        log_warn(
            f"[l3 repair attempt {attempt_number}] hit the LangGraph recursion_limit "
            f"without producing a repaired Dockerfile; tagging trace recursion_limit_hit."
        )
        trace.append({"step": len(trace) + 1, "role": "system", "content": "", "tool_calls": [], "stop_reason": "recursion_limit_hit"})
    return repaired, len(result.get("messages") or []), trace


async def run_l3_verification_command_react(
    *,
    repo_url: str,
    prompt: str,
    verify_timeout: int,
    l3_react_max_steps: int,
    thread_suffix: str,
    system_prompt: str,
    candidate_keys: list[str],
    runtime: RepairRuntime,
) -> tuple[str, int]:
    model_name = runtime.model_name
    new_prebuilt_chat_model = runtime.new_prebuilt_chat_model
    build_think_tool = runtime.build_think_tool

    think = build_think_tool()
    recursion_limit = max(8, int(l3_react_max_steps) * 4)
    verify_agent = create_react_agent(
        model=new_prebuilt_chat_model(verify_timeout),
        tools=[think, build_finalize_tool()],
        prompt=system_prompt.replace("{{MAX_TOOL_CALLS}}", str(hooked_tool_call_budget(recursion_limit))),
        pre_model_hook=make_history_trim_hook(model_name, HISTORY_BUDGET),
    )

    result = await ainvoke_with_recursion_guard(
        verify_agent,
        {"messages": [{"role": "user", "content": prompt}]},
        {
            "configurable": {"thread_id": f"{repo_url}:{thread_suffix}"},
            "recursion_limit": recursion_limit,
        },
    )

    command = _extract_react_command(result, candidate_keys)
    return command, len(result.get("messages") or [])
