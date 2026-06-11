import re
from typing import Any, Callable

import yaml
from langgraph.prebuilt import create_react_agent

try:
    from RepoBuilderAgent.src.core.common import prompt_path
except ImportError:
    from core.common import prompt_path


def _extract_react_payload(result: dict) -> dict:
    messages = result.get("messages") or []
    if not messages:
        return {}
    last = messages[-1]
    content = getattr(last, "content", "")
    if not isinstance(content, str) or not content.strip():
        return {}
    match = re.search(r"```(?:yaml)?\\n(.*?)```", content, re.DOTALL | re.IGNORECASE)
    yaml_text = match.group(1) if match else content
    try:
        parsed = yaml.safe_load(yaml_text)
    except Exception:
        return {}
    return parsed if isinstance(parsed, dict) else {}


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
    new_prebuilt_chat_model: Callable[[int], Any],
    build_think_tool: Callable[[], Any],
    build_hadolint_snippet_tool: Callable[[], Any],
    extract_dockerfile: Callable[[str], str],
    build_snippet_tool: "Callable[[], Any] | None" = None,
) -> tuple[str, int]:
    think = build_think_tool()
    hadolint_tool = build_hadolint_snippet_tool()
    tools: list[Any] = [think, hadolint_tool]
    if build_snippet_tool is not None:
        tools.append(build_snippet_tool())
    _snippet_hint = (
        "Use get_dockerfile_snippet to retrieve validated RUN-block snippets for common toolchains "
        "(call with action='list_actions' to see all available snippets). "
        if build_snippet_tool is not None else ""
    )
    repair_agent = create_react_agent(
        model=new_prebuilt_chat_model(repair_timeout),
        tools=tools,
        prompt=(
            prompt_path("PROMPT_L3_REPAIR_SYSTEM.md").read_text(encoding="utf-8")
            .replace("{{SNIPPET_TOOL_HINT}}", _snippet_hint)
            .strip()
        ),
    )

    result = await repair_agent.ainvoke(
        {"messages": [{"role": "user", "content": prompt}]},
        config={
            "configurable": {"thread_id": f"{repo_url}:l3-repair:{attempt_number}"},
            "recursion_limit": max(8, int(l3_react_max_steps) * 4),
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

    return repaired, len(result.get("messages") or [])


async def run_l3_verification_command_react(
    *,
    repo_url: str,
    prompt: str,
    verify_timeout: int,
    l3_react_max_steps: int,
    thread_suffix: str,
    system_prompt: str,
    candidate_keys: list[str],
    new_prebuilt_chat_model: Callable[[int], Any],
    build_think_tool: Callable[[], Any],
) -> tuple[str, int]:
    think = build_think_tool()
    verify_agent = create_react_agent(
        model=new_prebuilt_chat_model(verify_timeout),
        tools=[think],
        prompt=system_prompt,
    )

    result = await verify_agent.ainvoke(
        {"messages": [{"role": "user", "content": prompt}]},
        config={
            "configurable": {"thread_id": f"{repo_url}:{thread_suffix}"},
            "recursion_limit": max(8, int(l3_react_max_steps) * 4),
        },
    )

    command = _extract_react_command(result, candidate_keys)
    return command, len(result.get("messages") or [])
