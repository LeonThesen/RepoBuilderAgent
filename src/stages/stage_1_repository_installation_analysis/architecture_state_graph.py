from typing import Any, Callable, cast
from typing_extensions import NotRequired, TypedDict

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.store.memory import InMemoryStore
from langgraph.graph import END, START, StateGraph

try:
    from RepoBuilderAgent.src.stages.stage_1_repository_installation_analysis.l2_install_command_extraction import run_l2_synthesis_loop
    from RepoBuilderAgent.src.stages.stage_1_repository_installation_analysis.classify_validation_loop import run_classify_validation_loop
except ImportError:
    from stages.stage_1_repository_installation_analysis.l2_install_command_extraction import run_l2_synthesis_loop
    from stages.stage_1_repository_installation_analysis.classify_validation_loop import run_classify_validation_loop


class ArchitectureLoopState(TypedDict):
    repo_url: str
    repo_name: str
    repo_path: str
    summary: str
    selected_files: list[str]
    file_context_by_path: dict[str, str]
    exploration_artifact: dict[str, Any]
    synthesis_artifact: NotRequired[dict[str, Any]]
    synthesis_loop_trace: NotRequired[list[dict[str, Any]]]
    synthesis_stop_reason: NotRequired[str]
    validation_artifact: NotRequired[dict[str, Any]]
    validation_loop_trace: NotRequired[list[dict[str, Any]]]
    validation_stop_reason: NotRequired[str]
    l2_to_classify_validation_route_reason: NotRequired[str]
    run_validation: bool
    l3_retry_count: NotRequired[int]


async def run_architecture_state_graph(
    *,
    repo_url: str,
    repo_name: str,
    repo_path: str,
    summary: str,
    selected_files: list[str],
    exploration_artifact: dict[str, Any],
    file_context_by_path: dict[str, str],
    run_validation: bool,
    classification_timeout: int,
    synthesis_react_max_steps: int,
    synthesis_review_rounds: int,
    validation_react_max_steps: int,
    synthesis_subagents_enabled: bool,
    snippet_tools_enabled: bool = False,
    model_name: str,
    new_prebuilt_chat_model: Callable[[int], Any],
    extract_agent_payload: Callable[[dict[str, Any]], Any],
    extract_agent_trace: Callable[[dict[str, Any]], list[dict[str, Any]]],
    normalize_text_list: Callable[[Any], list[str]],
    normalize_validation_checks: Callable[[Any], dict[str, dict[str, str]]],
) -> tuple[dict[str, Any], list[dict[str, Any]], str, dict[str, Any], list[dict[str, Any]], str]:
    graph = StateGraph(ArchitectureLoopState)

    async def l2_node(state: ArchitectureLoopState) -> dict[str, Any]:
        synthesis_artifact, synthesis_loop_trace, synthesis_stop_reason = await run_l2_synthesis_loop(
            repo_url=state["repo_url"],
            repo_name=state["repo_name"],
            repo_path=state["repo_path"],
            selected_files=state["selected_files"],
            summary=state["summary"],
            exploration_artifact=state["exploration_artifact"],
            file_context_by_path=state["file_context_by_path"],
            classification_timeout=classification_timeout,
            synthesis_react_max_steps=synthesis_react_max_steps,
            synthesis_review_rounds=synthesis_review_rounds,
            synthesis_subagents_enabled=synthesis_subagents_enabled,
            snippet_tools_enabled=snippet_tools_enabled,
            model_name=model_name,
            new_prebuilt_chat_model=new_prebuilt_chat_model,
            extract_agent_payload=extract_agent_payload,
            extract_agent_trace=extract_agent_trace,
            normalize_text_list=normalize_text_list,
        )
        transition_policy = cast(dict[str, Any], synthesis_artifact.get("transition_policy", {}))
        route_reason = "policy_default"
        run_validation_flag = transition_policy.get("run_classify_validation")
        if run_validation_flag is False:
            route_reason = "l2_confident_terminal"
        elif run_validation_flag is True:
            route_reason = "l2_requires_validation"
        return {
            "synthesis_artifact": synthesis_artifact,
            "synthesis_loop_trace": synthesis_loop_trace,
            "synthesis_stop_reason": synthesis_stop_reason,
            "l2_to_classify_validation_route_reason": route_reason,
        }

    async def classify_validation_node(state: ArchitectureLoopState) -> dict[str, Any]:
        synthesis_artifact = cast(dict[str, Any], state.get("synthesis_artifact", {}))
        validation_artifact, validation_loop_trace, validation_stop_reason = await run_classify_validation_loop(
            repo_url=state["repo_url"],
            summary=state["summary"],
            synthesis_artifact=synthesis_artifact,
            selected_files=state["selected_files"],
            file_context_by_path=state["file_context_by_path"],
            classification_timeout=classification_timeout,
            validation_react_max_steps=validation_react_max_steps,
            model_name=model_name,
            new_prebuilt_chat_model=new_prebuilt_chat_model,
            extract_agent_payload=extract_agent_payload,
            extract_agent_trace=extract_agent_trace,
            normalize_text_list=normalize_text_list,
            normalize_validation_checks=normalize_validation_checks,
        )
        return {
            "validation_artifact": validation_artifact,
            "validation_loop_trace": validation_loop_trace,
            "validation_stop_reason": validation_stop_reason,
            "l3_retry_count": int(state.get("l3_retry_count", 0)) + 1,
        }

    def route_after_l2(state: ArchitectureLoopState) -> str:
        if not bool(state.get("run_validation", True)):
            return END
        synthesis_artifact = cast(dict[str, Any], state.get("synthesis_artifact", {}))
        transition_policy = cast(dict[str, Any], synthesis_artifact.get("transition_policy", {}))
        run_validation_flag = transition_policy.get("run_classify_validation")
        if run_validation_flag is False:
            return END
        return "classify_validation"

    def route_after_l3(state: ArchitectureLoopState) -> str:
        """Escalate back to L2 once when L3 detects zero build evidence (runtime gap)."""
        validation_artifact = cast(dict[str, Any], state.get("validation_artifact", {}))
        retry_count = int(state.get("l3_retry_count", 0))
        if validation_artifact.get("runtime_gap_detected") and retry_count <= 1:
            return "l2_synthesis"
        return END

    graph.add_node("l2_synthesis", l2_node)
    graph.add_node("classify_validation", classify_validation_node)
    graph.add_edge(START, "l2_synthesis")
    graph.add_conditional_edges("l2_synthesis", route_after_l2, {"classify_validation": "classify_validation", END: END})
    graph.add_conditional_edges("classify_validation", route_after_l3, {"l2_synthesis": "l2_synthesis", END: END})

    compiled = graph.compile(checkpointer=InMemorySaver(), store=InMemoryStore())
    result = await compiled.ainvoke(
        {
            "repo_url": repo_url,
            "repo_name": repo_name,
            "repo_path": repo_path,
            "summary": summary,
            "selected_files": selected_files,
            "exploration_artifact": exploration_artifact,
            "file_context_by_path": file_context_by_path,
            "run_validation": run_validation,
        },
        config={"configurable": {"thread_id": f"{repo_name}:architecture-loop"}},
    )

    synthesis_artifact = result["synthesis_artifact"]
    synthesis_loop_trace = result["synthesis_loop_trace"]
    synthesis_stop_reason = result["synthesis_stop_reason"]
    validation_artifact = result.get(
        "validation_artifact",
        {
            "repo": repo_url,
            "stage": "validation",
            "react": {"steps": 0, "max_steps": max(1, int(validation_react_max_steps)), "stop_reason": "disabled"},
            "checks": {},
            "warnings": [],
            "loop_trace": [],
        },
    )
    validation_loop_trace = result.get("validation_loop_trace", [])
    validation_stop_reason = result.get("validation_stop_reason", "disabled")
    return (
        synthesis_artifact,
        synthesis_loop_trace,
        synthesis_stop_reason,
        validation_artifact,
        validation_loop_trace,
        validation_stop_reason,
    )
