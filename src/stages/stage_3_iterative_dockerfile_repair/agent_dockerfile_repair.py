import argparse
import asyncio
import hashlib
import json
import os
import re
import shlex
import ssl
from pathlib import Path

import httpx
import yaml
from openai import APIError, APITimeoutError, AsyncOpenAI
from tqdm import tqdm

try:
    from RepoBuilderAgent.src.core.config import OPENAI_API_KEY, OPENAI_BASE_URL, OPENAI_MODEL
    from RepoBuilderAgent.src.core.log_utils import log_error, log_info, log_trace, log_warn, set_tqdm_bar, set_trace_enabled
    from RepoBuilderAgent.src.agent_tools.react_loop_tools import build_hadolint_snippet_tool, build_think_tool
    from RepoBuilderAgent.src.core.chat_model_factory import make_prebuilt_chat_model_factory
    from RepoBuilderAgent.src.core.dockerfile_utils import extract_dockerfile, get_base_template
    from RepoBuilderAgent.src.core.file_io import write_text
    from RepoBuilderAgent.src.core.repo_cleanup import delete_docs_build_context
    from RepoBuilderAgent.src.stages.stage_3_iterative_dockerfile_repair.l3_react_loop import (
        run_l3_dockerfile_repair_react,
        run_l3_verification_command_react,
    )
    from RepoBuilderAgent.src.core.timeout_config import load_timeout_defaults
    from RepoBuilderAgent.src.core.prompt_profiles import (
        apply_prompt_profile,
        prompt_profile_metadata,
        resolve_prompt_profile,
        resolve_prompt_temperature,
    )
    from RepoBuilderAgent.src.core.common import (
        ensure_repo_checkout,
        finalize_llm_metrics,
        init_llm_metrics,
        inject_ca_cert_into_dockerfile,
        load_architecture_scratchpad,
        load_shared_repository_state,
        load_repo_urls,
        load_summary,
        prompt_path,
        read_yaml_file,
        render_architecture_scratchpad_for_prompt,
        render_shared_repository_state_for_prompt,
        render_validation_findings_for_prompt,
        render_yaml,
        repo_name_from_url,
        should_use_progress,
        upsert_shared_repository_state,
        update_progress,
        validate_dockerfile_syntax,
    )
except ImportError:
    # Fallback for direct script execution from RepoBuilderAgent/src
    import core.config as _config
    from core.log_utils import log_error, log_info, log_trace, log_warn, set_tqdm_bar, set_trace_enabled
    from agent_tools.react_loop_tools import build_hadolint_snippet_tool, build_think_tool
    from core.chat_model_factory import make_prebuilt_chat_model_factory
    from core.dockerfile_utils import extract_dockerfile, get_base_template
    from core.file_io import write_text
    from core.repo_cleanup import delete_docs_build_context
    from stages.stage_3_iterative_dockerfile_repair.l3_react_loop import (
        run_l3_dockerfile_repair_react,
        run_l3_verification_command_react,
    )
    from core.timeout_config import load_timeout_defaults
    from core.prompt_profiles import (
        apply_prompt_profile,
        prompt_profile_metadata,
        resolve_prompt_profile,
        resolve_prompt_temperature,
    )
    from core.common import (
        ensure_repo_checkout,
        finalize_llm_metrics,
        init_llm_metrics,
        inject_ca_cert_into_dockerfile,
        load_architecture_scratchpad,
        load_shared_repository_state,
        load_repo_urls,
        load_summary,
        prompt_path,
        read_yaml_file,
        render_architecture_scratchpad_for_prompt,
        render_shared_repository_state_for_prompt,
        render_validation_findings_for_prompt,
        render_yaml,
        repo_name_from_url,
        should_use_progress,
        upsert_shared_repository_state,
        update_progress,
        validate_dockerfile_syntax,
    )

    OPENAI_API_KEY = getattr(_config, "OPENAI_API_KEY", "")
    OPENAI_BASE_URL = getattr(_config, "OPENAI_BASE_URL", "https://api.openai.com/v1")
    OPENAI_MODEL = getattr(_config, "OPENAI_MODEL", "gpt-4o")

TIMEOUTS = load_timeout_defaults(
    "agent_dockerfile_repair",
    {
        "timeout": 120,
        "llm_max_retries": 2,
        "llm_retry_backoff_seconds": 2.0,
        "repair_timeout": 240,
        "verify_repair_timeout": 180,
        "verify_timeout": 30,
    },
)


parser = argparse.ArgumentParser(
    description="Build generated Dockerfiles, diagnose failures, and iteratively repair them up to N attempts."
)
parser.add_argument("--input-file", default="repos.json", help="Path to input file containing repository URLs")
parser.add_argument(
    "--repo-url",
    action="append",
    default=[],
    help="Repair the Dockerfile for a specific repository URL (can be passed multiple times). Overrides --input-file when provided.",
)
parser.add_argument("--endpoint", default=os.getenv("LLM_ENDPOINT", OPENAI_BASE_URL), help="Custom API endpoint URL")
parser.add_argument("--model", default=os.getenv("LLM_MODEL", OPENAI_MODEL), help="Model name")
parser.add_argument("--api-key", default=os.getenv("LLM_API_KEY", OPENAI_API_KEY), help="API key")
parser.add_argument("--prompt-profile", default=os.getenv("PROMPT_PROFILE", "P*"), help="Prompt profile name from RepoBuilderAgent/config/prompt_profiles.yaml (supports alias P*)")
parser.add_argument("--temperature", type=float, default=None, help="Temperature override for the model; defaults to selected prompt profile value")
parser.add_argument("--timeout", type=int, default=int(TIMEOUTS["timeout"]), help="Timeout for API requests in seconds")
parser.add_argument("--llm-max-retries", type=int, default=int(TIMEOUTS["llm_max_retries"]), help="Maximum retries for transient LLM timeouts and retryable API errors")
parser.add_argument("--llm-retry-backoff-seconds", type=float, default=float(TIMEOUTS["llm_retry_backoff_seconds"]), help="Base exponential backoff delay in seconds for LLM retries")
parser.add_argument("--repair-timeout", type=int, default=int(TIMEOUTS["repair_timeout"]), help="Timeout for Dockerfile repair LLM calls in seconds")
parser.add_argument("--verify-repair-timeout", type=int, default=int(TIMEOUTS["verify_repair_timeout"]), help="Timeout for verification-command repair LLM calls in seconds")
parser.add_argument("--trace", action="store_true", help="Enable verbose trace logs")
parser.add_argument("--results-dir", default="classification_results", help="Directory containing classification result YAML files")
parser.add_argument("--summaries-dir", default="summaries", help="Directory containing repository summary files")
parser.add_argument("--repos-dir", default="repos", help="Directory containing cloned repositories")
parser.add_argument("--dockerfiles-dir", default="dockerfiles", help="Directory containing generated Dockerfiles")
parser.add_argument("--reports-dir", default="repair-reports", help="Directory where repair attempt logs and reports will be written")
parser.add_argument("--container-cli", default="docker", help="Container CLI to use for builds")
parser.add_argument("--max-attempts", type=int, default=3, help="Maximum number of build and repair attempts per repository")
parser.add_argument("--max-log-chars", type=int, default=24000, help="Maximum number of build log characters to send to the model")
parser.add_argument("--skip-delete-docs", action="store_true", help="Skip deleting documentation and CI/CD files from the build context before building")
parser.add_argument("--skip-hadolint", action="store_true", help="Skip Dockerfile syntax validation via hadolint before docker build")
parser.add_argument("--verify-command", default="echo build-ok", help="Shell command executed inside the built image to verify the build produced working software")
parser.add_argument("--verify-timeout", type=int, default=int(TIMEOUTS["verify_timeout"]), help="Timeout in seconds for build verification container execution")
stateful_group = parser.add_mutually_exclusive_group()
stateful_group.add_argument(
    "--stateful-repair",
    dest="stateful_repair",
    action="store_true",
    help="Enable stateful repair prompts that include compact summaries of previous repair attempts.",
)
stateful_group.add_argument(
    "--no-stateful-repair",
    dest="stateful_repair",
    action="store_false",
    help="Disable stateful repair prompts and use only the current-attempt evidence.",
)
parser.set_defaults(stateful_repair=False)
parser.add_argument(
    "--stateful-history-window",
    type=int,
    default=4,
    help="When stateful repair is enabled, include at most this many recent repair attempts in the prompt history.",
)
parser.add_argument(
    "--stateful-history-max-chars",
    type=int,
    default=4000,
    help="Maximum characters from serialized repair history to include in each stateful repair prompt.",
)
stateful_tree_group = parser.add_mutually_exclusive_group()
stateful_tree_group.add_argument(
    "--stateful-repair-tree",
    dest="stateful_repair_tree",
    action="store_true",
    help="When stateful repair is enabled, also include a compact decision-tree summary of prior attempts.",
)
stateful_tree_group.add_argument(
    "--no-stateful-repair-tree",
    dest="stateful_repair_tree",
    action="store_false",
    help="Disable decision-tree serialization for stateful repair prompts.",
)
parser.set_defaults(stateful_repair_tree=False)
parser.add_argument(
    "--stateful-tree-max-chars",
    type=int,
    default=2500,
    help="Maximum characters from serialized stateful decision tree to include in each repair prompt.",
)
parser.add_argument(
    "--stateful-tree-max-children",
    type=int,
    default=5,
    help="Maximum child nodes retained per decision-tree node before pruning low-frequency branches.",
)
parser.add_argument(
    "--l3-react-max-steps",
    type=int,
    default=6,
    help="Maximum built-in ReAct steps for Loop 3 Dockerfile repair decisions.",
)
parser.add_argument("--force", action="store_true", help="Re-run repair even if a successful report.yaml already exists")
args = parser.parse_args()
PROMPT_PROFILE = resolve_prompt_profile(args.prompt_profile)
EFFECTIVE_TEMPERATURE = resolve_prompt_temperature(args.temperature, PROMPT_PROFILE)


# httpx defaults to certifi's CA bundle, which does not include corporate / internal CAs.
# Use ssl.create_default_context() to pull in the OS trust store instead.
_ssl_context = ssl.create_default_context()
_http_client = httpx.AsyncClient(verify=_ssl_context)

client = AsyncOpenAI(
    base_url=args.endpoint,
    api_key=args.api_key,
    timeout=args.timeout,
    http_client=_http_client,
)

with open(prompt_path("PROMPT_DOCKERFILE_REPAIR.md"), "r", encoding="utf-8") as prompt_file:
    PROMPT_TEMPLATE = apply_prompt_profile(prompt_file.read(), PROMPT_PROFILE, "repair")

with open(prompt_path("PROMPT_BUILD_VERIFICATION.md"), "r", encoding="utf-8") as prompt_file:
    VERIFY_PROMPT_TEMPLATE = prompt_file.read()

sem = asyncio.Semaphore(1)

set_trace_enabled(args.trace)


_new_prebuilt_chat_model = make_prebuilt_chat_model_factory(
    model=args.model,
    temperature=EFFECTIVE_TEMPERATURE,
    api_key=args.api_key,
    base_url=args.endpoint,
    max_retries=args.llm_max_retries,
    http_async_client=_http_client,
)


def sanitize_image_tag(value: str) -> str:
    sanitized = re.sub(r"[^a-z0-9._-]+", "-", value.lower()).strip("-._")
    return sanitized or "image"


def combine_build_output(command: list[str], returncode: int, output: str) -> str:
    rendered_command = " ".join(command)
    return (
        f"$ {rendered_command}\n"
        f"exit_code: {returncode}\n\n"
        f"OUTPUT:\n{output}"
    )


def trim_log(log: str) -> str:
    if len(log) <= args.max_log_chars:
        return log

    half = args.max_log_chars // 2
    return log[:half] + "\n\n... [log truncated] ...\n\n" + log[-half:]


def _extract_evidence_lines(log: str, keywords: list[str], limit: int = 3) -> list[str]:
    evidence: list[str] = []
    lowered_keywords = [value.lower() for value in keywords]
    for raw_line in log.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        lowered = line.lower()
        if any(keyword in lowered for keyword in lowered_keywords):
            evidence.append(line)
            if len(evidence) >= limit:
                break
    return evidence


def extract_failure_hints(log: str, phase: str, exit_code: int, timed_out: bool) -> dict:
    lowered = log.lower()

    if timed_out or exit_code == 124:
        return {
            "phase": phase,
            "category": "timeout",
            "confidence": "high",
            "exit_code": exit_code,
            "timed_out": timed_out,
            "evidence": _extract_evidence_lines(log, ["timeout", "timed out", "exit_code: 124"]),
        }

    category_rules: list[tuple[str, list[str], str]] = [
        ("python_missing", ["env: 'python': no such file or directory", "env: ''python'': no such file or directory", "python: not found"], "high"),
        ("shell_missing", ["no usable shell found", "unable to find executable file", "exec: \"/bin/sh\": stat /bin/sh: no such file or directory"], "high"),
        ("missing_command", ["command not found", "not found in $path", "executable file not found"], "high"),
        ("permission_error", ["permission denied", "operation not permitted", "eacces"], "high"),
        ("network_tls", ["certificate verify failed", "x509", "pkix", "unable to get local issuer certificate", "self-signed certificate"], "medium"),
        ("network_resolution", ["could not resolve host", "temporary failure resolving", "connection timed out"], "medium"),
        ("missing_dependency", ["unable to locate package", "no package", "not installed", "could not find", "fatal error:"], "medium"),
    ]

    for category, keywords, confidence in category_rules:
        if any(keyword in lowered for keyword in keywords):
            return {
                "phase": phase,
                "category": category,
                "confidence": confidence,
                "exit_code": exit_code,
                "timed_out": timed_out,
                "evidence": _extract_evidence_lines(log, keywords),
            }

    return {
        "phase": phase,
        "category": "unknown",
        "confidence": "low",
        "exit_code": exit_code,
        "timed_out": timed_out,
        "evidence": _extract_evidence_lines(log, ["error", "failed", "exception"]),
    }


def render_failure_hints_for_prompt(failure_hints: dict | None) -> str:
    if not failure_hints:
        return ""
    return json.dumps(failure_hints, indent=2, sort_keys=True)


def _fingerprint_text(content: str) -> str:
    return hashlib.sha256(content.encode("utf-8")).hexdigest()[:12]


def _summarize_failure_hints_for_history(payload):
    if isinstance(payload, dict):
        summary: dict = {}
        keys = ["phase", "category", "confidence", "exit_code", "timed_out", "evidence"]
        for key in keys:
            if key in payload:
                summary[key] = payload[key]
        if summary:
            return summary
        return {k: _summarize_failure_hints_for_history(v) for k, v in payload.items()}
    if isinstance(payload, list):
        return [_summarize_failure_hints_for_history(value) for value in payload]
    return payload


def render_stateful_history_for_prompt(repair_history: list[dict]) -> str:
    if not repair_history:
        return ""

    window = max(1, args.stateful_history_window)
    scoped_history = repair_history[-window:]
    rendered = render_yaml({"previous_repair_attempts": scoped_history})

    max_chars = max(1, args.stateful_history_max_chars)
    if len(rendered) > max_chars:
        rendered = "... [stateful repair history truncated] ...\n" + rendered[-max_chars:]

    return rendered


def append_stateful_repair_history(
    repair_history: list[dict],
    *,
    attempt: int,
    trigger: str,
    prior_dockerfile: str,
    repaired_dockerfile: str,
    should_stop: bool,
    failure_hints: dict,
    build_exit_code: int | None = None,
    verify_exit_code: int | None = None,
    verify_retry_exit_code: int | None = None,
) -> None:
    repair_history.append(
        {
            "attempt": attempt,
            "trigger": trigger,
            "build_exit_code": build_exit_code,
            "verify_exit_code": verify_exit_code,
            "verify_retry_exit_code": verify_retry_exit_code,
            "failure_hints": _summarize_failure_hints_for_history(failure_hints),
            "repair_result": {
                "changed_dockerfile": repaired_dockerfile.strip() != prior_dockerfile.strip(),
                "returned_empty": not repaired_dockerfile.strip(),
                "stopped_retries": should_stop,
                "prior_dockerfile_fingerprint": _fingerprint_text(prior_dockerfile),
                "repaired_dockerfile_fingerprint": _fingerprint_text(repaired_dockerfile) if repaired_dockerfile.strip() else "",
            },
        }
    )


def _extract_failure_categories(payload) -> list[str]:
    categories: list[str] = []

    def _walk(value) -> None:
        if isinstance(value, dict):
            category = value.get("category")
            if isinstance(category, str) and category.strip():
                categories.append(category.strip())
            for nested in value.values():
                _walk(nested)
            return
        if isinstance(value, list):
            for nested in value:
                _walk(nested)

    _walk(payload)

    deduped: list[str] = []
    for category in categories:
        if category not in deduped:
            deduped.append(category)
    return deduped


def _get_or_create_tree_child(node: dict, key: str) -> dict:
    for child in node["children"]:
        if child.get("key") == key:
            return child
    created = {
        "key": key,
        "attempts": 0,
        "children": [],
        "changed_count": 0,
        "returned_empty_count": 0,
        "stopped_retries_count": 0,
        "latest_attempt": 0,
    }
    node["children"].append(created)
    return created


def _prune_decision_tree(node: dict, max_children: int) -> dict:
    children = node.get("children", [])
    for child in children:
        _prune_decision_tree(child, max_children)

    if max_children > 0 and len(children) > max_children:
        children.sort(key=lambda child: (child.get("attempts", 0), child.get("latest_attempt", 0)), reverse=True)
        node["children"] = children[:max_children]

    return node


def build_stateful_decision_tree(repair_history: list[dict]) -> dict:
    tree = {
        "key": "root",
        "attempts": len(repair_history),
        "children": [],
    }

    for item in repair_history:
        attempt = int(item.get("attempt", 0) or 0)
        trigger = str(item.get("trigger") or "unknown_trigger")
        categories = _extract_failure_categories(item.get("failure_hints", {}))
        if not categories:
            categories = ["unknown_category"]

        repair_result = item.get("repair_result", {}) if isinstance(item.get("repair_result"), dict) else {}
        changed = bool(repair_result.get("changed_dockerfile", False))
        returned_empty = bool(repair_result.get("returned_empty", False))
        stopped = bool(repair_result.get("stopped_retries", False))

        trigger_node = _get_or_create_tree_child(tree, f"trigger:{trigger}")
        trigger_node["attempts"] += 1
        trigger_node["latest_attempt"] = max(trigger_node.get("latest_attempt", 0), attempt)

        for category in categories:
            category_node = _get_or_create_tree_child(trigger_node, f"category:{category}")
            category_node["attempts"] += 1
            category_node["latest_attempt"] = max(category_node.get("latest_attempt", 0), attempt)
            if changed:
                category_node["changed_count"] += 1
            if returned_empty:
                category_node["returned_empty_count"] += 1
            if stopped:
                category_node["stopped_retries_count"] += 1

    return _prune_decision_tree(tree, max(1, args.stateful_tree_max_children))


def render_stateful_decision_tree_for_prompt(repair_history: list[dict]) -> str:
    if not repair_history:
        return ""

    rendered = render_yaml({"stateful_repair_decision_tree": build_stateful_decision_tree(repair_history)})
    max_chars = max(1, args.stateful_tree_max_chars)
    if len(rendered) > max_chars:
        rendered = "... [stateful repair decision tree truncated] ...\n" + rendered[-max_chars:]

    return rendered


def normalize_verify_command(command: str) -> str:
    sanitized_command = command.strip()
    if not sanitized_command:
        return "echo build-ok"
    return (
        "export PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin; "
        f"{sanitized_command}"
    )


def _extract_verify_binary_name(command: str) -> str:
    """Best-effort extraction of the primary executable from a verify command."""
    stripped = command.strip()
    if not stripped:
        return ""

    # Ignore optional PATH export prefix used by normalize_verify_command.
    if stripped.startswith("export PATH=") and ";" in stripped:
        stripped = stripped.split(";", 1)[1].strip()

    primary = re.split(r"\s|&&|\|\|", stripped, maxsplit=1)[0].strip()
    if not primary or primary in {"if", "test", "[", "echo", "true", "false"}:
        return ""
    return primary


def build_deterministic_verify_fallback(repo_name: str, verify_command: str) -> str:
    """Create a deterministic fallback verify command for command-not-found cases."""
    primary = _extract_verify_binary_name(verify_command)
    candidates: list[str] = []

    for value in [primary, repo_name, repo_name.replace("_", "-")]:
        cleaned = value.strip()
        if cleaned and cleaned not in candidates:
            candidates.append(cleaned)

    shell_candidates = " ".join(f'"{candidate}"' for candidate in candidates)

    # Deterministic fallback:
    # 1) Try likely binary names.
    # 2) If no binary exists, confirm build artifacts exist in known build/output paths.
    return (
        "set -eu; "
        f"for cmd in {shell_candidates}; do "
        "if command -v \"$cmd\" >/dev/null 2>&1; then "
        "echo \"fallback: found command $cmd\"; "
        "\"$cmd\" --version >/dev/null 2>&1 || \"$cmd\" -V >/dev/null 2>&1 || \"$cmd\" -v >/dev/null 2>&1 || true; "
        "exit 0; "
        "fi; "
        "done; "
        "for base in \"$PWD\" /home/manualrepos/repo /workspace /work /app /src /repo /home/manualrepos; do "
        "if [ -d \"$base\" ] && find \"$base\" -maxdepth 6 -type f "
        "\\( -path '*/target/release/*' -o -path '*/build/*' -o -name '*.so*' -o -name '*.a' -o -name '*.jar' -o -name '*.whl' \\) "
        "| head -n 1 | grep -q .; then "
        "echo 'fallback: found build artifact in repository'; "
        "exit 0; "
        "fi; "
        "done; "
        "exit 127"
    )


def _apply_repair(
    repo_url: str,
    repo_name: str,
    current: str,
    repaired: str,
    dockerfile_path: Path,
    report_dir: Path,
    attempt: int,
) -> tuple[str, bool]:
    """Validate and write a repaired Dockerfile. Returns (updated_content, should_stop)."""
    if not repaired.strip():
        log_warn(f"Repair model returned an empty Dockerfile for {repo_url}; stopping retries.")
        return current, True
    if repaired.strip() == current.strip():
        log_warn(f"Repair model returned an unchanged Dockerfile for {repo_url}; stopping retries.")
        return current, True
    write_text(dockerfile_path, repaired)
    write_text(report_dir / f"attempt-{attempt}.repaired.Dockerfile", repaired)
    log_trace(f"Updated Dockerfile for {repo_name} after attempt {attempt}")
    return repaired, False


async def run_build(command: list[str], repo_name: str, attempt: int) -> tuple[int, str]:
    process = await asyncio.create_subprocess_exec(
        *command,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
    )

    output_chunks: list[str] = []
    assert process.stdout is not None

    while True:
        line = await process.stdout.readline()
        if not line:
            break

        decoded_line = line.decode("utf-8", errors="replace")
        output_chunks.append(decoded_line)
        log_info(f"[build {repo_name} attempt {attempt}] {decoded_line.rstrip()}")

    returncode = await process.wait()
    return returncode, "".join(output_chunks)


async def get_image_runtime_context(image_tag: str) -> tuple[str, str]:
    """Resolve runtime user/workdir from built image config for verify execution."""
    command = [
        args.container_cli,
        "image",
        "inspect",
        "--format",
        "{{.Config.User}}|{{.Config.WorkingDir}}",
        image_tag,
    ]
    process = await asyncio.create_subprocess_exec(
        *command,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
    )
    output, _ = await process.communicate()
    if process.returncode != 0:
        return "", ""

    rendered = output.decode("utf-8", errors="replace").strip()
    if "|" not in rendered:
        return "", ""
    user, workdir = rendered.split("|", 1)
    return user.strip(), workdir.strip()


def _extract_direct_exec_command(smoke_command: str) -> list[str] | None:
    command = smoke_command.strip()
    if command.startswith("export PATH=") and ";" in command:
        command = command.split(";", 1)[1].strip()

    # Direct exec fallback only works for plain argv commands (no shell operators).
    if re.search(r"[;&|<>`$()]", command):
        return None

    try:
        argv = shlex.split(command)
    except ValueError:
        return None
    return argv or None


def _is_missing_shell(shell_path: str, output: str) -> bool:
    lowered = output.lower()
    return shell_path.lower() in lowered and (
        "no such file or directory" in lowered or "executable file not found" in lowered
    )


async def _run_container_command(command: list[str], repo_name: str, attempt: int) -> tuple[int, str, bool]:
    process = await asyncio.create_subprocess_exec(
        *command,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
    )

    output_chunks: list[str] = []
    assert process.stdout is not None

    while True:
        line = await process.stdout.readline()
        if not line:
            break

        decoded_line = line.decode("utf-8", errors="replace")
        output_chunks.append(decoded_line)
        log_info(f"[verify {repo_name} attempt {attempt}] {decoded_line.rstrip()}")

    timed_out = False
    try:
        returncode = await asyncio.wait_for(process.wait(), timeout=args.verify_timeout)
    except asyncio.TimeoutError:
        timed_out = True
        process.kill()
        await process.wait()
        returncode = 124

    return returncode, "".join(output_chunks), timed_out


async def run_build_verification(image_tag: str, repo_name: str, attempt: int, smoke_command: str) -> tuple[int, str, list[str], bool]:
    user, workdir = await get_image_runtime_context(image_tag)
    base_command = [
        args.container_cli,
        "run",
        "--rm",
    ]

    if user:
        base_command.extend(["--user", user])
    if workdir:
        base_command.extend(["--workdir", workdir])

    for shell_path in ["/bin/sh", "/busybox/sh", "/bin/bash"]:
        command = base_command + [
            "--entrypoint",
            shell_path,
            image_tag,
            "-lc",
            smoke_command,
        ]
        exit_code, output, timed_out = await _run_container_command(command, repo_name, attempt)
        if timed_out:
            return exit_code, output, command, timed_out
        if not _is_missing_shell(shell_path, output):
            return exit_code, output, command, timed_out

    direct_argv = _extract_direct_exec_command(smoke_command)
    if direct_argv:
        command = base_command + [
            "--entrypoint",
            direct_argv[0],
            image_tag,
            *direct_argv[1:],
        ]
        exit_code, output, timed_out = await _run_container_command(command, repo_name, attempt)
        return exit_code, output, command, timed_out

    command = base_command + ["--entrypoint", "/bin/sh", image_tag, "-lc", smoke_command]
    output = "No usable shell found in image for verification command and command could not be executed directly."
    return 127, output, command, False


async def request_repair(
    repo_url: str,
    attempt_number: int,
    classification: dict,
    summary: str,
    dockerfile_content: str,
    build_log: str,
    failure_hints: dict | None,
    llm_metrics: dict,
    repair_history: list[dict] | None = None,
    architecture_scratchpad_context: str = "",
) -> str:
    base_template = get_base_template(
        classification,
        Path(__file__).resolve().parents[3] / "templates",
        log_warn=log_warn,
        log_error=log_error,
    )
    prompt = (
        PROMPT_TEMPLATE.replace("{{REPO_URL}}", repo_url)
        .replace("{{ATTEMPT_NUMBER}}", str(attempt_number))
        .replace("{{BASE_TEMPLATE_CONTENT}}", base_template)
        .replace("{{CLASSIFICATION_RESULT}}", render_yaml(classification))
        .replace("{{SUMMARY_CONTENT}}", summary)
        .replace("{{DOCKERFILE_CONTENT}}", dockerfile_content)
        .replace("{{BUILD_LOG}}", trim_log(build_log))
    )
    if failure_hints:
        prompt += (
            "\n\nUse these normalized failure hints as primary guidance for your fix."
            "\nPrefer addressing high-confidence categories first."
            f"\n\nFAILURE_HINTS_JSON:\n{render_failure_hints_for_prompt(failure_hints)}\n"
        )
        if failure_hints.get("category") == "python_missing":
            prompt += (
                "\nTargeted remediation hint: build tooling expects `python` on PATH."
                "\nPrefer adding `python-is-python3` (Debian/Ubuntu) or a safe equivalent symlink in the image,"
                "\nrather than editing project source or skipping required build steps.\n"
            )
    if args.stateful_repair and repair_history:
        prompt += (
            "\n\nStateful repair mode is enabled."
            "\nUse the following summaries of prior repair attempts to avoid repeating ineffective fixes."
            f"\n\nSTATEFUL_REPAIR_HISTORY:\n{render_stateful_history_for_prompt(repair_history)}\n"
        )
        if args.stateful_repair_tree:
            prompt += (
                "\nAlso use this compact decision tree to choose strategy shifts when repeated branches fail."
                f"\n\nSTATEFUL_REPAIR_DECISION_TREE:\n{render_stateful_decision_tree_for_prompt(repair_history)}\n"
            )
    if architecture_scratchpad_context:
        prompt += architecture_scratchpad_context

    repaired, trace_steps = await run_l3_dockerfile_repair_react(
        repo_url=repo_url,
        attempt_number=attempt_number,
        prompt=prompt,
        repair_timeout=args.repair_timeout,
        l3_react_max_steps=args.l3_react_max_steps,
        new_prebuilt_chat_model=_new_prebuilt_chat_model,
        build_think_tool=build_think_tool,
        build_hadolint_snippet_tool=build_hadolint_snippet_tool,
        extract_dockerfile=extract_dockerfile,
    )

    if not repaired:
        log_warn(f"L3 ReAct repair agent returned empty content for {repo_url} attempt {attempt_number}")
        return ""

    llm_metrics.setdefault("l3_react", {}).update(
        {
            "enabled": True,
            "max_steps": int(args.l3_react_max_steps),
            "last_attempt": attempt_number,
            "trace_steps": trace_steps,
        }
    )
    return repaired


async def request_verification_command_repair(
    repo_url: str,
    classification: dict,
    dockerfile_content: str,
    current_verify_command: str,
    verify_log: str,
    failure_hints: dict | None,
    llm_metrics: dict,
    architecture_scratchpad_context: str = "",
) -> str:
    prompt = (
        VERIFY_PROMPT_TEMPLATE
        .replace("{{REPO_URL}}", repo_url)
        .replace("{{CLASSIFICATION_RESULT}}", render_yaml(classification))
        .replace("{{DOCKERFILE_CONTENT}}", dockerfile_content)
    )
    prompt += (
        "\n\nThe previous verification command failed."
        "\nReturn a replacement command that addresses the failure evidence below."
        "\nDo not require a background daemon unless the Dockerfile clearly starts it within the same one-liner."
        "\nPreserve the same verification intent, but prefer offline, foreground, deterministic checks."
        "\nPrefer a quick smoke check over a heavyweight full-suite test by default."
        "\nIf failure evidence indicates command-not-found or exit code 127, choose a command that exists in the final image (use command -v guards when uncertain)."
        "\nDo not assume the executable name equals the repository name."
        "\nReturn only one shell command, no prose."
        f"\n\nPrevious command:\n{current_verify_command}"
        f"\n\nFailure log:\n{trim_log(verify_log)}\n"
    )
    if failure_hints:
        prompt += (
            "\nUse these normalized failure hints to choose a command that matches the actual runtime context."
            f"\n\nFAILURE_HINTS_JSON:\n{render_failure_hints_for_prompt(failure_hints)}\n"
        )
    if architecture_scratchpad_context:
        prompt += architecture_scratchpad_context

    command, trace_steps = await run_l3_verification_command_react(
        repo_url=repo_url,
        prompt=prompt,
        verify_timeout=args.verify_repair_timeout,
        l3_react_max_steps=args.l3_react_max_steps,
        thread_suffix="l3-verify-repair",
        system_prompt=(
            "You are Loop 3 (L3) verification-command repair ReAct agent. "
            "Use think before major command changes and return only YAML-compatible output. "
            "Return keys: thought, verification_command, done, stop_reason."
        ),
        candidate_keys=["verification_command", "verify_command", "repaired_verification_command", "command"],
        new_prebuilt_chat_model=_new_prebuilt_chat_model,
        build_think_tool=build_think_tool,
    )
    if not command:
        log_warn(f"L3 ReAct verify-repair agent returned empty command for {repo_url}")
        return ""

    l3_metrics = llm_metrics.setdefault("l3_react", {})
    l3_metrics.update(
        {
            "verify_repair_enabled": True,
            "verify_repair_trace_steps": trace_steps,
            "verify_repair_max_steps": int(args.l3_react_max_steps),
        }
    )
    return command


async def request_verification_command_refresh(
    repo_url: str,
    classification: dict,
    dockerfile_content: str,
    current_verify_command: str,
    llm_metrics: dict,
    architecture_scratchpad_context: str = "",
) -> str:
    """Refresh verify command when Dockerfile changes materially during repair."""
    prompt = (
        VERIFY_PROMPT_TEMPLATE
        .replace("{{REPO_URL}}", repo_url)
        .replace("{{CLASSIFICATION_RESULT}}", render_yaml(classification))
        .replace("{{DOCKERFILE_CONTENT}}", dockerfile_content)
    )
    prompt += (
        "\n\nThe Dockerfile was rewritten during repair."
        "\nReturn the best verification command for the updated Dockerfile context."
        "\nPrefer a quick deterministic check over heavyweight full-suite tests by default."
        "\nDo not assume the executable name equals the repository name."
        "\nUse command -v guards when command availability is uncertain."
        "\nUse one shell command only, no prose or markdown fences."
        f"\n\nCurrent command:\n{current_verify_command}\n"
    )
    if architecture_scratchpad_context:
        prompt += architecture_scratchpad_context

    command, trace_steps = await run_l3_verification_command_react(
        repo_url=repo_url,
        prompt=prompt,
        verify_timeout=args.verify_repair_timeout,
        l3_react_max_steps=args.l3_react_max_steps,
        thread_suffix="l3-verify-refresh",
        system_prompt=(
            "You are Loop 3 (L3) verification-command refresh ReAct agent. "
            "Use think before major command changes and return only YAML-compatible output. "
            "Return keys: thought, verification_command, done, stop_reason."
        ),
        candidate_keys=["verification_command", "verify_command", "refreshed_verification_command", "command"],
        new_prebuilt_chat_model=_new_prebuilt_chat_model,
        build_think_tool=build_think_tool,
    )
    if not command:
        log_warn(f"L3 ReAct verify-refresh agent returned empty command for {repo_url}")
        return ""

    l3_metrics = llm_metrics.setdefault("l3_react", {})
    l3_metrics.update(
        {
            "verify_refresh_enabled": True,
            "verify_refresh_trace_steps": trace_steps,
            "verify_refresh_max_steps": int(args.l3_react_max_steps),
        }
    )
    return command


async def repair_repository(
    repo_url: str,
    repos_dir: Path,
    summaries_dir: Path,
    results_dir: Path,
    dockerfiles_dir: Path,
    reports_dir: Path,
    progress_state: dict,
) -> None:
    async with sem:
        repo_name = repo_name_from_url(repo_url)
        dockerfile_path = dockerfiles_dir / f"{repo_name}.Dockerfile"
        report_dir = reports_dir / repo_name
        report_path = report_dir / "report.yaml"
        llm_metrics_path = report_dir / "llm-metrics.yaml"
        llm_metrics = init_llm_metrics(repo_url, args.model, args.endpoint, args.timeout, args.llm_max_retries)
        llm_metrics["prompt_profile"] = prompt_profile_metadata(PROMPT_PROFILE, EFFECTIVE_TEMPERATURE)

        report: dict = {
            "repo": repo_url,
            "dockerfile": str(dockerfile_path),
            "max_attempts": args.max_attempts,
            "success": False,
            "attempts": [],
            "prompt_profile": prompt_profile_metadata(PROMPT_PROFILE, EFFECTIVE_TEMPERATURE),
            "stateful_repair": {
                "enabled": args.stateful_repair,
                "tree_enabled": args.stateful_repair_tree,
                "history_window": args.stateful_history_window,
                "history_max_chars": args.stateful_history_max_chars,
                "tree_max_chars": args.stateful_tree_max_chars,
                "tree_max_children": args.stateful_tree_max_children,
            },
        }

        try:
            classification_path = results_dir / f"{repo_name}.yaml"
            classification = read_yaml_file(classification_path)
            if not classification:
                log_warn(
                    f"Skipping {repo_url}: classification result missing at {classification_path}. Run agent_classify.py first."
                )
                return

            if not dockerfile_path.exists():
                log_warn(
                    f"Skipping {repo_url}: Dockerfile missing at {dockerfile_path}. Run agent_dockerfile.py first."
                )
                return

            # Resume: skip repair if a successful report already exists and --force is not set.
            if not args.force and report_path.exists():
                existing_report = read_yaml_file(report_path)
                if existing_report and existing_report.get("success"):
                    log_info(f"Skipping {repo_url}: existing successful repair report found at {report_path}")
                    return

            repo_path = repos_dir / repo_name
            if not await ensure_repo_checkout(repo_url, repo_path, "skipping Dockerfile repair"):
                return

            # Reset repo to clean state for reproducibility
            try:
                result = await asyncio.create_subprocess_exec("git", "-C", str(repo_path), "reset", "--hard", "HEAD", stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
                returncode = await result.wait()
                if returncode == 0:
                    log_info(f"[git-reset {repo_name}] git reset --hard HEAD successful")
                result = await asyncio.create_subprocess_exec("git", "-C", str(repo_path), "clean", "-fdx", stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
                returncode = await result.wait()
                if returncode == 0:
                    log_info(f"[git-clean {repo_name}] git clean -fdx successful")
            except Exception as e:
                log_warn(f"[git-reset {repo_name}] Failed to reset/clean repo: {e}")

            if not args.skip_delete_docs:
                await asyncio.to_thread(delete_docs_build_context, repo_path, repo_name)
            else:
                log_info(f"[delete-docs {repo_name}] Skipping docs/CI deletion (--skip-delete-docs set)")

            # Load the build verification command: prefer LLM-generated sidecar, fall back to CLI arg.
            verify_command_path = dockerfiles_dir / f"{repo_name}.verify-command"
            if verify_command_path.exists():
                verify_command_to_use = normalize_verify_command(verify_command_path.read_text(encoding="utf-8"))
                log_info(f"[verify {repo_name}] Using generated verification command from {verify_command_path}: {verify_command_to_use}")
            else:
                verify_command_to_use = normalize_verify_command(args.verify_command)
                log_info(f"[verify {repo_name}] No generated verification command found; using default: {verify_command_to_use}")

            summary = load_summary(repo_name, repo_path, summaries_dir)
            architecture_scratchpad = load_architecture_scratchpad(repo_name, summaries_dir)
            shared_repository_state = load_shared_repository_state(repo_name, summaries_dir)
            validation_artifact = read_yaml_file(summaries_dir / f"{repo_name}.validation.yaml")
            postgen_validation_artifact = read_yaml_file(summaries_dir / f"{repo_name}.postgen-validation.yaml")
            if isinstance(postgen_validation_artifact, dict):
                decision = postgen_validation_artifact.get("decision")
                if isinstance(decision, dict) and decision.get("run_repair") is False:
                    reason = str(decision.get("reason", "validation_gate_blocked"))
                    log_warn(f"Skipping repair for {repo_url}: post-generation validation gate blocked repair ({reason}).")
                    upsert_shared_repository_state(
                        repo_name,
                        summaries_dir,
                        repo_url=repo_url,
                        stage_name="repair",
                        stage_update={
                            "status": "skipped_by_validation_gate",
                            "reason": reason,
                            "postgen_validation_artifact": str(summaries_dir / f"{repo_name}.postgen-validation.yaml"),
                        },
                    )
                    return
            architecture_scratchpad_context = (
                render_validation_findings_for_prompt(validation_artifact)
                + render_architecture_scratchpad_for_prompt(architecture_scratchpad)
                + render_shared_repository_state_for_prompt(shared_repository_state)
            )
            current_dockerfile = dockerfile_path.read_text(encoding="utf-8")
            repair_history: list[dict] = []
            report_dir.mkdir(parents=True, exist_ok=True)

            for attempt in range(1, args.max_attempts + 1):
                image_tag = f"{sanitize_image_tag(repo_name)}-repair-{attempt}"
                build_log_path = report_dir / f"attempt-{attempt}.build.log"
                dockerfile_snapshot_path = report_dir / f"attempt-{attempt}.Dockerfile"
                
                # Inject CA cert setup if present, to fix TLS/certificate errors behind corporate proxies
                dockerfile_for_build = inject_ca_cert_into_dockerfile(current_dockerfile)
                
                # Write injected dockerfile temporarily for build, but keep original for repair prompt
                build_command = [
                    args.container_cli,
                    "build",
                    "-f",
                    str(dockerfile_path),
                    "-t",
                    image_tag,
                    str(repo_path),
                ]
                
                # Temporarily replace dockerfile with injected version for build
                original_dockerfile_content = dockerfile_path.read_text(encoding="utf-8")
                write_text(dockerfile_path, dockerfile_for_build)

                # Check Dockerfile syntax with hadolint (if not skipped)
                # Hadolint errors are prepended to build log for LLM feedback, but build still attempts
                hadolint_error = ""
                if not args.skip_hadolint:
                    is_valid, validation_error = await validate_dockerfile_syntax(dockerfile_path, repo_name)
                    if not is_valid:
                        hadolint_error = validation_error[:1000]
                        log_warn(f"[hadolint {repo_name}] Dockerfile syntax warning: {hadolint_error[:200]}")

                log_info(f"Build attempt {attempt}/{args.max_attempts} for {repo_url}...")
                log_info(f"Streaming build output; full log will be written to {build_log_path}")
                exit_code, streamed_output = await run_build(build_command, repo_name, attempt)
                build_log = combine_build_output(build_command, exit_code, streamed_output)
                
                # Prepend hadolint errors to build log for repair LLM
                if hadolint_error:
                    build_log = f"[HADOLINT VALIDATION WARNING]\n{hadolint_error}\n\n[DOCKER BUILD OUTPUT]\n{build_log}"

                # Restore original dockerfile for repair prompt
                write_text(dockerfile_path, original_dockerfile_content)

                write_text(build_log_path, build_log)
                write_text(dockerfile_snapshot_path, current_dockerfile)

                report["attempts"].append(
                    {
                        "attempt": attempt,
                        "exit_code": exit_code,
                        "image_tag": image_tag,
                        "build_log": str(build_log_path),
                        "dockerfile_snapshot": str(dockerfile_snapshot_path),
                    }
                )

                build_failure_hints = extract_failure_hints(build_log, "build", exit_code, timed_out=False)
                report["attempts"][-1]["failure_hints_build"] = build_failure_hints
                upsert_shared_repository_state(
                    repo_name,
                    summaries_dir,
                    repo_url=repo_url,
                    stage_name="repair",
                    stage_update={"status": "in_progress", "attempt": attempt},
                    failure_hint=build_failure_hints,
                )

                if exit_code == 0:
                    verify_log_path = report_dir / f"attempt-{attempt}.verify.log"
                    log_info(
                        f"Running build verification for {repo_url} using command: {verify_command_to_use}"
                    )
                    verify_exit_code, verify_output, verify_cmd_list, verify_timed_out = await run_build_verification(
                        image_tag,
                        repo_name,
                        attempt,
                        verify_command_to_use,
                    )
                    verify_log = combine_build_output(verify_cmd_list, verify_exit_code, verify_output)
                    write_text(verify_log_path, verify_log)
                    report["attempts"][-1]["build_verification"] = {
                        "exit_code": verify_exit_code,
                        "timed_out": verify_timed_out,
                        "command": verify_cmd_list,
                        "log": str(verify_log_path),
                    }
                    verify_failure_hints = extract_failure_hints(verify_log, "verify", verify_exit_code, verify_timed_out)
                    report["attempts"][-1]["failure_hints_verify"] = verify_failure_hints
                    upsert_shared_repository_state(
                        repo_name,
                        summaries_dir,
                        repo_url=repo_url,
                        stage_name="repair",
                        stage_update={"status": "in_progress", "attempt": attempt},
                        failure_hint=verify_failure_hints,
                    )

                    if verify_exit_code == 0:
                        report["success"] = True
                        report["successful_attempt"] = attempt
                        log_info(
                            f"Build and verification succeeded for {repo_url} on attempt {attempt}; image tag: {image_tag}; build log: {build_log_path}; verify log: {verify_log_path}"
                        )
                        break

                    if verify_exit_code == 127:
                        fallback_command = build_deterministic_verify_fallback(repo_name, verify_command_to_use)
                        fallback_log_path = report_dir / f"attempt-{attempt}.verify-fallback.log"
                        log_info(
                            f"Verification command not found for {repo_url}; running deterministic fallback verification"
                        )
                        fb_exit_code, fb_output, fb_cmd_list, fb_timed_out = await run_build_verification(
                            image_tag,
                            repo_name,
                            attempt,
                            fallback_command,
                        )
                        fallback_log = combine_build_output(fb_cmd_list, fb_exit_code, fb_output)
                        write_text(fallback_log_path, fallback_log)
                        report["attempts"][-1]["build_verification_fallback"] = {
                            "exit_code": fb_exit_code,
                            "timed_out": fb_timed_out,
                            "command": fb_cmd_list,
                            "log": str(fallback_log_path),
                        }
                        if fb_exit_code == 0:
                            report["success"] = True
                            report["successful_attempt"] = attempt
                            log_info(
                                f"Build and deterministic fallback verification succeeded for {repo_url} on attempt {attempt}; image tag: {image_tag}; build log: {build_log_path}; fallback log: {fallback_log_path}"
                            )
                            break

                    log_warn(
                        f"Build verification failed for {repo_url} on attempt {attempt} with exit code {verify_exit_code}; verify log: {verify_log_path}"
                    )

                    log_info(f"Diagnosing failed build verification for {repo_url} and rewriting the verification command...")
                    repaired_verify_command = normalize_verify_command(
                        await request_verification_command_repair(
                            repo_url=repo_url,
                            classification=classification,
                            dockerfile_content=current_dockerfile,
                            current_verify_command=verify_command_to_use,
                            verify_log=verify_log,
                            failure_hints=verify_failure_hints,
                            llm_metrics=llm_metrics,
                            architecture_scratchpad_context=architecture_scratchpad_context,
                        )
                    )

                    if repaired_verify_command == verify_command_to_use:
                        log_warn(
                            f"Verification command repair produced no change for {repo_url}; preserving Dockerfile and stopping verification retries"
                        )
                        if attempt == args.max_attempts:
                            log_warn(f"Build verification still failing for {repo_url} after {args.max_attempts} attempts")
                        break

                    verify_command_to_use = repaired_verify_command
                    write_text(verify_command_path, verify_command_to_use + "\n")
                    log_info(
                        f"Retrying build verification for {repo_url} using updated command: {verify_command_to_use}"
                    )
                    retry_verify_log_path = report_dir / f"attempt-{attempt}.verify-retry.log"
                    retry_exit_code, retry_output, retry_cmd_list, retry_timed_out = await run_build_verification(
                        image_tag,
                        repo_name,
                        attempt,
                        verify_command_to_use,
                    )
                    retry_verify_log = combine_build_output(retry_cmd_list, retry_exit_code, retry_output)
                    write_text(retry_verify_log_path, retry_verify_log)
                    report["attempts"][-1]["build_verification_retry"] = {
                        "exit_code": retry_exit_code,
                        "timed_out": retry_timed_out,
                        "command": retry_cmd_list,
                        "log": str(retry_verify_log_path),
                    }
                    retry_failure_hints = extract_failure_hints(retry_verify_log, "verify-retry", retry_exit_code, retry_timed_out)
                    report["attempts"][-1]["failure_hints_verify_retry"] = retry_failure_hints
                    upsert_shared_repository_state(
                        repo_name,
                        summaries_dir,
                        repo_url=repo_url,
                        stage_name="repair",
                        stage_update={"status": "in_progress", "attempt": attempt},
                        failure_hint=retry_failure_hints,
                    )

                    if retry_exit_code == 0:
                        report["success"] = True
                        report["successful_attempt"] = attempt
                        log_info(
                            f"Build and verification succeeded for {repo_url} on attempt {attempt} after updating the verification command; image tag: {image_tag}; build log: {build_log_path}; verify log: {retry_verify_log_path}"
                        )
                        break

                    if retry_exit_code == 127:
                        fallback_command = build_deterministic_verify_fallback(repo_name, verify_command_to_use)
                        fallback_log_path = report_dir / f"attempt-{attempt}.verify-fallback.log"
                        log_info(
                            f"Updated verification command not found for {repo_url}; running deterministic fallback verification"
                        )
                        fb_exit_code, fb_output, fb_cmd_list, fb_timed_out = await run_build_verification(
                            image_tag,
                            repo_name,
                            attempt,
                            fallback_command,
                        )
                        fallback_log = combine_build_output(fb_cmd_list, fb_exit_code, fb_output)
                        write_text(fallback_log_path, fallback_log)
                        report["attempts"][-1]["build_verification_fallback"] = {
                            "exit_code": fb_exit_code,
                            "timed_out": fb_timed_out,
                            "command": fb_cmd_list,
                            "log": str(fallback_log_path),
                        }
                        if fb_exit_code == 0:
                            report["success"] = True
                            report["successful_attempt"] = attempt
                            log_info(
                                f"Build and deterministic fallback verification succeeded for {repo_url} on attempt {attempt} after verification command update; image tag: {image_tag}; build log: {build_log_path}; fallback log: {fallback_log_path}"
                            )
                            break

                    log_warn(
                        f"Updated build verification still failing for {repo_url} on attempt {attempt} with exit code {retry_exit_code}; verify log: {retry_verify_log_path}"
                    )

                    build_log = build_log + "\n\nBUILD_VERIFICATION_LOG:\n" + verify_log + "\n\nBUILD_VERIFICATION_RETRY_LOG:\n" + retry_verify_log

                    if attempt == args.max_attempts:
                        log_warn(f"Build verification still failing for {repo_url} after {args.max_attempts} attempts")
                        break

                    log_info(f"Verification repair was insufficient for {repo_url}; rewriting Dockerfile as a fallback...")
                    repaired_dockerfile = await request_repair(
                        repo_url=repo_url,
                        attempt_number=attempt,
                        classification=classification,
                        summary=summary,
                        dockerfile_content=current_dockerfile,
                        build_log=build_log,
                        failure_hints={
                            "build": build_failure_hints,
                            "verify": verify_failure_hints,
                            "verify_retry": retry_failure_hints,
                        },
                        llm_metrics=llm_metrics,
                        repair_history=repair_history,
                        architecture_scratchpad_context=architecture_scratchpad_context,
                    )
                    prior_dockerfile = current_dockerfile
                    current_dockerfile, stop = _apply_repair(
                        repo_url, repo_name, current_dockerfile, repaired_dockerfile,
                        dockerfile_path, report_dir, attempt,
                    )
                    if args.stateful_repair:
                        append_stateful_repair_history(
                            repair_history,
                            attempt=attempt,
                            trigger="verification_repair_insufficient",
                            prior_dockerfile=prior_dockerfile,
                            repaired_dockerfile=repaired_dockerfile,
                            should_stop=stop,
                            failure_hints={
                                "build": build_failure_hints,
                                "verify": verify_failure_hints,
                                "verify_retry": retry_failure_hints,
                            },
                            build_exit_code=exit_code,
                            verify_exit_code=verify_exit_code,
                            verify_retry_exit_code=retry_exit_code,
                        )
                    if stop:
                        break
                    refreshed_verify_command = normalize_verify_command(
                        await request_verification_command_refresh(
                            repo_url=repo_url,
                            classification=classification,
                            dockerfile_content=current_dockerfile,
                            current_verify_command=verify_command_to_use,
                            llm_metrics=llm_metrics,
                            architecture_scratchpad_context=architecture_scratchpad_context,
                        )
                    )
                    if refreshed_verify_command and refreshed_verify_command != verify_command_to_use:
                        verify_command_to_use = refreshed_verify_command
                        write_text(verify_command_path, verify_command_to_use + "\n")
                        log_info(
                            f"Updated verification command after Dockerfile repair for {repo_url}: {verify_command_to_use}"
                        )

                log_warn(
                    f"Build attempt {attempt} failed for {repo_url} with exit code {exit_code}; build log: {build_log_path}"
                )

                if attempt == args.max_attempts:
                    log_warn(f"Docker build still failing for {repo_url} after {args.max_attempts} attempts")
                    break

                log_info(f"Diagnosing failed build for {repo_url} and rewriting Dockerfile...")
                repaired_dockerfile = await request_repair(
                    repo_url=repo_url,
                    attempt_number=attempt,
                    classification=classification,
                    summary=summary,
                    dockerfile_content=current_dockerfile,
                    build_log=build_log,
                    failure_hints={"build": build_failure_hints},
                    llm_metrics=llm_metrics,
                    repair_history=repair_history,
                    architecture_scratchpad_context=architecture_scratchpad_context,
                )
                prior_dockerfile = current_dockerfile
                current_dockerfile, stop = _apply_repair(
                    repo_url, repo_name, current_dockerfile, repaired_dockerfile,
                    dockerfile_path, report_dir, attempt,
                )
                if args.stateful_repair:
                    append_stateful_repair_history(
                        repair_history,
                        attempt=attempt,
                        trigger="build_failure",
                        prior_dockerfile=prior_dockerfile,
                        repaired_dockerfile=repaired_dockerfile,
                        should_stop=stop,
                        failure_hints={"build": build_failure_hints},
                        build_exit_code=exit_code,
                    )
                if stop:
                    break
                refreshed_verify_command = normalize_verify_command(
                    await request_verification_command_refresh(
                        repo_url=repo_url,
                        classification=classification,
                        dockerfile_content=current_dockerfile,
                        current_verify_command=verify_command_to_use,
                        llm_metrics=llm_metrics,
                        architecture_scratchpad_context=architecture_scratchpad_context,
                    )
                )
                if refreshed_verify_command and refreshed_verify_command != verify_command_to_use:
                    verify_command_to_use = refreshed_verify_command
                    write_text(verify_command_path, verify_command_to_use + "\n")
                    log_info(
                        f"Updated verification command after Dockerfile repair for {repo_url}: {verify_command_to_use}"
                    )

            write_text(report_path, render_yaml(report))
            log_info(f"Repair report written to {report_path}")
            if args.stateful_repair:
                report["stateful_repair"]["history"] = repair_history
                if args.stateful_repair_tree:
                    report["stateful_repair"]["decision_tree"] = build_stateful_decision_tree(repair_history)
            write_text(llm_metrics_path, render_yaml(finalize_llm_metrics(llm_metrics)))
            log_info(f"LLM metrics saved at {llm_metrics_path}")
            upsert_shared_repository_state(
                repo_name,
                summaries_dir,
                repo_url=repo_url,
                stage_name="repair",
                stage_update={
                    "status": "completed" if report.get("success") else "failed",
                    "success": bool(report.get("success", False)),
                    "attempts": len(report.get("attempts", [])),
                    "report_path": str(report_path),
                },
            )

        except httpx.HTTPError as error:
            log_warn(f"HTTP error for {repo_url}: {error}")
        except ssl.SSLError as error:
            log_warn(f"SSL error for {repo_url}: {error}")
        except APITimeoutError as error:
            log_warn(f"OpenAI timeout for {repo_url}: {error}")
        except APIError as error:
            log_warn(f"OpenAI API error for {repo_url}: {error}")
        except Exception as error:
            log_error(f"Unexpected error while repairing Dockerfile for {repo_url}: {error}")
        finally:
            if report["attempts"]:
                write_text(report_path, render_yaml(report))
            write_text(llm_metrics_path, render_yaml(finalize_llm_metrics(llm_metrics)))
            await update_progress(progress_state, repo_name)


async def main() -> None:
    repos = load_repo_urls(args.input_file, args.repo_url)
    if not repos:
        log_error("No repositories to process. Provide --repo-url or a non-empty --input-file.")
        return

    workspace_root = Path(args.input_file).parent
    repos_dir = workspace_root / args.repos_dir
    summaries_dir = workspace_root / args.summaries_dir
    results_dir = workspace_root / args.results_dir
    dockerfiles_dir = workspace_root / args.dockerfiles_dir
    reports_dir = workspace_root / args.reports_dir
    reports_dir.mkdir(parents=True, exist_ok=True)

    progress_bar = None
    if should_use_progress(len(repos), args.trace):
        progress_bar = tqdm(total=len(repos), desc="Repairing Dockerfiles", unit="repo", dynamic_ncols=True)

    progress_state = {
        "lock": asyncio.Lock(),
        "bar": progress_bar,
    }
    set_tqdm_bar(progress_state["bar"])
    log_info(f"Starting Dockerfile repair for {len(repos)} repositories")

    tasks = [
        repair_repository(repo, repos_dir, summaries_dir, results_dir, dockerfiles_dir, reports_dir, progress_state)
        for repo in repos
    ]

    try:
        await asyncio.gather(*tasks, return_exceptions=True)
    finally:
        if progress_state["bar"] is not None:
            progress_state["bar"].close()
        set_tqdm_bar(None)

    log_info("Done.")


if __name__ == "__main__":
    asyncio.run(main())