import argparse
import asyncio
import hashlib
import json
import os
import re
import shlex
import ssl
from pathlib import Path
from typing import Optional

import httpx
import xxhash
import yaml
from openai import APIError, APITimeoutError, AsyncOpenAI
from tqdm import tqdm

try:
    from RepoBuilderAgent.src.core.config import OPENAI_API_KEY, OPENAI_BASE_URL, OPENAI_MODEL
    from RepoBuilderAgent.src.core.log_utils import log_error, log_info, log_trace, log_warn, set_dump_prompts_dir, set_tqdm_bar, set_trace_enabled
    from RepoBuilderAgent.src.agent_tools.react_loop_tools import build_hadolint_snippet_tool, build_get_dockerfile_snippet_tool, build_think_tool, build_read_file_tool, build_list_tree_tool, build_search_pattern_tool, build_apt_search_tool, apt_search_packages
    from RepoBuilderAgent.src.metrics.eval_metrics_lib import load_gt_for_repo, get_gt_verify_commands, get_gt_key_artifact
    from RepoBuilderAgent.src.core.chat_model_factory import make_prebuilt_chat_model_factory
    from RepoBuilderAgent.src.core.agent_runtime import RepairRuntime
    from RepoBuilderAgent.src.core.dockerfile_utils import ensure_base_template, extract_dockerfile, get_base_template, extract_base_image
    from RepoBuilderAgent.src.core.llm_yaml import extract_command_from_reply
    from RepoBuilderAgent.src.core.file_io import write_text
    from RepoBuilderAgent.src.core.repo_cleanup import delete_files_build_context, get_files_to_delete
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
        length_mode_for,
    )
    from RepoBuilderAgent.src.core.common import (
        ensure_repo_checkout,
        resolve_repo_checkout_dir,
        finalize_llm_metrics,
        init_llm_metrics,
        inject_ca_cert_into_dockerfile,
        strip_ca_cert_from_dockerfile,
        sanitize_build_log_for_prompt,
        build_async_http_client,
        chat_completion_with_retries,
        clamp_summary_in_prompt,
        DEFAULT_MAX_INPUT_TOKENS,
        load_architecture_scratchpad,
        load_shared_repository_state,
        load_repo_urls,
        load_summary,
        prompt_path,
        set_prompt_length_mode,
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
    from core.log_utils import log_error, log_info, log_trace, log_warn, set_dump_prompts_dir, set_tqdm_bar, set_trace_enabled
    from agent_tools.react_loop_tools import build_hadolint_snippet_tool, build_get_dockerfile_snippet_tool, build_think_tool, build_read_file_tool, build_list_tree_tool, build_search_pattern_tool, build_apt_search_tool, apt_search_packages
    from metrics.eval_metrics_lib import load_gt_for_repo, get_gt_verify_commands, get_gt_key_artifact
    from core.chat_model_factory import make_prebuilt_chat_model_factory
    from core.agent_runtime import RepairRuntime
    from core.dockerfile_utils import ensure_base_template, extract_dockerfile, get_base_template, extract_base_image
    from core.llm_yaml import extract_command_from_reply
    from core.file_io import write_text
    from core.repo_cleanup import delete_files_build_context, get_files_to_delete
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
        length_mode_for,
    )
    from core.common import (
        ensure_repo_checkout,
        resolve_repo_checkout_dir,
        finalize_llm_metrics,
        init_llm_metrics,
        inject_ca_cert_into_dockerfile,
        strip_ca_cert_from_dockerfile,
        sanitize_build_log_for_prompt,
        build_async_http_client,
        chat_completion_with_retries,
        clamp_summary_in_prompt,
        DEFAULT_MAX_INPUT_TOKENS,
        load_architecture_scratchpad,
        load_shared_repository_state,
        load_repo_urls,
        load_summary,
        prompt_path,
        set_prompt_length_mode,
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
        "repair_max_output_tokens": 4096,
        "build_timeout": 3600,
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
parser.add_argument("--repair-timeout", type=int, default=int(TIMEOUTS["repair_timeout"]), help="Timeout for Dockerfile repair LLM calls in seconds")
parser.add_argument("--verify-repair-timeout", type=int, default=int(TIMEOUTS["verify_repair_timeout"]), help="Timeout for verification-command repair LLM calls in seconds")
parser.add_argument("--repair-max-output-tokens", type=int, default=int(TIMEOUTS["repair_max_output_tokens"]), help="Hard cap on output tokens for repair LLM calls; bounds runaway/non-terminating generations that otherwise hang until the wall-clock timeout")
parser.add_argument("--build-timeout", type=int, default=int(TIMEOUTS["build_timeout"]), help="Hard wall-clock cap (seconds) on a single docker build attempt; a hung build (e.g. a package manager stalling on a registry) is killed and counted as a failed attempt rather than blocking the run indefinitely")
parser.add_argument("--trace", action="store_true", help="Enable verbose trace logs")
parser.add_argument("--dump-prompts", default=None, metavar="PATH", help="Write each rendered prompt to PATH/<repo>/<phase>.<n>.txt before the LLM call")
parser.add_argument("--results-dir", default="classification_results", help="Directory containing classification result YAML files")
parser.add_argument("--summaries-dir", default="summaries", help="Directory containing repository summary files")
parser.add_argument("--repos-dir", default="repos", help="Directory containing cloned repositories")
parser.add_argument("--dockerfiles-dir", default="dockerfiles", help="Directory containing generated Dockerfiles")
parser.add_argument("--reports-dir", default="repair-reports", help="Directory where repair attempt logs and reports will be written")
parser.add_argument("--dataset-dir", default=None, help="Path to RepoBuilderDataset directory; enables GT verify command injection and binary metrics collection")
parser.add_argument("--container-cli", default="docker", help="Container CLI to use for builds")
parser.add_argument("--max-attempts", type=int, default=5, help="Maximum number of build and repair attempts per repository")
parser.add_argument("--max-log-chars", type=int, default=24000, help="Maximum number of build log characters to send to the model")
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
parser.add_argument(
    "--snippet-tools",
    action=argparse.BooleanOptionalAction,
    default=False,
    help="Enable Dockerfile snippet tools (get_dockerfile_snippet) in the L3 repair agent.",
)
parser.add_argument(
    "--repair-repo-tools",
    action=argparse.BooleanOptionalAction,
    default=False,
    help="Give the L3 repair agent read-only repository tools (read_file, list_tree, search_pattern) "
    "so it can inspect the source it is fixing. Building/verifying stays deterministic in the outer loop.",
)
parser.add_argument("--force", action="store_true", help="Re-run repair even if a successful report.yaml already exists")
parser.add_argument(
    "--react-repair",
    dest="react_repair",
    action="store_true",
    help="Use the L3 ReAct repair agent (think/hadolint/finalize tools, multi-step loop).",
)
parser.add_argument(
    "--no-react-repair",
    dest="react_repair",
    action="store_false",
    help="Use the baseline single-shot repair (current Dockerfile + build log -> one LLM call).",
)
parser.set_defaults(react_repair=True)
# Master toggle for the agent-REASONING repair upgrades (F1 strategy ledger, F2 anti-gaming
# lint, LLM failure diagnosis, F6 build/verify separation, F5 malformed-repair guard). ON by
# default; --no-repair-reasoning turns them all off for an A/B that isolates their value from
# the always-on ENVIRONMENT levers (JDK/python3-dev/submodule/gcc-strip), which are NOT gated.
parser.add_argument("--repair-reasoning", dest="repair_reasoning", action="store_true", help="Enable diagnose-then-act / anti-gaming / LLM-diagnosis / build-verify-separation / minimal-diff repair upgrades (default).")
parser.add_argument("--no-repair-reasoning", dest="repair_reasoning", action="store_false", help="Disable the repair-reasoning upgrades (A/B control arm); environment levers stay on.")
parser.set_defaults(repair_reasoning=True)
parser.add_argument(
    "--llm-retry-backoff-seconds",
    type=float,
    default=float(TIMEOUTS["llm_retry_backoff_seconds"]),
    help="Base backoff seconds between LLM retries (used by the single-shot repair call).",
)
parser.add_argument(
    "--max-input-tokens",
    type=int,
    default=DEFAULT_MAX_INPUT_TOKENS,
    help="Hard input-token cap for the repair prompt; the repository summary is trimmed to fit (endpoint-specific, e.g. 64000 for gpt-4o).",
)
args = parser.parse_args()
PROMPT_PROFILE = resolve_prompt_profile(args.prompt_profile)
set_prompt_length_mode(length_mode_for(PROMPT_PROFILE, "repair"))
EFFECTIVE_TEMPERATURE = resolve_prompt_temperature(args.temperature, PROMPT_PROFILE)


# Shared httpx client: OS trust store for corporate CAs + bounded timeout.
_http_client = build_async_http_client(args.timeout)

client = AsyncOpenAI(
    base_url=args.endpoint,
    api_key=args.api_key,
    timeout=args.timeout,
    http_client=_http_client,
)

with open(prompt_path("PROMPT_DOCKERFILE_REPAIR.md"), "r", encoding="utf-8") as prompt_file:
    PROMPT_TEMPLATE = apply_prompt_profile(prompt_file.read(), PROMPT_PROFILE, "repair")

# Baseline single-shot repair prompt (current Dockerfile + build log only), used
# when the ReAct repair agent is disabled (the flat_baseline repair mechanism).
with open(prompt_path("PROMPT_DOCKERFILE_REPAIR_SIMPLE.md"), "r", encoding="utf-8") as prompt_file:
    SIMPLE_PROMPT_TEMPLATE = apply_prompt_profile(prompt_file.read(), PROMPT_PROFILE, "repair")

with open(prompt_path("PROMPT_BUILD_VERIFICATION.md"), "r", encoding="utf-8") as prompt_file:
    VERIFY_PROMPT_TEMPLATE = prompt_file.read()

VERIFY_REPAIR_SYSTEM_PROMPT = prompt_path("PROMPT_L3_VERIFY_REPAIR_SYSTEM.md").read_text(encoding="utf-8").strip()
VERIFY_REFRESH_SYSTEM_PROMPT = prompt_path("PROMPT_L3_VERIFY_REFRESH_SYSTEM.md").read_text(encoding="utf-8").strip()

sem = asyncio.Semaphore(1)

set_trace_enabled(args.trace)
if args.dump_prompts:
    set_dump_prompts_dir(args.dump_prompts)


_new_prebuilt_chat_model = make_prebuilt_chat_model_factory(
    model=args.model,
    temperature=EFFECTIVE_TEMPERATURE,
    api_key=args.api_key,
    base_url=args.endpoint,
    max_retries=args.llm_max_retries,
    http_async_client=_http_client,
)


async def _reset_repo_for_repair(repo_path: "Path", repo_name: str) -> None:
    """Reset the checkout to a clean state (git reset --hard + git clean -fdx) so each
    repair run starts reproducibly. Failures are logged, not fatal."""
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


def _make_repair_runtime() -> RepairRuntime:
    """Bundle the model factory + tool-builder callables the L3 repair/verify loops
    need, so they pass one object instead of five separate params."""
    return RepairRuntime(
        model_name=args.model,
        new_prebuilt_chat_model=_new_prebuilt_chat_model,
        build_think_tool=build_think_tool,
        build_hadolint_snippet_tool=build_hadolint_snippet_tool,
        extract_dockerfile=extract_dockerfile,
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
        # Correct JDK is installed but the system default (update-alternatives, newest wins)
        # points at a too-new JDK — e.g. maven/default-jdk pull java-25 alongside the
        # requested openjdk-21, and the build rejects 25. Distinct from a missing package:
        # the fix is JAVA_HOME, not another apt install.
        ("jdk_version_mismatch", ["not in the allowed range", "unsupported class file major version", "requires java", "invalid target release", "no compiler is provided"], "high"),
        # Python build front-end (meson/cython) is older than the project requires: the
        # distro ships a stale meson/cython and the build rejects it. The base now ships an
        # activated venv, so the fix is `pip install -U` the build tool there — NOT apt.
        ("python_build_tool_stale", ["module \"features\" does not exist", "requires cython", "cython >=", "requires meson", "meson >=", "a newer version of meson", "newer meson"], "high"),
        ("shell_missing", ["no usable shell found", "unable to find executable file", "exec: \"/bin/sh\": stat /bin/sh: no such file or directory"], "high"),
        ("missing_command", ["command not found", "not found in $path", "executable file not found"], "high"),
        ("permission_error", ["permission denied", "operation not permitted", "eacces"], "high"),
        ("network_tls", ["certificate verify failed", "x509", "pkix", "unable to get local issuer certificate", "self-signed certificate"], "medium"),
        ("network_resolution", ["could not resolve host", "temporary failure resolving", "connection timed out"], "medium"),
        ("missing_dependency", ["unable to locate package", "no package", "not installed", "could not find", "fatal error:", "system library", "not found in the pkg-config", "required by crate"], "medium"),
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


_SUPPRESSION_PATTERNS = [
    ("|| true", "`|| true` masks a failing command"),
    ("||true", "`||true` masks a failing command"),
    ("|| exit 0", "`|| exit 0` masks a failing command"),
    ("|| :", "`|| :` (no-op) masks a failing command"),
    ("|| echo", "`|| echo ...` swallows a failure into a message"),
    ("; true", "trailing `; true` forces a zero exit"),
    ("; exit 0", "trailing `; exit 0` forces a zero exit"),
]


def detect_build_gaming(dockerfile_text: str) -> list[str]:
    """Return human-readable descriptions of error-suppression / metric-gaming patterns in
    a generated Dockerfile's RUN steps (F2: agent optimizes for exit-0, not a working
    build — e.g. `pip install . || true`). Empty list = clean. Pure/offline-testable."""
    if not dockerfile_text:
        return []
    findings: list[str] = []
    for raw_line in dockerfile_text.splitlines():
        line = raw_line.strip()
        if not line.upper().startswith("RUN"):
            continue
        lowered = line.lower()
        for needle, desc in _SUPPRESSION_PATTERNS:
            if needle in lowered and desc not in findings:
                findings.append(desc)
    return findings


_DOCKERFILE_INSTRUCTIONS = {
    "FROM", "RUN", "CMD", "LABEL", "MAINTAINER", "EXPOSE", "ENV", "ADD", "COPY",
    "ENTRYPOINT", "VOLUME", "USER", "WORKDIR", "ARG", "ONBUILD", "STOPSIGNAL",
    "HEALTHCHECK", "SHELL",
}


# Only gcc/g++/cpp: the base ships build-essential (gcc/g++/make), so a version-pinned
# `gcc-N`/`g++-N` is redundant and, if the pin names a major Ubuntu 24.04 does not carry,
# also broken. Clang is deliberately NOT matched: the base ships no clang, and on Ubuntu
# 24.04 a modern `clang-N` (e.g. clang-21, only available via the upstream apt.llvm.org
# repo) is the ONLY way to get that compiler — stripping it would delete a required package.
_VERSIONED_TOOLCHAIN_RE = re.compile(r"\b(?:gcc|g\+\+|cpp)-\d+\b")


def strip_versioned_toolchain(dockerfile_text: str) -> tuple[str, list[str]]:
    """Deterministically remove version-pinned gcc/g++/cpp packages (`gcc-11`, `g++-11`, ...)
    from apt-get install lines. The base already ships `build-essential` (gcc/g++/make), so a
    pinned `gcc-N`/`g++-N` is redundant, and a major that Ubuntu 24.04 does not carry also
    fails with `Unable to locate package`. Clang is intentionally excluded (see regex note):
    it has no base fallback and modern versions come only from the upstream apt repo. A RUN
    whose apt install is left with no packages is dropped entirely. Returns
    (cleaned_text, removed_tokens). Pure."""
    if not dockerfile_text or not _VERSIONED_TOOLCHAIN_RE.search(dockerfile_text):
        return dockerfile_text, []
    removed: list[str] = []
    out_lines: list[str] = []
    for line in dockerfile_text.splitlines():
        if "apt-get install" not in line or not _VERSIONED_TOOLCHAIN_RE.search(line):
            out_lines.append(line)
            continue
        found = _VERSIONED_TOOLCHAIN_RE.findall(line)
        removed.extend(found)
        cleaned = _VERSIONED_TOOLCHAIN_RE.sub("", line)
        cleaned = re.sub(r"[ \t]{2,}", " ", cleaned).rstrip()
        # If the apt-get install now has no package operands, the RUN is a no-op that would
        # error (`apt-get install` with nothing) — drop the whole line. Heuristic: the part
        # after "install" (minus flags) has no bare package token left.
        after = cleaned.split("apt-get install", 1)[1] if "apt-get install" in cleaned else ""
        after_pkgs = re.sub(r"(^|\s)(-[-\w]+|&&.*|;.*|\|\|.*)", " ", after).strip()
        if not after_pkgs:
            # Drop the now-empty install entirely (keep a generic marker; do NOT echo the
            # stripped token names, so the dropped package never lingers in the Dockerfile).
            out_lines.append("# [auto] dropped a version-pinned compiler install; build-essential in base provides gcc/g++")
            continue
        out_lines.append(cleaned)
    return "\n".join(out_lines) + ("\n" if dockerfile_text.endswith("\n") else ""), removed


def validate_dockerfile_structure(text: str) -> list[str]:
    """Return structural problems in a Dockerfile (F5 pre-build gate). Catches the
    regression we observed — a repair producing a `dockerfile parse error on line 1`
    (prose/markdown leaked in, or FROM dropped). Joins line-continuations so a wrapped
    RUN body is not mis-read as an instruction. Empty list = well-formed. Pure/offline."""
    if not text or not text.strip():
        return ["empty Dockerfile"]
    problems: list[str] = []
    has_from = False
    continued = False  # previous logical line ended with a backslash
    for lineno, raw in enumerate(text.splitlines(), start=1):
        stripped = raw.strip()
        if continued:
            # This physical line is the continuation of a prior instruction's body.
            continued = raw.rstrip().endswith("\\")
            continue
        if not stripped or stripped.startswith("#"):
            continue
        token = stripped.split(None, 1)[0].upper()
        if token not in _DOCKERFILE_INSTRUCTIONS:
            problems.append(f"line {lineno} is not a valid Dockerfile instruction: {stripped[:60]!r}")
        elif token == "FROM":
            has_from = True
        continued = raw.rstrip().endswith("\\")
    if not has_from:
        problems.append("no FROM instruction")
    return problems


def detect_verify_gaming(verify_command: str) -> list[str]:
    """Return descriptions of a trivially-passing verify command (F2 on the verify side:
    `true`, `echo ok`, force `exit 0`, tacked-on `|| true`). A verify that does not run the
    built artifact is not a real check. Word-boundary matched to avoid flagging real
    commands like `construe` or paths with a colon. Pure/offline-testable."""
    cmd = (verify_command or "").strip()
    if not cmd:
        return []
    normalized = cmd.lower().strip().rstrip(";").strip()
    findings: list[str] = []
    # A verify that is *only* a trivial no-op token.
    if normalized in {"true", ":", "exit 0"}:
        return ["the verify command is a bare no-op (always passes)"]
    # An otherwise-real command with a success shim tacked on.
    for needle, desc in (
        ("|| true", "the verify command tacks on `|| true` (always passes)"),
        ("|| exit 0", "the verify command tacks on `|| exit 0` (always passes)"),
        ("; true", "the verify command tacks on `; true` (always passes)"),
        ("; exit 0", "the verify command tacks on `; exit 0` (always passes)"),
    ):
        if needle in normalized:
            findings.append(desc)
    # A verify whose only real token is an echo (does not exercise the artifact).
    stripped = re.sub(r"^(export [^;]+;\s*)+", "", normalized).strip()
    if re.match(r"^echo\b", stripped) and "&&" not in stripped and "|" not in stripped:
        findings.append("the verify command only `echo`s — it does not exercise the artifact")
    return findings


# Fixed taxonomy the LLM diagnosis must choose from, so its category flows into the same
# strategy ledger / targeted-hint machinery as the regex classifier. Superset of the regex
# categories plus the two the regex misses most (build_config_error, verify_command_wrong).
DIAGNOSIS_CATEGORIES = [
    "missing_dependency", "missing_command", "jdk_version_mismatch", "python_missing",
    "python_build_tool_stale", "shell_missing", "permission_error", "network_tls",
    "network_resolution", "submodule_missing", "build_config_error",
    "wrong_build_invocation", "verify_command_wrong", "timeout", "other",
]

DIAGNOSIS_SYSTEM_PROMPT = (
    "You are a build-failure triage expert. Given a failing Docker build (or verify) log, "
    "classify the ROOT cause into exactly one category and explain it in one line. Do not "
    "propose a full fix; name the failure class and the single underlying cause. Distinguish "
    "a BUILD failure (the artifact did not compile/install) from a VERIFY failure (the build "
    "succeeded but the check command is wrong, e.g. wrong binary name/path -> "
    "verify_command_wrong). Reply with ONLY a JSON object."
)


def build_diagnosis_prompt(log: str, phase: str, exit_code: int, dockerfile_text: str = "", max_log_chars: int = 4000) -> str:
    """Pure prompt builder for LLM failure diagnosis. Offline-testable."""
    tail = (log or "").strip()[-max_log_chars:]
    categories = ", ".join(DIAGNOSIS_CATEGORIES)
    return (
        f"Phase: {phase}\nExit code: {exit_code}\n\n"
        f"Allowed categories (choose exactly one): {categories}\n\n"
        "Dockerfile being built:\n-----\n"
        f"{dockerfile_text.strip() or '(not provided)'}\n-----\n\n"
        "Tail of the failing log:\n-----\n"
        f"{tail or '(empty)'}\n-----\n\n"
        'Reply with ONLY: {"category": "<one of the allowed>", "root_cause": '
        '"<one sentence>", "confidence": <0..1>}'
    )


def _safe_json_object(text: str) -> dict | None:
    """Tolerantly extract the first JSON object from a model response (strips code fences).
    Returns None on any failure. Mirrors mid_verify.parse_mid_verdict's extraction."""
    raw = (text or "").strip()
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw, re.DOTALL)
    candidate = fenced.group(1) if fenced else raw
    match = re.search(r"\{.*\}", candidate, re.DOTALL)
    if not match:
        return None
    try:
        parsed = json.loads(match.group(0))
    except (ValueError, TypeError):
        return None
    return parsed if isinstance(parsed, dict) else None


def parse_diagnosis(raw: str) -> dict | None:
    """Parse the diagnosis JSON. Returns None (inconclusive) on any failure or an
    out-of-taxonomy category, so a bad judge response never overrides the regex hint.
    Pure/offline-testable."""
    obj = _safe_json_object(raw)
    if not isinstance(obj, dict):
        return None
    category = obj.get("category")
    if not isinstance(category, str) or category not in DIAGNOSIS_CATEGORIES:
        return None
    root_cause = obj.get("root_cause")
    conf = obj.get("confidence")
    return {
        "category": category,
        "root_cause": root_cause if isinstance(root_cause, str) else "",
        "confidence": float(conf) if isinstance(conf, (int, float)) else None,
    }


async def refine_failure_hints_with_llm(
    hints: dict,
    *,
    log: str,
    dockerfile_content: str,
    llm_metrics: dict,
    repo_url: str,
) -> dict:
    """Upgrade a regex `unknown`/low-confidence classification with an LLM diagnosis so the
    strategy ledger and targeted hints get a real failure class (closes the regex blind spot
    that left cyclang-style exit-127 failures as `unknown`). Gated to the ReAct repair path
    and only invoked when the regex is uninformative, to bound cost. On any LLM/parse
    failure it returns the original hints unchanged — the regex result is never lost."""
    if not args.react_repair or not args.repair_reasoning:
        return hints
    if not isinstance(hints, dict):
        return hints
    if hints.get("category") not in (None, "", "unknown"):
        return hints  # regex already produced a confident class; don't pay for an LLM call.
    try:
        response = await chat_completion_with_retries(
            client=client,
            model=args.model,
            temperature=EFFECTIVE_TEMPERATURE,
            messages=[
                {"role": "system", "content": DIAGNOSIS_SYSTEM_PROMPT},
                {"role": "user", "content": build_diagnosis_prompt(
                    log, hints.get("phase", "build"), hints.get("exit_code", 0), dockerfile_content,
                )},
            ],
            repo_url=repo_url,
            phase="diagnose",
            metrics=llm_metrics,
            timeout_seconds=args.repair_timeout,
            max_retries=args.llm_max_retries,
            retry_backoff_seconds=args.llm_retry_backoff_seconds,
            max_tokens=300,
        )
        raw = response.choices[0].message.content or ""
    except Exception as exc:  # noqa: BLE001 — diagnosis is best-effort; never fail the repair
        log_warn(f"[diagnose {repo_url}] LLM diagnosis failed, keeping regex hint: {exc}")
        return hints
    diagnosis = parse_diagnosis(raw)
    if not diagnosis:
        return hints
    refined = dict(hints)
    refined["category"] = diagnosis["category"]
    refined["confidence"] = "llm"
    if diagnosis.get("root_cause"):
        refined["root_cause"] = diagnosis["root_cause"]
    refined["diagnosed_by"] = "llm"
    log_info(f"[diagnose {repo_url}] llm refined `unknown` -> `{diagnosis['category']}`")
    return refined


def render_failure_hints_for_prompt(failure_hints: dict | None) -> str:
    if not failure_hints:
        return ""
    return json.dumps(failure_hints, indent=2, sort_keys=True)


_UNAVAILABLE_PKG_RE = re.compile(r"unable to locate package\s+(\S+)", re.IGNORECASE)


def _parse_unavailable_packages(log: str) -> list[str]:
    """Pull the package names apt reported as `Unable to locate package <pkg>` from a build
    log. These are almost always version-pinned names Ubuntu 24.04 does not carry (e.g.
    a too-new openjdk-N or llvm-N) that the agent copied from project docs."""
    seen: set[str] = set()
    out: list[str] = []
    for match in _UNAVAILABLE_PKG_RE.finditer(log or ""):
        pkg = match.group(1).strip().strip("'\".,()")
        if pkg and pkg not in seen:
            seen.add(pkg)
            out.append(pkg)
    return out[:5]


def _apt_search_keyword(pkg: str) -> str:
    """Reduce a version-pinned package name to the family keyword an apt-cache search will
    match: openjdk-17-jdk -> openjdk, llvm-15 -> llvm, python3.11 -> python. The exact
    pinned name returns nothing; the family stem surfaces the candidates that do exist."""
    stem = re.sub(r"[-.]?\d.*$", "", pkg).strip("-.")
    return stem or pkg


def resolve_unavailable_apt_packages(log: str, container_cli: str, base_image: str) -> dict:
    """For each `Unable to locate package` in the log, query the base image's apt repos and
    return {unavailable_name: candidate_listing}. This runs the lookup EAGERLY and
    deterministically: the repair LLM under-invoked the apt_search tool on its own, so the
    answer is computed here and handed to it rather than left to its discretion."""
    resolved: dict[str, str] = {}
    for pkg in _parse_unavailable_packages(log):
        resolved[pkg] = apt_search_packages(container_cli, base_image, _apt_search_keyword(pkg))
    return resolved


def _has_category(failure_hints: dict | None, category: str) -> bool:
    """True if `category` appears in a failure-hints payload, whether flat (a single phase
    dict with a top-level "category") or nested ({"build": {...}, "verify": {...}}). The
    repair call sites pass the nested form, so a flat-only `.get("category")` check silently
    never fires — this handles both."""
    if not isinstance(failure_hints, dict):
        return False
    if failure_hints.get("category") == category:
        return True
    return any(
        isinstance(value, dict) and value.get("category") == category
        for value in failure_hints.values()
    )


# A build/configure/pkg-config step that needs a system library reports it many ways; the
# apt package that provides the headers is almost always lib<name>-dev. Capture the library
# name so we can search the base for the matching -dev package.
_MISSING_SYSTEM_LIB_RES = [
    re.compile(r"system library [`'\"]?([\w.+-]+)[`'\"]?[^\n]*?not found", re.IGNORECASE),
    re.compile(r"could not find (?:native )?(?:static |shared )?library [`'\"]?([\w.+-]+)", re.IGNORECASE),
    re.compile(r"no package '([\w.+-]+)' found", re.IGNORECASE),
    re.compile(r"package '?([\w.+-]+)'? was not found in the pkg-config", re.IGNORECASE),
]


def _parse_missing_system_libs(log: str) -> list[str]:
    """Pull system-library names from pkg-config / cargo / configure 'not found' errors
    (e.g. webkit2gtk-4.1, Polly, libpsl). These are NOT apt package names — the fix is the
    matching -dev package, which the base-apt search surfaces."""
    seen: set[str] = set()
    out: list[str] = []
    for rx in _MISSING_SYSTEM_LIB_RES:
        for match in rx.finditer(log or ""):
            lib = match.group(1).strip().strip("'\".,()")
            if lib and lib.lower() not in seen:
                seen.add(lib.lower())
                out.append(lib)
    return out[:5]


def _lib_search_keyword(lib: str) -> str:
    """Search keyword for a system-library name. Unlike _apt_search_keyword (which strips at
    the FIRST digit — right for openjdk-17-jdk -> openjdk), library names carry the version
    as a TRAILING token and embed digits mid-name, so only the trailing version is dropped:
    webkit2gtk-4.1 -> webkit2gtk (hits libwebkit2gtk-4.1-dev precisely, not every webkit*),
    gtk+-3.0 -> gtk+, Polly -> polly. Stripping mid-name digits would over-broaden the
    search and let the agent pick the wrong sibling package."""
    stem = re.sub(r"[-.]?\d[\d.]*$", "", lib).strip("-.")
    return (stem or lib).lower()


def resolve_missing_system_libs(log: str, container_cli: str, base_image: str) -> dict:
    """For each missing system library in the log, search the base apt repos for the family
    keyword and return {lib: candidate_listing} so the repair prompt can name the real
    lib<...>-dev package instead of leaving the agent to guess. Eager + deterministic, same
    rationale as resolve_unavailable_apt_packages."""
    resolved: dict[str, str] = {}
    for lib in _parse_missing_system_libs(log):
        resolved[lib] = apt_search_packages(container_cli, base_image, _lib_search_keyword(lib))
    return resolved


def _find_candidate_map(failure_hints: dict | None, key: str) -> dict:
    """Locate a candidate mapping stored under `key` in a failure-hints payload, whether the
    payload is a single phase dict or the nested {"build": {...}, "verify": {...}} form used
    at the repair call site."""
    if not isinstance(failure_hints, dict):
        return {}
    if isinstance(failure_hints.get(key), dict):
        return failure_hints[key]
    for value in failure_hints.values():
        if isinstance(value, dict) and isinstance(value.get(key), dict):
            return value[key]
    return {}


def find_apt_candidates(failure_hints: dict | None) -> dict:
    """Unavailable apt-package candidates (Unable-to-locate-package resolution)."""
    return _find_candidate_map(failure_hints, "apt_candidates")


def find_dev_lib_candidates(failure_hints: dict | None) -> dict:
    """Missing system-library -> base-apt candidates (pkg-config/cargo 'not found')."""
    return _find_candidate_map(failure_hints, "dev_lib_candidates")


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


def _prior_attempt_categories(repair_history: list[dict] | None) -> list[tuple[int, str, bool]]:
    """Per-attempt (attempt_number, failure_category, changed_dockerfile) from history,
    oldest→newest. Used to build the strategy ledger so the repair loop can tell whether
    the current failure is a REPEAT of a class already tried (F1: version-chasing /
    retrying variations of a doomed approach)."""
    out: list[tuple[int, str, bool]] = []
    for entry in repair_history or []:
        if not isinstance(entry, dict):
            continue
        cats = _extract_failure_categories(entry.get("failure_hints"))
        category = cats[0] if cats else "unknown"
        changed = bool(entry.get("repair_result", {}).get("changed_dockerfile"))
        out.append((int(entry.get("attempt", 0)), category, changed))
    return out


def render_build_verify_separation(verify_only_failure: bool) -> str:
    """F6: when the build succeeded and only verification failed, instruct the Dockerfile
    rewrite to PRESERVE the working build steps and change only the artifact/verify surface.
    Empty string when it is a build failure. Pure/offline-testable."""
    if not verify_only_failure:
        return ""
    return (
        "\n\nBUILD/VERIFY SEPARATION — the Docker BUILD ALREADY SUCCEEDED (the image"
        " built). This is a VERIFY-ONLY failure: the artifact was produced but the"
        " verification did not pass. PRESERVE the working build steps VERBATIM — do NOT"
        " rewrite, reorder, or drop them. Change only what makes the built artifact"
        " reachable/usable by the verification (e.g. PATH, install location, the binary"
        " name/target the verify command expects, a runtime dependency). Make the"
        " smallest possible diff to the build steps — ideally none.\n"
    )


def render_strategy_ledger(repair_history: list[dict] | None, failure_hints: dict | None) -> str:
    """Force diagnose-then-act. Renders (a) a ledger of the failure classes already tried
    and (b) a hard directive when the CURRENT failure repeats the immediately-prior class —
    forbidding a same-class variant retry and requiring a strategy-class escalation. Returns
    "" when there is no prior attempt (nothing to diagnose against yet). Pure/offline-testable."""
    prior = _prior_attempt_categories(repair_history)
    if not prior:
        return ""

    current_cats = _extract_failure_categories(failure_hints)
    current = current_cats[0] if current_cats else "unknown"
    last_attempt, last_category, last_changed = prior[-1]

    lines = ["\nSTRATEGY LEDGER (failure classes already attempted, oldest→newest):"]
    for attempt_no, category, changed in prior:
        edit = "edited Dockerfile" if changed else "no Dockerfile change"
        lines.append(f"- attempt {attempt_no}: class `{category}` ({edit}) — still failing")
    lines.append(f"- current failure class: `{current}`")

    repeat = current != "unknown" and current == last_category
    if repeat:
        lines.append(
            f"\nREPEAT DETECTED: class `{current}` was already attempted and still fails. A"
            " same-class variant retry is NOT allowed (e.g. another package version, another"
            " flag toggle, re-pinning a version). You MUST escalate to a DIFFERENT strategy"
            " class — change the dependency source, the build invocation, or the environment"
            " assumption, not a parameter of the failed approach."
        )

    lines.append(
        "\nDIAGNOSE-THEN-ACT: before writing the Dockerfile, state your diagnosis in a think"
        " step — (1) the failure class, (2) the root cause in one line, (3) why the previous"
        " attempt's fix did not resolve it, (4) the strategy you will now apply and how it"
        " differs from every class in the ledger above. Only then output the Dockerfile."
    )
    return "\n".join(lines) + "\n"


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
    # Surface a trivially-passing verify (F2): it will pass regardless of whether the
    # artifact works, so a green SOFT verify on it is meaningless. MID verify judges this
    # post-hoc; the warning makes it visible in the run log too.
    verify_gaming = detect_verify_gaming(sanitized_command)
    if verify_gaming:
        log_warn(f"[verify-lint] trivially-passing verify command: {'; '.join(verify_gaming)}")
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
    # F5 minimal-diff / regression guard: never overwrite a well-formed Dockerfile with a
    # malformed one. A repair that introduces a structural break (e.g. prose leaking in ->
    # "parse error on line 1", or a dropped FROM) is a strict regression; reject it, keep the
    # current Dockerfile, and let the next attempt try again rather than building garbage.
    repaired_problems = validate_dockerfile_structure(repaired) if args.repair_reasoning else []
    if repaired_problems and not validate_dockerfile_structure(current):
        log_warn(
            f"Rejected a malformed repair for {repo_url} (would regress a well-formed "
            f"Dockerfile): {'; '.join(repaired_problems[:3])}. Keeping the current Dockerfile."
        )
        write_text(report_dir / f"attempt-{attempt}.rejected-malformed.Dockerfile", repaired)
        return current, False
    # Deterministically strip version-pinned gcc-N/g++-N (redundant — build-essential in base
    # already provides the compiler; a major Ubuntu 24.04 lacks would also fail to install) —
    # the agent re-adds them from repo CI despite the prompt (F4). Clang is preserved (no base
    # fallback; modern clang comes only from the upstream apt repo). Structural, not advice.
    repaired, removed_toolchain = strip_versioned_toolchain(repaired)
    if removed_toolchain:
        log_info(f"[toolchain-strip {repo_name}] removed version-pinned {sorted(set(removed_toolchain))} (build-essential covers it)")
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

    async def _stream_until_exit() -> int:
        while True:
            line = await process.stdout.readline()
            if not line:
                break
            decoded_line = line.decode("utf-8", errors="replace")
            output_chunks.append(decoded_line)
            log_info(f"[build {repo_name} attempt {attempt}] {decoded_line.rstrip()}")
        return await process.wait()

    try:
        # A hung build (e.g. a package manager stalling on a registry behind a proxy)
        # produces no output, so the readline loop would block forever. Cap the whole
        # build on a wall clock; on timeout, kill it and report a failed attempt (124)
        # so the repair loop and the overall run keep moving.
        returncode = await asyncio.wait_for(_stream_until_exit(), timeout=args.build_timeout)
    except asyncio.TimeoutError:
        log_warn(f"[build {repo_name} attempt {attempt}] timed out after {args.build_timeout}s; killing build")
        try:
            process.kill()
        except ProcessLookupError:
            pass
        await process.wait()
        output_chunks.append(f"\n[build timed out after {args.build_timeout}s and was killed]\n")
        return 124, "".join(output_chunks)
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


async def get_image_size_bytes(image_tag: str) -> int | None:
    """Total size of the built image in bytes (docker image inspect .Size). None on error."""
    command = [
        args.container_cli,
        "image",
        "inspect",
        "--format",
        "{{.Size}}",
        image_tag,
    ]
    process = await asyncio.create_subprocess_exec(
        *command,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
    )
    output, _ = await process.communicate()
    if process.returncode != 0:
        return None
    try:
        return int(output.decode("utf-8", errors="replace").strip())
    except ValueError:
        return None


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


async def measure_binary_in_image(
    image_tag: str,
    artifact_path: str,
    workdir: str,
    gt_size_bytes: Optional[int],
    gt_digest: Optional[str],
) -> dict:
    """Measure size and xxh64 hash of a produced binary inside a Docker image."""
    if artifact_path.startswith("/"):
        abs_path = artifact_path
    else:
        base = workdir.rstrip("/") if workdir else "/home/manualrepos/repo"
        abs_path = f"{base}/{artifact_path.lstrip('./')}"

    result: dict = {
        "gt_binary_path": artifact_path,
        "gt_binary_size_bytes": gt_size_bytes,
        "gt_binary_hash": gt_digest,
        "measured_size_bytes": None,
        "binary_size_plausible": None,
        "binary_hash_match": None,
    }

    # Measure file size via stat
    size_cmd = [
        args.container_cli, "run", "--rm",
        "--entrypoint", "/bin/sh",
        image_tag, "-c", f"stat -c '%s' {shlex.quote(abs_path)}",
    ]
    try:
        proc = await asyncio.create_subprocess_exec(
            *size_cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=15)
        if proc.returncode == 0:
            result["measured_size_bytes"] = int(stdout.decode("utf-8", errors="replace").strip())
    except Exception:
        pass

    if result["measured_size_bytes"] is not None and gt_size_bytes and gt_size_bytes > 0:
        ratio = result["measured_size_bytes"] / gt_size_bytes
        result["binary_size_plausible"] = 0.5 <= ratio <= 2.0

    # Measure the xxh64 hash on the HOST: the slim image has no xxhsum, so stream the file
    # out with `cat` and hash it locally. GT digests are stored as `xxh64:<hex>`.
    if gt_digest and gt_digest.startswith("xxh64:"):
        cat_cmd = [
            args.container_cli, "run", "--rm",
            "--entrypoint", "/bin/sh",
            image_tag, "-c", f"cat {shlex.quote(abs_path)}",
        ]
        try:
            proc = await asyncio.create_subprocess_exec(
                *cat_cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.DEVNULL,
            )
            stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=30)
            if proc.returncode == 0 and stdout:
                measured_hex = xxhash.xxh64(stdout).hexdigest().lower()
                result["binary_hash_match"] = (f"xxh64:{measured_hex}" == gt_digest.lower())
        except Exception:
            pass

    return result


async def gather_artifact_listing(image_tag: str, workdir: str) -> Optional[dict]:
    """Collect the raw inputs HARD verification needs from the built image: the
    git-tracked source list and an xxh64sum of every file under the workdir. The
    evaluator (eval.py) turns these into the produced-artifact combined hash and
    compares it to ground truth, so the agent submodule stays free of dataset logic."""
    wd = workdir or "/home/manualrepos/repo"

    async def _run(shell: str, timeout: int) -> Optional[str]:
        cmd = [args.container_cli, "run", "--rm", "--entrypoint", "/bin/sh", image_tag, "-c", shell]
        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.DEVNULL
            )
            stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=timeout)
            if proc.returncode == 0:
                return stdout.decode("utf-8", errors="replace")
        except Exception:
            return None
        return None

    wd_q = shlex.quote(wd)
    git_files = await _run(f"cd {wd_q} && git ls-files", 60)
    # %P drops the leading "./"; xargs -0 keeps it fast on repos with thousands of files.
    all_hashes = await _run(
        f"cd {wd_q} && find . -type f -printf '%P\\0' | xargs -0 xxh64sum", 180
    )
    if git_files is None and all_hashes is None:
        return None
    return {"git_files": git_files or "", "all_hashes": all_hashes or ""}


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
    repo_path: "Path | None" = None,
    report_dir: "Path | None" = None,
    verify_only_failure: bool = False,
) -> str:
    # De-noise the inputs before they reach the model. The auto-injected corporate-CA
    # bootstrap (a multi-kilobyte base64 blob) is stripped from the Dockerfile — it is
    # re-added at build time by inject_ca_cert_into_dockerfile, so the model must never
    # reproduce it (forcing it to was driving non-terminating generations that hung
    # until the wall-clock repair timeout). The build log is stripped of echoed base64
    # blobs and carriage-return progress-bar spam for the same reason.
    dockerfile_content = strip_ca_cert_from_dockerfile(dockerfile_content)
    build_log = sanitize_build_log_for_prompt(build_log)

    # The contract for both repair mechanisms is "return the COMPLETE Dockerfile, base
    # template preserved verbatim". A model that returns only the AGENT_BUILD_STEPS body
    # (dropping the base → no FROM) is self-healed before the repaired file is written,
    # so a malformed repair can never overwrite a valid Dockerfile with a baseless one.
    base_template = get_base_template(
        classification,
        Path(__file__).resolve().parents[3] / "templates",
        log_warn=log_warn,
        log_error=log_error,
    )

    # Baseline repair: a single LLM call over just the current Dockerfile and the
    # build log. This is the flat_baseline mechanism; the ReAct agent below is a
    # gated architecture component.
    if not args.react_repair:
        simple_prompt = (
            SIMPLE_PROMPT_TEMPLATE.replace("{{REPO_URL}}", repo_url)
            .replace("{{ATTEMPT_NUMBER}}", str(attempt_number))
            .replace("{{DOCKERFILE_CONTENT}}", dockerfile_content)
            .replace("{{BUILD_LOG}}", trim_log(build_log))
        )
        response = await chat_completion_with_retries(
            client=client,
            model=args.model,
            temperature=EFFECTIVE_TEMPERATURE,
            messages=[{"role": "user", "content": simple_prompt}],
            repo_url=repo_url,
            phase="repair",
            metrics=llm_metrics,
            timeout_seconds=args.repair_timeout,
            max_retries=args.llm_max_retries,
            retry_backoff_seconds=args.llm_retry_backoff_seconds,
            max_tokens=args.repair_max_output_tokens,
        )
        raw = response.choices[0].message.content or ""
        return ensure_base_template(extract_dockerfile(raw.strip()), base_template, log_warn=log_warn)

    prompt = (
        PROMPT_TEMPLATE.replace("{{REPO_URL}}", repo_url)
        .replace("{{ATTEMPT_NUMBER}}", str(attempt_number))
        .replace("{{BASE_TEMPLATE_CONTENT}}", base_template)
        .replace("{{CLASSIFICATION_RESULT}}", render_yaml(classification))
        .replace("{{SUMMARY_CONTENT}}", summary)
        .replace("{{DOCKERFILE_CONTENT}}", dockerfile_content)
        .replace("{{BUILD_LOG}}", trim_log(build_log))
    )
    # Diagnose-then-act: the strategy ledger forces the model to reason about the failure
    # class BEFORE editing and forbids retrying a class that already failed (F1). Injected
    # ahead of the raw hints so the "escalate, don't repeat" directive frames them.
    # The agent-reasoning repair upgrades (F1/F2/F6) are gated together for the A/B; the
    # environment levers remain unconditional.
    if args.repair_reasoning:
        # Build/verify separation (F6): if the build already succeeded and only verification
        # failed, the working build steps must be preserved — the problem is the artifact's
        # usability or the check, not the build. Frame this FIRST so it governs every edit.
        prompt += render_build_verify_separation(verify_only_failure)

        strategy_ledger = render_strategy_ledger(repair_history, failure_hints)
        if strategy_ledger:
            prompt += "\n" + strategy_ledger

        # Anti-gaming lint (F2): if the Dockerfile being repaired suppresses build errors
        # (`|| true`, `; exit 0`, ...), a "successful" build is a masked failure. Call it out
        # explicitly and forbid carrying it forward — the model tends to keep such shims.
        gaming = detect_build_gaming(dockerfile_content)
        if gaming:
            prompt += (
                "\nANTI-GAMING: the current Dockerfile suppresses build errors — "
                + "; ".join(gaming)
                + ". This makes the build exit 0 without actually succeeding, which is NOT a"
                " real fix. REMOVE every error-suppression (`|| true`, `|| exit 0`, `; true`,"
                " trailing `; exit 0`) and make the underlying command actually succeed.\n"
            )
    if failure_hints:
        prompt += (
            "\n\nUse these normalized failure hints as primary guidance for your fix."
            "\nPrefer addressing high-confidence categories first."
            f"\n\nFAILURE_HINTS_JSON:\n{render_failure_hints_for_prompt(failure_hints)}\n"
        )
        if _has_category(failure_hints, "python_missing"):
            prompt += (
                "\nTargeted remediation hint: build tooling expects `python` on PATH."
                "\nPrefer adding `python-is-python3` (Debian/Ubuntu) or a safe equivalent symlink in the image,"
                "\nrather than editing project source or skipping required build steps.\n"
            )
        if _has_category(failure_hints, "jdk_version_mismatch"):
            prompt += (
                "\nTargeted remediation hint (JDK selection, not a missing package): the base"
                " image ships a JDK (the unversioned `default-jdk`, openjdk-21 on Ubuntu 24.04)"
                " on PATH with JAVA_HOME set. If the build's Gradle/Maven toolchain rejects the"
                " pre-installed JDK, prefer NOT apt-installing a versioned JDK (Ubuntu 24.04 may"
                " not carry the requested major). Instead:"
                " (a) let Gradle's toolchain auto-download fetch the exact JDK (enable toolchain"
                " download repositories), or (b) disable toolchain enforcement and point the build"
                " at the installed JDK (`-Porg.gradle.java.installations.paths=$JAVA_HOME`), or"
                " (c) relax/remove the `languageVersion` pin so the build uses $JAVA_HOME.\n"
            )
        if _has_category(failure_hints, "python_build_tool_stale"):
            prompt += (
                "\nTargeted remediation hint (stale Python build tool, not a missing package):"
                " the distro's meson/cython is older than the project requires. The base image"
                " ships an activated virtualenv on PATH, so upgrade the build front-end in it"
                " BEFORE building — `RUN pip install -U meson meson-python cython ninja` (keep"
                " ALL of these; cython is required for meson's cython compiler check). Then build"
                " with build isolation OFF so pip uses the venv's upgraded tools instead of"
                " re-fetching the stale pinned versions into an isolated env:"
                " `RUN pip install --no-build-isolation .` (isolation re-installs the project's"
                " pinned build-requires and reintroduces the stale meson/cython — the usual cause"
                " of `metadata-generation-failed` after an upgrade). Do not apt-install meson or"
                " edit project source.\n"
            )
        apt_candidates = find_apt_candidates(failure_hints)
        if apt_candidates:
            listing = "\n".join(
                f"- `{pkg}` is NOT available on this base. Candidates that exist:\n{candidates}"
                for pkg, candidates in apt_candidates.items()
            )
            prompt += (
                "\nDETERMINISTIC APT RESOLUTION (already queried against this build's base"
                " image — do NOT re-guess version-pinned names from project docs). Replace"
                " each unavailable package below with a real candidate. For a JDK, the base"
                " already ships the unversioned `default-jdk` — drop the JDK install entirely"
                " rather than substituting another version-pinned openjdk-N:\n"
                f"{listing}\n"
            )
        dev_lib_candidates = find_dev_lib_candidates(failure_hints)
        if dev_lib_candidates:
            dev_listing = "\n".join(
                f"- system library `{lib}` is missing — install the matching dev package."
                f" Base apt candidates:\n{candidates}"
                for lib, candidates in dev_lib_candidates.items()
            )
            prompt += (
                "\nDETERMINISTIC DEV-LIBRARY RESOLUTION (a build/pkg-config/cargo step needs a"
                " system library; the base apt repos were queried). Install the matching"
                " development package — typically `lib<name>-dev` — from the candidates below"
                " (NOT the bare library name, which is not an apt package):\n"
                f"{dev_listing}\n"
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

    prompt = clamp_summary_in_prompt(
        prompt, summary, args.max_input_tokens, model=args.model, phase=f"repair attempt {attempt_number}"
    )

    repo_tools = []
    apt_tool_added = False
    # apt_search is ENVIRONMENT access (what packages the base distro actually ships),
    # not repo inspection — so it is always available, independent of --repair-repo-tools
    # (which gates repo *file* access for the AB-26/27 study). Without it, L3 is blind to
    # the distro and blindly guesses versions on `Unable to locate package` (e.g. a repo
    # asks for a too-new openjdk-N; the answer is default-jdk). Tied to the Dockerfile's own FROM
    # so the query matches the base the build uses.
    base_image = extract_base_image(dockerfile_content)
    if base_image:
        repo_tools.append(build_apt_search_tool(args.container_cli, base_image))
        apt_tool_added = True
    if args.repair_repo_tools and repo_path is not None:
        repo_tools.extend([
            build_read_file_tool(repo_path),
            build_list_tree_tool(repo_path),
            build_search_pattern_tool(repo_path),
        ])
    repo_tools = repo_tools or None

    repaired, trace_steps, l3_trace = await run_l3_dockerfile_repair_react(
        repo_url=repo_url,
        attempt_number=attempt_number,
        prompt=prompt,
        repair_timeout=args.repair_timeout,
        l3_react_max_steps=args.l3_react_max_steps,
        runtime=_make_repair_runtime(),
        build_snippet_tool=build_get_dockerfile_snippet_tool if args.snippet_tools else None,
        repo_tools=repo_tools,
    )

    # Persist the repair agent's ReAct trace (tool calls + reasoning) so the trace
    # viewer can show which tools — including the optional repo tools — were used.
    if report_dir is not None:
        tools_available = ["think", "run_hadolint_on_snippet"]
        if args.snippet_tools:
            tools_available.append("get_dockerfile_snippet")
        if apt_tool_added:
            tools_available.append("apt_search")
        if args.repair_repo_tools and repo_path is not None:
            tools_available.extend(["read_file", "list_tree", "search_pattern"])
        tool_calls_made = [
            tc.get("name")
            for event in l3_trace
            for tc in (event.get("tool_calls") or [])
            if tc.get("name")
        ]
        try:
            write_text(
                report_dir / f"attempt-{attempt_number}.l3-trace.yaml",
                render_yaml(
                    {
                        "repo": repo_url,
                        "attempt": attempt_number,
                        "max_steps": int(args.l3_react_max_steps),
                        "trace_steps": trace_steps,
                        "repo_tools_enabled": bool(repo_tools),
                        "snippet_tools_enabled": bool(args.snippet_tools),
                        "tools_available": tools_available,
                        "tool_calls_made": tool_calls_made,
                        "trace": l3_trace,
                    }
                ),
            )
        except Exception as exc:  # tracing must never break a repair attempt
            log_warn(f"Failed to write L3 repair trace for {repo_url} attempt {attempt_number}: {exc}")

    if not repaired:
        log_warn(f"L3 ReAct repair agent returned empty content for {repo_url} attempt {attempt_number}")
        return ""

    llm_metrics.setdefault("l3_react", {}).update(
        {
            "enabled": True,
            "max_steps": int(args.l3_react_max_steps),
            "last_attempt": attempt_number,
            "trace_steps": trace_steps,
            "repo_tools_enabled": bool(repo_tools),
        }
    )
    return ensure_base_template(repaired, base_template, log_warn=log_warn)


# Transient LLM/transport failures that should burn a single repair attempt and let
# the loop retry on the next one, rather than escaping the whole per-repo attempt loop
# (a flaky endpoint must not cost a repo all of its remaining repair budget).
_TRANSIENT_REPAIR_ERRORS = (
    APITimeoutError,
    APIError,
    httpx.HTTPError,
    ssl.SSLError,
    asyncio.TimeoutError,
)


async def _safe_request_repair(repo_url: str, attempt: int, repair_coro):
    """Await a request_repair() coroutine, swallowing transient errors so one failed
    generation does not abort the remaining attempts. Returns None on transient failure;
    callers should skip applying a repair and continue to the next attempt."""
    try:
        return await repair_coro
    except _TRANSIENT_REPAIR_ERRORS as error:
        log_warn(
            f"Repair generation failed for {repo_url} attempt {attempt} "
            f"({type(error).__name__}: {error}); retrying on next attempt"
        )
        return None


# Keys the model may wrap its single command in; mirrors the ReAct verify path so
# both derivation routes survive a ```yaml-fenced reply (otherwise the fence leaks
# verbatim into the verify command -> "Syntax error: EOF in backquote substitution").
_VERIFY_COMMAND_KEYS = [
    "verification_command",
    "verify_command",
    "repaired_verification_command",
    "refreshed_verification_command",
    "command",
]


async def _simple_verify_command_call(
    repo_url: str, system_prompt: str, prompt: str, llm_metrics: dict, phase: str
) -> str:
    """Baseline verify-command derivation: a single LLM call (no ReAct agent).
    The prompts instruct the model to return exactly one shell command, but models
    routinely answer in a ```yaml fenced block; extract the command field instead of
    trusting the raw reply."""
    response = await chat_completion_with_retries(
        client=client,
        model=args.model,
        temperature=EFFECTIVE_TEMPERATURE,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ],
        repo_url=repo_url,
        phase=phase,
        metrics=llm_metrics,
        timeout_seconds=args.verify_repair_timeout,
        max_retries=args.llm_max_retries,
        retry_backoff_seconds=args.llm_retry_backoff_seconds,
        max_tokens=args.repair_max_output_tokens,
    )
    return extract_command_from_reply(
        response.choices[0].message.content or "", _VERIFY_COMMAND_KEYS
    )


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

    if not args.react_repair:
        return await _simple_verify_command_call(
            repo_url, VERIFY_REPAIR_SYSTEM_PROMPT, prompt, llm_metrics, "verify-repair"
        )

    command, trace_steps = await run_l3_verification_command_react(
        repo_url=repo_url,
        prompt=prompt,
        verify_timeout=args.verify_repair_timeout,
        l3_react_max_steps=args.l3_react_max_steps,
        thread_suffix="l3-verify-repair",
        system_prompt=VERIFY_REPAIR_SYSTEM_PROMPT,
        candidate_keys=["verification_command", "verify_command", "repaired_verification_command", "command"],
        runtime=_make_repair_runtime(),
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

    if not args.react_repair:
        return await _simple_verify_command_call(
            repo_url, VERIFY_REFRESH_SYSTEM_PROMPT, prompt, llm_metrics, "verify-refresh"
        )

    command, trace_steps = await run_l3_verification_command_react(
        repo_url=repo_url,
        prompt=prompt,
        verify_timeout=args.verify_repair_timeout,
        l3_react_max_steps=args.l3_react_max_steps,
        thread_suffix="l3-verify-refresh",
        system_prompt=VERIFY_REFRESH_SYSTEM_PROMPT,
        candidate_keys=["verification_command", "verify_command", "refreshed_verification_command", "command"],
        runtime=_make_repair_runtime(),
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


def _init_repair_report(repo_url: str, dockerfile_path: Path) -> dict:
    """Build the initial repair-report skeleton: target, attempt budget, and a
    snapshot of the stateful-repair config. Attempts/success are filled in by the
    repair loop. Pure (reads the parsed args + active prompt profile)."""
    return {
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

        report: dict = _init_repair_report(repo_url, dockerfile_path)

        try:
            classification_path = results_dir / f"{repo_name}.yaml"
            # Classification is optional: building + verifying needs only the Dockerfile.
            # one_shot_direct skips Stage 1, so there is no classification — proceed with an
            # empty one (it only feeds the repair PROMPT, and one_shot runs a single attempt
            # with no repair). Other variants do produce it.
            classification = read_yaml_file(classification_path) or {}
            if not classification:
                log_warn(
                    f"No classification for {repo_url} at {classification_path}; proceeding with build+verify only (expected for one_shot_direct)."
                )

            if not dockerfile_path.exists():
                log_warn(
                    f"Skipping {repo_url}: Dockerfile missing at {dockerfile_path}. Run agent_dockerfile.py first."
                )
                return

            # Load ground truth doc for this repo (enables GT verify injection and binary metrics).
            gt_doc: Optional[dict] = None
            gt_verify_commands: list[str] = []
            if args.dataset_dir:
                gt_doc = load_gt_for_repo(Path(args.dataset_dir), repo_url)
                if gt_doc:
                    gt_verify_commands = get_gt_verify_commands(gt_doc)
                    if gt_verify_commands:
                        # Record GT commands whenever they exist (not only when executed):
                        # the evaluator-side similarity judge and HARD hashing consume them,
                        # but SOFT verify no longer executes them.
                        report["gt_verify_commands"] = gt_verify_commands
                        log_info(f"[gt {repo_name}] Loaded {len(gt_verify_commands)} GT verify command(s) from dataset")

            # Resume: skip repair if a successful report already exists and --force is not set.
            if not args.force and report_path.exists():
                existing_report = read_yaml_file(report_path)
                if existing_report and existing_report.get("success"):
                    log_info(f"Skipping {repo_url}: existing successful repair report found at {report_path}")
                    return

            repo_path = resolve_repo_checkout_dir(repos_dir, repo_name)
            if not await ensure_repo_checkout(repo_url, repo_path, "skipping Dockerfile repair"):
                return

            # Reset repo to clean state for reproducibility
            await _reset_repo_for_repair(repo_path, repo_name)

            # Docs/CI are ALWAYS stripped before the build — no opt-out. Mirrors the
            # Stage-1 strip; the hard reset above restored the files, so re-strip here.
            await asyncio.to_thread(
                delete_files_build_context, repo_path, repo_name, get_files_to_delete(gt_doc)
            )

            # Load the build verification command: agent-generated sidecar > CLI default.
            # SOFT verify executes the AGENT's own command. The GT command is recorded
            # (report["gt_verify_commands"], above) and consumed only by the evaluator-side
            # similarity judge and HARD artifact hashing — it is never executed here.
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
            # Clean version-pinned gcc-N/g++-N from the initial (stage-2) Dockerfile too, so
            # the first build is not wasted on a redundant/absent compiler package
            # (build-essential in base provides the compiler).
            current_dockerfile, _removed_initial = strip_versioned_toolchain(current_dockerfile)
            if _removed_initial:
                write_text(dockerfile_path, current_dockerfile)
                log_info(f"[toolchain-strip {repo_name}] removed version-pinned {sorted(set(_removed_initial))} from initial Dockerfile")
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
                # Snapshot what actually built (post root-wrapping injection), not the
                # pre-injection source: build-time mutations like ensure_root_for_sensitive_runs
                # can change structure/line numbers, and the build log references the injected
                # file's line numbers. Strip the multi-kilobyte CA-cert block so the snapshot
                # stays readable while still reflecting the real built Dockerfile.
                write_text(dockerfile_snapshot_path, strip_ca_cert_from_dockerfile(dockerfile_for_build))

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
                # LLM-driven diagnosis upgrades a regex `unknown` into a real class (feeds the
                # strategy ledger + targeted hints); no-op when the regex is already confident
                # or on the baseline path. Falls back to the regex hint on any failure.
                build_failure_hints = await refine_failure_hints_with_llm(
                    build_failure_hints,
                    log=build_log,
                    dockerfile_content=current_dockerfile,
                    llm_metrics=llm_metrics,
                    repo_url=repo_url,
                )
                # Eagerly resolve missing packages/libraries against the real base apt repos
                # and attach the candidates, so the repair prompt carries the answer instead
                # of relying on the LLM to invoke apt_search itself (which it under-used). Each
                # resolver self-gates on whether its parser finds anything in the log, so this
                # runs regardless of the (coarser) failure category. Flows into the prompt via
                # render_failure_hints_for_prompt + the directives in request_repair.
                repair_base_image = extract_base_image(current_dockerfile)
                if repair_base_image:
                    apt_candidates = resolve_unavailable_apt_packages(
                        build_log, args.container_cli, repair_base_image
                    )
                    if apt_candidates:
                        build_failure_hints["apt_candidates"] = apt_candidates
                    # pkg-config/cargo "system library X not found" -> matching -dev pkg
                    dev_lib_candidates = resolve_missing_system_libs(
                        build_log, args.container_cli, repair_base_image
                    )
                    if dev_lib_candidates:
                        build_failure_hints["dev_lib_candidates"] = dev_lib_candidates
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
                    # Record final image size (quality metric: multi-stage builds aim small).
                    report["attempts"][-1]["image_size_bytes"] = await get_image_size_bytes(image_tag)
                    # HARD verify inputs (TODO 1/28): hash the produced artifact and gather the
                    # artifact listing on ANY successful build, independent of whether the agent's
                    # verify command passes. This decouples HARD (build_ok AND hash-match) from SOFT.
                    # The evaluator's combined-hash check consumes artifact_listing; binary_metrics
                    # holds the single key-artifact hash. Latest successful build wins.
                    if gt_doc:
                        _, hard_workdir = await get_image_runtime_context(image_tag)
                        gt_artifact = get_gt_key_artifact(gt_doc, gt_verify_commands)
                        if gt_artifact:
                            binary_metrics = await measure_binary_in_image(
                                image_tag,
                                gt_artifact["path"],
                                hard_workdir,
                                gt_artifact.get("size_bytes"),
                                gt_artifact.get("digest"),
                            )
                            report["binary_metrics"] = binary_metrics
                            log_info(
                                f"[binary {repo_name}] size={binary_metrics.get('measured_size_bytes')}, "
                                f"plausible={binary_metrics.get('binary_size_plausible')}, "
                                f"hash_match={binary_metrics.get('binary_hash_match')}"
                            )
                        # The artifact listing (xxh64sum of every file under the workdir) can be
                        # hundreds of MB on large repos (e.g. linux). Spill it to a sidecar file
                        # and keep only a path ref in report.yaml so the report stays small/parseable;
                        # eval.py loads the sidecar to run the combined-hash hard verify.
                        listing = await gather_artifact_listing(image_tag, hard_workdir)
                        if listing:
                            listing_path = report_dir / "artifact-listing.json"
                            write_text(listing_path, json.dumps(listing))
                            report["artifact_listing"] = {"path": str(listing_path)}
                        else:
                            report["artifact_listing"] = None
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
                    # LLM diagnosis can classify a verify-only failure (build ok, wrong check)
                    # as verify_command_wrong — the signal the loop needs to fix the command,
                    # not the build (F6).
                    verify_failure_hints = await refine_failure_hints_with_llm(
                        verify_failure_hints,
                        log=verify_log,
                        dockerfile_content=current_dockerfile,
                        llm_metrics=llm_metrics,
                        repo_url=repo_url,
                    )
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
                    repaired_dockerfile = await _safe_request_repair(
                        repo_url,
                        attempt,
                        request_repair(
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
                            repo_path=repo_path,
                            report_dir=report_dir,
                            # Reached only after a SUCCESSFUL build whose verification failed
                            # and whose verify-command repair was insufficient (F6): tell the
                            # Dockerfile rewrite to preserve the working build steps.
                            verify_only_failure=True,
                        ),
                    )
                    if repaired_dockerfile is None:
                        continue
                    prior_dockerfile = current_dockerfile
                    current_dockerfile, stop = _apply_repair(
                        repo_url, repo_name, current_dockerfile, repaired_dockerfile,
                        dockerfile_path, report_dir, attempt,
                    )
                    # Track history UNCONDITIONALLY: the always-on diagnose-then-act
                    # strategy ledger needs it. The stateful-repair PROMPT rendering
                    # (history text + decision tree) stays gated on args.stateful_repair,
                    # so the stateful ablation semantics are unchanged.
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
                repaired_dockerfile = await _safe_request_repair(
                    repo_url,
                    attempt,
                    request_repair(
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
                        repo_path=repo_path,
                        report_dir=report_dir,
                    ),
                )
                if repaired_dockerfile is None:
                    continue
                prior_dockerfile = current_dockerfile
                current_dockerfile, stop = _apply_repair(
                    repo_url, repo_name, current_dockerfile, repaired_dockerfile,
                    dockerfile_path, report_dir, attempt,
                )
                # Unconditional history for the diagnose-then-act ledger (see note above).
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
            log_error(
                f"Unexpected error while repairing Dockerfile for {repo_url}: "
                f"{type(error).__name__}: {error}"
            )
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