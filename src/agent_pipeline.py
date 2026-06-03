import argparse
from datetime import datetime, timezone
import os
import subprocess
import sys
import time
from pathlib import Path

try:
    from RepoBuilderAgent.src.config import OPENAI_API_KEY, OPENAI_BASE_URL, OPENAI_MODEL
    from RepoBuilderAgent.src.log_utils import log_error, log_info, set_trace_enabled
    from RepoBuilderAgent.src.timeout_config import load_timeout_defaults
    from RepoBuilderAgent.src.prompt_profiles import (
        prompt_profile_metadata,
        resolve_prompt_profile,
        resolve_prompt_temperature,
    )
except ImportError:
    # Fallback for direct script execution from RepoBuilderAgent/src
    import config as _config
    from log_utils import log_error, log_info, set_trace_enabled
    from timeout_config import load_timeout_defaults
    from prompt_profiles import (
        prompt_profile_metadata,
        resolve_prompt_profile,
        resolve_prompt_temperature,
    )

    OPENAI_API_KEY = getattr(_config, "OPENAI_API_KEY", "")
    OPENAI_BASE_URL = getattr(_config, "OPENAI_BASE_URL", "https://api.openai.com/v1")
    OPENAI_MODEL = getattr(_config, "OPENAI_MODEL", "gpt-4o")
import yaml


TIMEOUTS = load_timeout_defaults(
    "agent_pipeline",
    {
        "timeout": 120,
        "llm_max_retries": 2,
        "llm_retry_backoff_seconds": 2.0,
        "selection_timeout": 120,
        "classification_timeout": 240,
        "dockerfile_timeout": 240,
        "verify_cmd_timeout": 180,
        "repair_timeout": 240,
        "verify_repair_timeout": 180,
        "install_guide_timeout": 240,
        "verify_timeout": 30,
    },
)


parser = argparse.ArgumentParser(
    description="Run the full repository pipeline: classify, generate Dockerfiles, and repair failing Docker builds."
)
parser.add_argument("--input-file", default="repos.json", help="Path to input file containing repository URLs")
parser.add_argument(
    "--repo-url",
    action="append",
    default=[],
    help="Run the pipeline for a specific repository URL (can be passed multiple times). Overrides --input-file when provided.",
)
parser.add_argument("--endpoint", default=os.getenv("LLM_ENDPOINT", OPENAI_BASE_URL), help="Custom API endpoint URL")
parser.add_argument("--model", default=os.getenv("LLM_MODEL", OPENAI_MODEL), help="Model name")
parser.add_argument("--api-key", default=os.getenv("LLM_API_KEY", OPENAI_API_KEY), help="API key")
parser.add_argument("--prompt-profile", default=os.getenv("PROMPT_PROFILE", "P*"), help="Prompt profile name from RepoBuilderAgent/config/prompt_profiles.yaml (supports alias P*)")
parser.add_argument("--temperature", type=float, default=None, help="Temperature override for model calls; defaults to selected prompt profile value")
parser.add_argument("--timeout", type=int, default=int(TIMEOUTS["timeout"]), help="Timeout for API requests in seconds")
parser.add_argument("--llm-max-retries", type=int, default=int(TIMEOUTS["llm_max_retries"]), help="Maximum retries for transient LLM timeouts and retryable API errors")
parser.add_argument("--llm-retry-backoff-seconds", type=float, default=float(TIMEOUTS["llm_retry_backoff_seconds"]), help="Base exponential backoff delay in seconds for LLM retries")
parser.add_argument("--selection-timeout", type=int, default=int(TIMEOUTS["selection_timeout"]), help="Timeout for classify step1 file-selection calls in seconds")
parser.add_argument("--classification-timeout", type=int, default=int(TIMEOUTS["classification_timeout"]), help="Timeout for classify step2 classification calls in seconds")
parser.add_argument("--dockerfile-timeout", type=int, default=int(TIMEOUTS["dockerfile_timeout"]), help="Timeout for Dockerfile generation calls in seconds")
parser.add_argument("--verify-cmd-timeout", type=int, default=int(TIMEOUTS["verify_cmd_timeout"]), help="Timeout for Dockerfile verification-command generation calls in seconds")
parser.add_argument("--repair-timeout", type=int, default=int(TIMEOUTS["repair_timeout"]), help="Timeout for Dockerfile repair calls in seconds")
parser.add_argument("--verify-repair-timeout", type=int, default=int(TIMEOUTS["verify_repair_timeout"]), help="Timeout for verification-command repair calls in seconds")
parser.add_argument("--install-guide-timeout", type=int, default=int(TIMEOUTS["install_guide_timeout"]), help="Timeout for install-guide generation calls in seconds")
parser.add_argument("--trace", action="store_true", help="Enable verbose trace logs")
parser.add_argument("--force", action="store_true", help="Overwrite existing generated artifacts where supported")
parser.add_argument("--learn", action="store_true", help="Enable learning of new manifest file patterns during classification")
parser.add_argument("--preprocess", action="store_true", help="Enable repository preprocessing during classification")
parser.add_argument("--deletion-patterns", default="config/deletion-patterns.yaml", help="Path to YAML file with deletion patterns for preprocessing")
parser.add_argument("--results-dir", default="classification_results", help="Directory containing classification result YAML files")
parser.add_argument("--summaries-dir", default="summaries", help="Directory containing repository summary files")
parser.add_argument("--repos-dir", default="repos", help="Directory containing cloned repositories")
parser.add_argument("--dockerfiles-dir", default="dockerfiles", help="Directory containing generated Dockerfiles")
parser.add_argument("--reports-dir", default="repair-reports", help="Directory where repair logs and reports are written")
parser.add_argument("--install-guides-dir", default="install-guides", help="Directory where generated INSTALL.md guides are written")
parser.add_argument("--analysis-dir", default="analysis", help="Directory where analysis outputs are written when --run-analysis is enabled")
parser.add_argument("--container-cli", default="docker", help="Container CLI to use for repair builds")
parser.add_argument("--max-attempts", type=int, default=3, help="Maximum number of repair attempts per repository")
parser.add_argument("--max-log-chars", type=int, default=24000, help="Maximum number of build log characters to send to the repair model")
parser.add_argument("--skip-delete-docs", action="store_true", help="Skip deleting documentation and CI/CD files from the build context before building")
parser.add_argument("--skip-hadolint", action="store_true", help="Skip Dockerfile syntax validation via hadolint before docker build")
parser.add_argument("--verify-command", default="echo build-ok", help="Shell command executed inside built images to verify the build produced working software")
parser.add_argument("--verify-timeout", type=int, default=int(TIMEOUTS["verify_timeout"]), help="Timeout in seconds for build verification container execution")
stateful_group = parser.add_mutually_exclusive_group()
stateful_group.add_argument(
    "--stateful-repair",
    dest="stateful_repair",
    action="store_true",
    help="Enable stateful Dockerfile repair prompts that include compact summaries of previous repair attempts.",
)
stateful_group.add_argument(
    "--no-stateful-repair",
    dest="stateful_repair",
    action="store_false",
    help="Disable stateful Dockerfile repair prompts.",
)
parser.set_defaults(stateful_repair=False)
parser.add_argument(
    "--stateful-history-window",
    type=int,
    default=4,
    help="When stateful repair is enabled, include at most this many recent repair attempts in prompt history.",
)
parser.add_argument(
    "--stateful-history-max-chars",
    type=int,
    default=4000,
    help="Maximum characters from serialized repair history included in each stateful repair prompt.",
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
    help="Maximum characters from serialized stateful decision tree included in each repair prompt.",
)
parser.add_argument(
    "--stateful-tree-max-children",
    type=int,
    default=5,
    help="Maximum child branches retained per decision-tree node before pruning.",
)
parser.add_argument("--skip-classify", action="store_true", help="Skip the classification phase")
parser.add_argument("--skip-dockerfile", action="store_true", help="Skip the Dockerfile generation phase")
parser.add_argument("--skip-repair", action="store_true", help="Skip the Dockerfile repair phase")
parser.add_argument("--skip-install-guide", action="store_true", help="Skip the INSTALL.md generation phase")
parser.add_argument("--run-analysis", action="store_true", help="Run parse_results.py after classification completes")
parser.add_argument("--pipeline-reports-dir", default="pipeline-reports", help="Directory where pipeline logs and summary are written")
parser.add_argument("--pipeline-summary-path", default="", help="Optional explicit path for the pipeline summary YAML")
parser.add_argument("--print-summary", action="store_true", help="Print planned pipeline summary and exit without running phases")
args = parser.parse_args()
PROMPT_PROFILE = resolve_prompt_profile(args.prompt_profile)
EFFECTIVE_TEMPERATURE = resolve_prompt_temperature(args.temperature, PROMPT_PROFILE)


DEFAULT_RESULTS_DIR = "classification_results"
DEFAULT_SUMMARIES_DIR = "summaries"
DEFAULT_DOCKERFILES_DIR = "dockerfiles"
DEFAULT_REPORTS_DIR = "repair-reports"
DEFAULT_INSTALL_GUIDES_DIR = "install-guides"
DEFAULT_PIPELINE_REPORTS_DIR = "pipeline-reports"
DEFAULT_ANALYSIS_DIR = "analysis"

RUN_DIR_DEFAULTS = {
    "results_dir": (DEFAULT_RESULTS_DIR, "classification_results"),
    "summaries_dir": (DEFAULT_SUMMARIES_DIR, "summaries"),
    "dockerfiles_dir": (DEFAULT_DOCKERFILES_DIR, "dockerfiles"),
    "reports_dir": (DEFAULT_REPORTS_DIR, "repair-reports"),
    "install_guides_dir": (DEFAULT_INSTALL_GUIDES_DIR, "install-guides"),
    "pipeline_reports_dir": (DEFAULT_PIPELINE_REPORTS_DIR, "pipeline-reports"),
    "analysis_dir": (DEFAULT_ANALYSIS_DIR, "analysis"),
}


set_trace_enabled(args.trace)


def _load_dotenv_fallback(dotenv_path: Path) -> dict[str, str]:
    loaded: dict[str, str] = {}
    if not dotenv_path.exists():
        return loaded
    for raw_line in dotenv_path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key:
            loaded[key] = value
    return loaded


def _resolve_llm_arg_defaults(workspace_root: Path) -> None:
    dotenv = _load_dotenv_fallback(workspace_root / ".env")

    if not args.endpoint:
        args.endpoint = (
            os.getenv("LLM_ENDPOINT")
            or os.getenv("OPENAI_BASE_URL")
            or dotenv.get("LLM_ENDPOINT")
            or dotenv.get("OPENAI_BASE_URL")
            or ""
        )

    if not args.model:
        args.model = (
            os.getenv("LLM_MODEL")
            or os.getenv("OPENAI_MODEL")
            or dotenv.get("LLM_MODEL")
            or dotenv.get("OPENAI_MODEL")
            or ""
        )

    if not args.api_key:
        args.api_key = (
            os.getenv("LLM_API_KEY")
            or os.getenv("OPENAI_API_KEY")
            or dotenv.get("LLM_API_KEY")
            or dotenv.get("OPENAI_API_KEY")
            or ""
        )


def _synchronize_llm_environment() -> None:
    if args.endpoint:
        os.environ["LLM_ENDPOINT"] = args.endpoint
        os.environ["OPENAI_BASE_URL"] = args.endpoint
    if args.model:
        os.environ["LLM_MODEL"] = args.model
        os.environ["OPENAI_MODEL"] = args.model
    if args.api_key:
        os.environ["LLM_API_KEY"] = args.api_key
        os.environ["OPENAI_API_KEY"] = args.api_key


def _resolve_workspace_root(src_dir: Path) -> Path:
    repo_root = src_dir.parent.parent
    input_path = Path(args.input_file).expanduser()

    candidates: list[Path] = []
    if input_path.is_absolute():
        candidates.append(input_path)
    else:
        candidates.append((Path.cwd() / input_path).resolve())
        candidates.append((repo_root / input_path).resolve())

    for candidate in candidates:
        if candidate.exists():
            return candidate.parent if candidate.is_file() else candidate

    return repo_root


def sanitize_command(command: list[str]) -> list[str]:
    sanitized = command.copy()
    for index, part in enumerate(sanitized[:-1]):
        if part == "--api-key":
            sanitized[index + 1] = "***REDACTED***"
    return sanitized


def render_command(command: list[str]) -> str:
    return " ".join(sanitize_command(command))


def append_shared_model_args(command: list[str]) -> list[str]:
    command.extend(["--prompt-profile", args.prompt_profile])
    if args.endpoint:
        command.extend(["--endpoint", args.endpoint])
    if args.model:
        command.extend(["--model", args.model])
    if args.api_key:
        command.extend(["--api-key", args.api_key])
    if args.temperature is not None:
        command.extend(["--temperature", str(args.temperature)])
    command.extend([
        "--timeout", str(args.timeout),
        "--llm-max-retries", str(args.llm_max_retries),
        "--llm-retry-backoff-seconds", str(args.llm_retry_backoff_seconds),
    ])
    if args.trace:
        command.append("--trace")
    return command


def append_repo_selection_args(command: list[str]) -> list[str]:
    command.extend(["--input-file", args.input_file])
    for repo_url in args.repo_url:
        command.extend(["--repo-url", repo_url])
    return command


def build_agent_command(python_executable: str, script_path: Path, *, include_model_args: bool = True) -> list[str]:
    command = [python_executable, str(script_path)]
    append_repo_selection_args(command)
    if include_model_args:
        append_shared_model_args(command)
    return command


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def resolve_summary_path(workspace_root: Path, run_id: str) -> Path:
    if args.pipeline_summary_path:
        candidate = Path(args.pipeline_summary_path)
        if not candidate.is_absolute():
            candidate = workspace_root / candidate
        return candidate
    return Path(args.pipeline_reports_dir) / f"pipeline-summary-{run_id}.yaml"


def resolve_output_dir(workspace_root: Path, run_dir: Path, value: str, default_value: str, run_subdir: str) -> Path:
    if value == default_value:
        return run_dir / run_subdir

    candidate = Path(value)
    if not candidate.is_absolute():
        candidate = workspace_root / candidate
    return candidate


def run_step(name: str, command: list[str], log_path: Path) -> dict:
    started_at = utc_now()
    started_ts = time.perf_counter()
    rendered_command = render_command(command)
    log_info(f"Running {name}: {rendered_command}")

    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "w", encoding="utf-8") as step_log:
        step_log.write(f"$ {rendered_command}\n\n")
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )

        assert process.stdout is not None
        for line in process.stdout:
            sys.stdout.write(line)
            step_log.write(line)

        return_code = process.wait()

    duration_seconds = round(time.perf_counter() - started_ts, 3)
    step_result = {
        "name": name,
        "status": "success" if return_code == 0 else "failed",
        "started_at": started_at,
        "ended_at": utc_now(),
        "duration_seconds": duration_seconds,
        "exit_code": return_code,
        "command": sanitize_command(command),
        "log_path": str(log_path),
    }

    if return_code != 0:
        raise RuntimeError(f"{name} failed with exit code {return_code}")
    return step_result


def build_classify_command(python_executable: str, script_path: Path) -> list[str]:
    command = build_agent_command(python_executable, script_path)
    if args.force:
        command.append("--force")
    if args.learn:
        command.append("--learn")
    if args.preprocess:
        command.append("--preprocess")
    command.extend([
        "--deletion-patterns", args.deletion_patterns,
        "--selection-timeout", str(args.selection_timeout),
        "--classification-timeout", str(args.classification_timeout),
        "--results-dir", args.results_dir,
        "--summaries-dir", args.summaries_dir,
        "--repos-dir", args.repos_dir,
        "--analysis-dir", args.analysis_dir,
    ])
    command.append("--no-analysis")
    return command


def build_dockerfile_command(python_executable: str, script_path: Path) -> list[str]:
    command = build_agent_command(python_executable, script_path)
    if args.force:
        command.append("--force")
    command.extend([
        "--dockerfile-timeout", str(args.dockerfile_timeout),
        "--verify-cmd-timeout", str(args.verify_cmd_timeout),
        "--results-dir", args.results_dir,
        "--summaries-dir", args.summaries_dir,
        "--repos-dir", args.repos_dir,
        "--output-dir", args.dockerfiles_dir,
    ])
    return command


def build_repair_command(python_executable: str, script_path: Path) -> list[str]:
    command = build_agent_command(python_executable, script_path)
    command.extend([
        "--repair-timeout", str(args.repair_timeout),
        "--verify-repair-timeout", str(args.verify_repair_timeout),
        "--results-dir", args.results_dir,
        "--summaries-dir", args.summaries_dir,
        "--repos-dir", args.repos_dir,
        "--dockerfiles-dir", args.dockerfiles_dir,
        "--reports-dir", args.reports_dir,
        "--container-cli", args.container_cli,
        "--max-attempts", str(args.max_attempts),
        "--max-log-chars", str(args.max_log_chars),
    ])
    if args.skip_delete_docs:
        command.append("--skip-delete-docs")
    if args.skip_hadolint:
        command.append("--skip-hadolint")
    command.extend([
        "--verify-command", args.verify_command,
        "--verify-timeout", str(args.verify_timeout),
    ])
    if args.stateful_repair:
        command.append("--stateful-repair")
    else:
        command.append("--no-stateful-repair")
    command.extend([
        "--stateful-history-window", str(args.stateful_history_window),
        "--stateful-history-max-chars", str(args.stateful_history_max_chars),
    ])
    if args.stateful_repair_tree:
        command.append("--stateful-repair-tree")
    else:
        command.append("--no-stateful-repair-tree")
    command.extend([
        "--stateful-tree-max-chars", str(args.stateful_tree_max_chars),
        "--stateful-tree-max-children", str(args.stateful_tree_max_children),
    ])
    return command


def build_analysis_command(python_executable: str, script_path: Path) -> list[str]:
    command = build_agent_command(python_executable, script_path, include_model_args=False)
    command.extend([
        "--results-dir", args.results_dir,
        "--summaries-dir", args.summaries_dir,
        "--analysis-dir", args.analysis_dir,
    ])
    return command


def build_install_guide_command(python_executable: str, script_path: Path) -> list[str]:
    command = build_agent_command(python_executable, script_path)
    if args.force:
        command.append("--force")
    command.extend([
        "--install-guide-timeout", str(args.install_guide_timeout),
        "--results-dir", args.results_dir,
        "--summaries-dir", args.summaries_dir,
        "--repos-dir", args.repos_dir,
        "--dockerfiles-dir", args.dockerfiles_dir,
        "--output-dir", args.install_guides_dir,
    ])
    return command


def resolve_python_executable(workspace_root: Path) -> str:
    """Prefer workspace venv Python for child agent invocations."""
    venv_root = workspace_root / ".venv"
    venv_python = workspace_root / ".venv" / "bin" / "python"
    current_prefix = Path(sys.prefix).resolve()
    if venv_python.exists() and current_prefix != venv_root.resolve():
        log_info(f"Using workspace venv interpreter for child agents: {venv_python}")
        return str(venv_python)
    return sys.executable


def write_summary(path: Path, summary: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as summary_file:
        yaml.dump(summary, summary_file, sort_keys=False, allow_unicode=True)


def collect_repair_outcomes(reports_dir: Path) -> dict:
    """Aggregate build/verification outcomes from per-repo repair reports."""
    outcome = {
        "total_reports": 0,
        "build_success": 0,
        "build_failed": 0,
        "verify_passed": 0,
        "verify_failed": 0,
        "verify_missing": 0,
    }

    if not reports_dir.exists():
        return outcome

    for report_path in reports_dir.glob("*/report.yaml"):
        try:
            with open(report_path, "r", encoding="utf-8") as report_file:
                report = yaml.safe_load(report_file) or {}
        except Exception:
            continue

        outcome["total_reports"] += 1

        build_ok = bool(report.get("success", False))
        if build_ok:
            outcome["build_success"] += 1
        else:
            outcome["build_failed"] += 1

        attempts = report.get("attempts", []) or []
        verification = None
        for attempt in reversed(attempts):
            build_verification = attempt.get("build_verification")
            if build_verification:
                verification = build_verification
                break

        if verification is None:
            outcome["verify_missing"] += 1
        elif verification.get("exit_code") == 0:
            outcome["verify_passed"] += 1
        else:
            outcome["verify_failed"] += 1

    return outcome


def _update_phase_totals(target: dict, phase_data: dict) -> None:
    for key in (
        "calls",
        "success",
        "timeout",
        "connection_error",
        "api_error",
        "http_error",
        "ssl_error",
        "other_error",
        "retries",
    ):
        target[key] = target.get(key, 0) + int(phase_data.get(key, 0) or 0)

    latencies = phase_data.get("latencies_seconds", []) or []
    latency_count = len(latencies)
    if latency_count:
        latency_sum = float(sum(latencies))
        target["latency_count"] = target.get("latency_count", 0) + latency_count
        target["latency_sum"] = target.get("latency_sum", 0.0) + latency_sum
        target["latency_min"] = min(target.get("latency_min", float("inf")), min(latencies))
        target["latency_max"] = max(target.get("latency_max", 0.0), max(latencies))


def _finalize_phase_totals(phase_totals: dict) -> None:
    latency_count = phase_totals.pop("latency_count", 0)
    latency_sum = phase_totals.pop("latency_sum", 0.0)
    latency_min = phase_totals.pop("latency_min", None)
    latency_max = phase_totals.pop("latency_max", None)

    if latency_count:
        phase_totals["latency_summary_seconds"] = {
            "min": round(float(latency_min), 3),
            "avg": round(float(latency_sum) / latency_count, 3),
            "max": round(float(latency_max), 3),
            "samples": latency_count,
        }


def aggregate_llm_metrics(results_dir: Path, dockerfiles_dir: Path, repair_reports_dir: Path, install_guides_dir: Path) -> dict:
    stage_globs = {
        "classification": (results_dir, "*.llm-metrics.yaml"),
        "dockerfile": (dockerfiles_dir, "*.llm-metrics.yaml"),
        "repair": (repair_reports_dir, "*/llm-metrics.yaml"),
        "install_guide": (install_guides_dir, "*/llm-metrics.yaml"),
    }

    summary: dict = {
        "stages": {},
        "overall": {
            "files": 0,
            "repos": 0,
            "phase_totals": {},
        },
    }

    overall_repos: set[str] = set()

    for stage_name, (base_dir, glob_pattern) in stage_globs.items():
        stage_phase_totals: dict = {}
        stage_repos: set[str] = set()
        stage_files: list[str] = []

        if not base_dir.exists():
            summary["stages"][stage_name] = {
                "files": 0,
                "repos": 0,
                "phase_totals": {},
                "metric_files": [],
            }
            continue

        metric_files = sorted(base_dir.glob(glob_pattern))

        for metrics_path in metric_files:
            try:
                with open(metrics_path, "r", encoding="utf-8") as metrics_file:
                    metrics = yaml.safe_load(metrics_file) or {}
            except Exception:
                continue

            stage_files.append(str(metrics_path))
            repo = str(metrics.get("repo", "")).strip()
            if repo:
                stage_repos.add(repo)
                overall_repos.add(repo)

            for phase_name, phase_data in (metrics.get("phases", {}) or {}).items():
                phase_totals = stage_phase_totals.setdefault(phase_name, {})
                _update_phase_totals(phase_totals, phase_data or {})

                overall_phase_totals = summary["overall"]["phase_totals"].setdefault(phase_name, {})
                _update_phase_totals(overall_phase_totals, phase_data or {})

        for phase_totals in stage_phase_totals.values():
            _finalize_phase_totals(phase_totals)

        summary["stages"][stage_name] = {
            "files": len(stage_files),
            "repos": len(stage_repos),
            "phase_totals": stage_phase_totals,
            "metric_files": stage_files,
        }

    for phase_totals in summary["overall"]["phase_totals"].values():
        _finalize_phase_totals(phase_totals)

    summary["overall"]["files"] = sum(stage.get("files", 0) for stage in summary["stages"].values())
    summary["overall"]["repos"] = len(overall_repos)
    return summary


def print_planned_summary() -> None:
    summary_text = (
        "Agentic flow: infer -> generate -> build -> diagnose -> repair -> verify -> document -> report.\n\n"
        "1. Input selection\n"
        "The pipeline picks which repos to process (from list or direct URL).\n\n"
        "2. Classification phase\n"
        "It inspects each repo and infers install/build facts (language, build tool, system deps, runtime hints, etc.).\n\n"
        "3. Optional analysis phase\n"
        "It can aggregate/parse classification artifacts into summary analysis outputs.\n\n"
        "4. Dockerfile generation\n"
        "It generates an initial Dockerfile from the classification + repository summary.\n\n"
        "5. Repair loop\n"
        "It tries to build the Dockerfile, captures full failure logs, asks the model to fix the Dockerfile, and retries up to N attempts.\n\n"
        "6. Build verification\n"
        "After a successful build, it runs an in-container verification command to confirm the built artifact is actually usable (not just \"image built\").\n\n"
        "7. INSTALL guide generation\n"
        "It generates a human-readable INSTALL.md from the final Dockerfile and verification command so a developer can reproduce the build outside the container.\n\n"
        "8. Reporting\n"
        "It writes per-attempt logs plus a pipeline summary with phase status, timings, commands, and artifact paths.\n\n"
    )
    print(summary_text, end="")


def main() -> int:
    run_id = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    pipeline_started_ts = time.perf_counter()
    pipeline_started_at = utc_now()
    src_dir = Path(__file__).resolve().parent
    workspace_root = _resolve_workspace_root(src_dir)
    _resolve_llm_arg_defaults(workspace_root)
    _synchronize_llm_environment()
    run_dir = workspace_root / "runs" / f"run-{run_id}"

    for attr_name, (default_value, run_subdir) in RUN_DIR_DEFAULTS.items():
        setattr(
            args,
            attr_name,
            str(resolve_output_dir(workspace_root, run_dir, getattr(args, attr_name), default_value, run_subdir)),
        )

    python_executable = resolve_python_executable(workspace_root)
    reports_dir = Path(args.pipeline_reports_dir)
    run_logs_dir = reports_dir / run_id
    summary_path = resolve_summary_path(workspace_root, run_id)
    llm_metrics_summary_path = reports_dir / f"llm-metrics-summary-{run_id}.yaml"

    classify_script = src_dir / "agent_classify.py"
    dockerfile_script = src_dir / "agent_dockerfile.py"
    repair_script = src_dir / "agent_dockerfile_repair.py"
    install_guide_script = src_dir / "agent_install_guide.py"
    analysis_script = src_dir / "parse_results.py"

    if args.print_summary:
        print_planned_summary()
        return 0

    phases_selected = not (args.skip_classify and args.skip_dockerfile and args.skip_repair and args.skip_install_guide)
    if not phases_selected and not args.run_analysis:
        log_error("Nothing to do. All pipeline phases were skipped and --run-analysis was not set.")
        return 1

    summary: dict = {
        "run_id": run_id,
        "started_at": pipeline_started_at,
        "ended_at": None,
        "duration_seconds": None,
        "status": "failed",
        "trace": {
            "enabled": args.trace,
            "forwarded_to_child_agents": args.trace,
        },
        "prompt_profile": prompt_profile_metadata(PROMPT_PROFILE, EFFECTIVE_TEMPERATURE),
        "paths": {
            "run_dir": str(run_dir),
            "reports_dir": str(reports_dir),
            "run_logs_dir": str(run_logs_dir),
            "summary_path": str(summary_path),
            "llm_metrics_summary_path": str(llm_metrics_summary_path),
            "results_dir": args.results_dir,
            "summaries_dir": args.summaries_dir,
            "dockerfiles_dir": args.dockerfiles_dir,
            "repair_reports_dir": args.reports_dir,
            "install_guides_dir": args.install_guides_dir,
            "analysis_dir": args.analysis_dir,
        },
        "phases": [],
        "error": None,
    }

    try:
        if not args.skip_classify:
            summary["phases"].append(
                run_step(
                    "classification",
                    build_classify_command(python_executable, classify_script),
                    run_logs_dir / "classification.log",
                )
            )

        if args.run_analysis:
            summary["phases"].append(
                run_step(
                    "analysis",
                    build_analysis_command(python_executable, analysis_script),
                    run_logs_dir / "analysis.log",
                )
            )

        if not args.skip_dockerfile:
            summary["phases"].append(
                run_step(
                    "dockerfile generation",
                    build_dockerfile_command(python_executable, dockerfile_script),
                    run_logs_dir / "dockerfile-generation.log",
                )
            )

        if not args.skip_repair:
            summary["phases"].append(
                run_step(
                    "dockerfile repair",
                    build_repair_command(python_executable, repair_script),
                    run_logs_dir / "dockerfile-repair.log",
                )
            )

        if not args.skip_install_guide:
            summary["phases"].append(
                run_step(
                    "install guide generation",
                    build_install_guide_command(python_executable, install_guide_script),
                    run_logs_dir / "install-guide-generation.log",
                )
            )

        summary["status"] = "success"

    except RuntimeError as error:
        summary["error"] = str(error)
        log_error(str(error))
    finally:
        summary["ended_at"] = utc_now()
        summary["duration_seconds"] = round(time.perf_counter() - pipeline_started_ts, 3)
        write_summary(summary_path, summary)
        log_info(f"Pipeline summary written to {summary_path}")

    if summary["status"] != "success":
        return 1

    if not args.skip_repair:
        repair_outcomes = collect_repair_outcomes(Path(args.reports_dir))
        log_info(
            f"Build outcomes ({repair_outcomes['total_reports']} reports)"
            f"  |  build ok={repair_outcomes['build_success']} failed={repair_outcomes['build_failed']}"
            f"  |  verify ok={repair_outcomes['verify_passed']} failed={repair_outcomes['verify_failed']} missing={repair_outcomes['verify_missing']}"
        )

    llm_metrics_summary = aggregate_llm_metrics(
        Path(args.results_dir),
        Path(args.dockerfiles_dir),
        Path(args.reports_dir),
        Path(args.install_guides_dir),
    )
    write_summary(llm_metrics_summary_path, llm_metrics_summary)
    log_info(
        f"LLM metrics summary written to {llm_metrics_summary_path} "
        f"(files={llm_metrics_summary['overall']['files']} repos={llm_metrics_summary['overall']['repos']})"
    )

    log_info("Pipeline process completed (note: this indicates the pipeline ran without errors, not that builds were successful).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())