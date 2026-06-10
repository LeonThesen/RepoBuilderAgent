import argparse
from datetime import datetime, timezone
import json
import os
import subprocess
import sys
import time
from pathlib import Path

try:
    from RepoBuilderAgent.src.core.config import OPENAI_API_KEY, OPENAI_BASE_URL, OPENAI_MODEL
    from RepoBuilderAgent.src.core.common import upsert_shared_repository_state
    from RepoBuilderAgent.src.core.log_utils import log_error, log_info, set_trace_enabled
    from RepoBuilderAgent.src.core.timeout_config import load_timeout_defaults
    from RepoBuilderAgent.src.core.prompt_profiles import (
        prompt_profile_metadata,
        resolve_prompt_profile,
        resolve_prompt_temperature,
    )
except ImportError:
    # Fallback for direct script execution from RepoBuilderAgent/src
    import core.config as _config
    from core.common import upsert_shared_repository_state
    from core.log_utils import log_error, log_info, set_trace_enabled
    from core.timeout_config import load_timeout_defaults
    from core.prompt_profiles import (
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
parser.add_argument("--endpoint", default="", help="Custom API endpoint URL")
parser.add_argument("--model", default="", help="Model name")
parser.add_argument("--api-key", default="", help="API key")
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
parser.add_argument("--skip-validation-gate", action="store_true", help="Skip the post-generation validation gate phase")
parser.add_argument("--skip-repair", action="store_true", help="Skip the Dockerfile repair phase")
parser.add_argument("--skip-install-guide", action="store_true", help="Skip the INSTALL.md generation phase")
parser.add_argument(
    "--variant",
    default="flat_baseline",
    choices=["flat_baseline", "exploration", "synthesis", "validation", "full_system", "ab_prev_attempt_ctx_on", "ab_prev_attempt_ctx_off", "ab_stateful_tree_on", "ab_stateful_tree_off", "ab_retrieval_bm25", "ab_retrieval_neural_embedding", "ab_retrieval_one_shot_fingerprint", "ab_retrieval_iterative_react", "one_shot_direct"],
    help="Pipeline variant for ablation runs.",
)
parser.add_argument(
    "--agent-config",
    default="",
    help="Optional JSON file with runtime architecture controls (phases/retrieval/ReAct/stateful repair).",
)
parser.add_argument(
    "--retrieval-strategy",
    default="",
    choices=["iterative_react", "bm25", "neural_embedding", "one_shot_fingerprint"],
    help="Override classify retrieval strategy independently of --variant.",
)
parser.add_argument(
    "--embedding-model",
    default=os.getenv("LLM_EMBEDDING_MODEL", os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")),
    help="Embedding model passed to classify when retrieval strategy is neural_embedding.",
)
parser.add_argument(
    "--react-max-steps",
    type=int,
    default=4,
    help="Maximum retrieval iterations passed to classify for iterative_react strategy.",
)
parser.add_argument(
    "--react-max-total-files",
    type=int,
    default=24,
    help="Maximum total selected files across iterative_react steps.",
)
parser.add_argument(
    "--react-final-cap",
    type=int,
    default=12,
    help="Hard final cap on selected files after retrieval normalization/reranking.",
)
parser.add_argument(
    "--step2-token-budget",
    type=int,
    default=12000,
    help="Target token budget for Step 2 classification prompt (0 disables budget packing).",
)
parser.add_argument(
    "--synthesis-react-max-steps",
    type=int,
    default=3,
    help="Maximum L2 synthesis loop iterations passed to classify.",
)
parser.add_argument(
    "--synthesis-review-rounds",
    type=int,
    default=1,
    help="Number of L2.5 reviewer rounds to run after generator output.",
)
parser.add_argument(
    "--validation-react-max-steps",
    type=int,
    default=3,
    help="Maximum classify validation ReAct loop iterations passed to classify.",
)
parser.add_argument(
    "--synthesis-subagents-enabled",
    action=argparse.BooleanOptionalAction,
    default=True,
    help="Enable parallel synthesis sub-agent passes in classify.",
)
parser.add_argument(
    "--dockerfile-one-shot-direct",
    action=argparse.BooleanOptionalAction,
    default=None,
    help="Override one-shot-direct Dockerfile generation mode without changing other phases.",
)
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

CANONICAL_EXECUTION_BRANCH = "ablation/full-system/runtime-agent-config"


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


def append_shared_model_args(command: list[str], *, include_retry_backoff: bool = True) -> list[str]:
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
    ])
    if include_retry_backoff:
        command.extend(["--llm-retry-backoff-seconds", str(args.llm_retry_backoff_seconds)])
    if args.trace:
        command.append("--trace")
    return command


def append_repo_selection_args(command: list[str]) -> list[str]:
    command.extend(["--input-file", args.input_file])
    for repo_url in args.repo_url:
        command.extend(["--repo-url", repo_url])
    return command


def build_agent_command(
    python_executable: str,
    script_path: Path,
    *,
    include_model_args: bool = True,
    include_retry_backoff: bool = True,
) -> list[str]:
    command = [python_executable, str(script_path)]
    append_repo_selection_args(command)
    if include_model_args:
        append_shared_model_args(command, include_retry_backoff=include_retry_backoff)
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


def _resolve_agent_config_path(workspace_root: Path) -> Path | None:
    if not args.agent_config:
        return None

    candidate = Path(args.agent_config).expanduser()
    if not candidate.is_absolute():
        candidate = (workspace_root / candidate).resolve()
    return candidate


def _load_agent_config(path: Path) -> dict:
    if not path.exists():
        raise ValueError(f"agent config not found: {path}")

    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON in agent config {path}: {exc}") from exc

    if not isinstance(loaded, dict):
        raise ValueError(f"agent config root must be a JSON object: {path}")

    return loaded


def _expect_bool(value, *, key: str) -> bool:
    if isinstance(value, bool):
        return value
    raise ValueError(f"agent_config.{key} must be boolean")


def _expect_int(value, *, key: str, min_value: int = 1) -> int:
    if not isinstance(value, int):
        raise ValueError(f"agent_config.{key} must be integer")
    if value < min_value:
        raise ValueError(f"agent_config.{key} must be >= {min_value}")
    return value


def _expect_str(value, *, key: str) -> str:
    if isinstance(value, str) and value.strip():
        return value.strip()
    raise ValueError(f"agent_config.{key} must be non-empty string")


def _expect_object(value, *, key: str) -> dict:
    if isinstance(value, dict):
        return value
    raise ValueError(f"agent_config.{key} must be an object")


def _repo_name_from_url(repo_url: str) -> str:
    return repo_url.rstrip("/").split("/")[-1].replace(".git", "")


def _resolve_repo_urls_for_run(workspace_root: Path) -> list[str]:
    if args.repo_url:
        deduped: list[str] = []
        seen: set[str] = set()
        for value in args.repo_url:
            normalized = value.strip()
            if not normalized or normalized in seen:
                continue
            seen.add(normalized)
            deduped.append(normalized)
        return deduped

    input_path = Path(args.input_file)
    candidates: list[Path] = []
    if input_path.is_absolute():
        candidates.append(input_path)
    else:
        candidates.append((workspace_root / input_path).resolve())
        candidates.append((Path.cwd() / input_path).resolve())

    selected: Path | None = None
    for candidate in candidates:
        if candidate.exists():
            selected = candidate
            break

    if selected is None:
        raise ValueError(f"input file not found for user-constraint seeding: {args.input_file}")

    try:
        loaded = json.loads(selected.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON in input file {selected}: {exc}") from exc

    if not isinstance(loaded, list):
        raise ValueError(f"input file must contain a JSON array: {selected}")

    deduped: list[str] = []
    seen: set[str] = set()
    for item in loaded:
        repo_url = ""
        if isinstance(item, dict):
            repo_url = str(item.get("url", "")).strip()
        elif isinstance(item, str):
            repo_url = item.strip()
        if not repo_url or repo_url in seen:
            continue
        seen.add(repo_url)
        deduped.append(repo_url)
    return deduped


def _seed_user_constraints(repo_urls: list[str], summaries_dir: Path, user_constraints: dict) -> int:
    if not user_constraints:
        return 0

    seeded = 0
    for repo_url in repo_urls:
        repo_name = _repo_name_from_url(repo_url)
        upsert_shared_repository_state(
            repo_name,
            summaries_dir,
            repo_url=repo_url,
            stage_name="pipeline",
            stage_update={
                "user_constraints": user_constraints,
                "source": "agent_config.user_constraints",
            },
        )
        seeded += 1
    return seeded


def _apply_agent_config_overrides(agent_config: dict, phase_skips: dict[str, bool]) -> dict:
    applied: dict = {}

    architecture_cfg = agent_config.get("architecture")
    if architecture_cfg is not None:
        if not isinstance(architecture_cfg, dict):
            raise ValueError("agent_config.architecture must be an object")
        if "exploration_enabled" in architecture_cfg:
            args.arch_exploration_enabled = _expect_bool(
                architecture_cfg["exploration_enabled"],
                key="architecture.exploration_enabled",
            )
            applied.setdefault("architecture", {})["exploration_enabled"] = args.arch_exploration_enabled
        if "synthesis_enabled" in architecture_cfg:
            args.arch_synthesis_enabled = _expect_bool(
                architecture_cfg["synthesis_enabled"],
                key="architecture.synthesis_enabled",
            )
            applied.setdefault("architecture", {})["synthesis_enabled"] = args.arch_synthesis_enabled
        if "validation_enabled" in architecture_cfg:
            args.arch_validation_enabled = _expect_bool(
                architecture_cfg["validation_enabled"],
                key="architecture.validation_enabled",
            )
            applied.setdefault("architecture", {})["validation_enabled"] = args.arch_validation_enabled
        if "scratchpads_enabled" in architecture_cfg:
            args.arch_scratchpads_enabled = _expect_bool(
                architecture_cfg["scratchpads_enabled"],
                key="architecture.scratchpads_enabled",
            )
            applied.setdefault("architecture", {})["scratchpads_enabled"] = args.arch_scratchpads_enabled
        if "synthesis_subagents_enabled" in architecture_cfg:
            args.synthesis_subagents_enabled = _expect_bool(
                architecture_cfg["synthesis_subagents_enabled"],
                key="architecture.synthesis_subagents_enabled",
            )
            applied.setdefault("architecture", {})["synthesis_subagents_enabled"] = args.synthesis_subagents_enabled
        if "synthesis_react_max_steps" in architecture_cfg:
            args.synthesis_react_max_steps = _expect_int(
                architecture_cfg["synthesis_react_max_steps"],
                key="architecture.synthesis_react_max_steps",
            )
            applied.setdefault("architecture", {})["synthesis_react_max_steps"] = args.synthesis_react_max_steps
        if "synthesis_review_rounds" in architecture_cfg:
            args.synthesis_review_rounds = _expect_int(
                architecture_cfg["synthesis_review_rounds"],
                key="architecture.synthesis_review_rounds",
            )
            applied.setdefault("architecture", {})["synthesis_review_rounds"] = args.synthesis_review_rounds
        if "validation_react_max_steps" in architecture_cfg:
            args.validation_react_max_steps = _expect_int(
                architecture_cfg["validation_react_max_steps"],
                key="architecture.validation_react_max_steps",
            )
            applied.setdefault("architecture", {})["validation_react_max_steps"] = args.validation_react_max_steps

    phases_cfg = agent_config.get("phases")
    if phases_cfg is not None:
        if not isinstance(phases_cfg, dict):
            raise ValueError("agent_config.phases must be an object")
        phase_key_map = {
            "classify": "skip_classify",
            "dockerfile": "skip_dockerfile",
            "validation_gate": "skip_validation_gate",
            "repair": "skip_repair",
            "install_guide": "skip_install_guide",
        }
        for phase_name, arg_name in phase_key_map.items():
            if phase_name not in phases_cfg:
                continue
            enabled = _expect_bool(phases_cfg[phase_name], key=f"phases.{phase_name}")
            phase_skips[phase_name] = not enabled
            setattr(args, arg_name, not enabled)
            applied.setdefault("phases", {})[phase_name] = enabled

    classification_cfg = agent_config.get("classification")
    if classification_cfg is not None:
        if not isinstance(classification_cfg, dict):
            raise ValueError("agent_config.classification must be an object")

        if "retrieval_strategy" in classification_cfg:
            retrieval_strategy = _expect_str(classification_cfg["retrieval_strategy"], key="classification.retrieval_strategy")
            allowed = {"iterative_react", "bm25", "neural_embedding", "one_shot_fingerprint"}
            if retrieval_strategy not in allowed:
                raise ValueError(
                    "agent_config.classification.retrieval_strategy must be one of "
                    + ", ".join(sorted(allowed))
                )
            args.retrieval_strategy = retrieval_strategy
            applied.setdefault("classification", {})["retrieval_strategy"] = retrieval_strategy

        if "embedding_model" in classification_cfg:
            embedding_model = _expect_str(classification_cfg["embedding_model"], key="classification.embedding_model")
            args.embedding_model = embedding_model
            applied.setdefault("classification", {})["embedding_model"] = embedding_model

        react_cfg = classification_cfg.get("react")
        if react_cfg is not None:
            if not isinstance(react_cfg, dict):
                raise ValueError("agent_config.classification.react must be an object")

            if "max_steps" in react_cfg:
                args.react_max_steps = _expect_int(react_cfg["max_steps"], key="classification.react.max_steps")
                applied.setdefault("classification", {}).setdefault("react", {})["max_steps"] = args.react_max_steps
            if "max_total_files" in react_cfg:
                args.react_max_total_files = _expect_int(react_cfg["max_total_files"], key="classification.react.max_total_files")
                applied.setdefault("classification", {}).setdefault("react", {})["max_total_files"] = args.react_max_total_files
            if "final_cap" in react_cfg:
                args.react_final_cap = _expect_int(react_cfg["final_cap"], key="classification.react.final_cap")
                applied.setdefault("classification", {}).setdefault("react", {})["final_cap"] = args.react_final_cap

        if "step2_token_budget" in classification_cfg:
            args.step2_token_budget = _expect_int(
                classification_cfg["step2_token_budget"],
                key="classification.step2_token_budget",
                min_value=0,
            )
            applied.setdefault("classification", {})["step2_token_budget"] = args.step2_token_budget

    dockerfile_cfg = agent_config.get("dockerfile")
    if dockerfile_cfg is not None:
        if not isinstance(dockerfile_cfg, dict):
            raise ValueError("agent_config.dockerfile must be an object")
        if "one_shot_direct" in dockerfile_cfg:
            args.dockerfile_one_shot_direct = _expect_bool(
                dockerfile_cfg["one_shot_direct"],
                key="dockerfile.one_shot_direct",
            )
            applied.setdefault("dockerfile", {})["one_shot_direct"] = args.dockerfile_one_shot_direct

    repair_cfg = agent_config.get("repair")
    if repair_cfg is not None:
        if not isinstance(repair_cfg, dict):
            raise ValueError("agent_config.repair must be an object")

        if "stateful_repair" in repair_cfg:
            args.stateful_repair = _expect_bool(repair_cfg["stateful_repair"], key="repair.stateful_repair")
            applied.setdefault("repair", {})["stateful_repair"] = args.stateful_repair
        if "stateful_repair_tree" in repair_cfg:
            args.stateful_repair_tree = _expect_bool(repair_cfg["stateful_repair_tree"], key="repair.stateful_repair_tree")
            applied.setdefault("repair", {})["stateful_repair_tree"] = args.stateful_repair_tree
        if "history_window" in repair_cfg:
            args.stateful_history_window = _expect_int(repair_cfg["history_window"], key="repair.history_window")
            applied.setdefault("repair", {})["history_window"] = args.stateful_history_window
        if "history_max_chars" in repair_cfg:
            args.stateful_history_max_chars = _expect_int(
                repair_cfg["history_max_chars"],
                key="repair.history_max_chars",
            )
            applied.setdefault("repair", {})["history_max_chars"] = args.stateful_history_max_chars
        if "tree_max_chars" in repair_cfg:
            args.stateful_tree_max_chars = _expect_int(repair_cfg["tree_max_chars"], key="repair.tree_max_chars")
            applied.setdefault("repair", {})["tree_max_chars"] = args.stateful_tree_max_chars
        if "tree_max_children" in repair_cfg:
            args.stateful_tree_max_children = _expect_int(
                repair_cfg["tree_max_children"],
                key="repair.tree_max_children",
            )
            applied.setdefault("repair", {})["tree_max_children"] = args.stateful_tree_max_children

    user_constraints_cfg = agent_config.get("user_constraints")
    if user_constraints_cfg is not None:
        applied["user_constraints"] = _expect_object(
            user_constraints_cfg,
            key="user_constraints",
        )

    return applied


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
        "--retrieval-strategy", resolve_classify_retrieval_strategy(),
        "--embedding-model", args.embedding_model,
        "--react-max-steps", str(args.react_max_steps),
        "--react-max-total-files", str(args.react_max_total_files),
        "--react-final-cap", str(args.react_final_cap),
        "--step2-token-budget", str(args.step2_token_budget),
        "--synthesis-react-max-steps", str(args.synthesis_react_max_steps),
        "--synthesis-review-rounds", str(args.synthesis_review_rounds),
        "--validation-react-max-steps", str(args.validation_react_max_steps),
        "--results-dir", args.results_dir,
        "--summaries-dir", args.summaries_dir,
        "--scratchpad-dir", args.summaries_dir,
        "--repos-dir", args.repos_dir,
        "--analysis-dir", args.analysis_dir,
    ])
    command.append("--synthesis-subagents-enabled" if args.synthesis_subagents_enabled else "--no-synthesis-subagents-enabled")
    command.append("--exploration-enabled" if args.arch_exploration_enabled else "--no-exploration-enabled")
    command.append("--synthesis-enabled" if args.arch_synthesis_enabled else "--no-synthesis-enabled")
    command.append("--validation-enabled" if args.arch_validation_enabled else "--no-validation-enabled")
    command.append("--scratchpads-enabled" if args.arch_scratchpads_enabled else "--no-scratchpads-enabled")
    command.append("--no-analysis")
    return command


def build_dockerfile_command(python_executable: str, script_path: Path) -> list[str]:
    command = build_agent_command(python_executable, script_path)
    if args.force:
        command.append("--force")
    one_shot_direct = args.variant == "one_shot_direct"
    if args.dockerfile_one_shot_direct is not None:
        one_shot_direct = args.dockerfile_one_shot_direct
    if one_shot_direct:
        command.append("--one-shot-direct")
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
    command = build_agent_command(python_executable, script_path, include_retry_backoff=False)
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


def build_validation_gate_command(python_executable: str, script_path: Path) -> list[str]:
    command = build_agent_command(python_executable, script_path, include_model_args=False)
    if args.skip_hadolint:
        command.append("--skip-hadolint-gate")
    if args.trace:
        command.append("--trace")
    command.extend([
        "--summaries-dir", args.summaries_dir,
        "--dockerfiles-dir", args.dockerfiles_dir,
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


def resolve_phase_skips() -> dict[str, bool]:
    skips = {
        "classify": args.skip_classify,
        "dockerfile": args.skip_dockerfile,
        "validation_gate": args.skip_validation_gate,
        "repair": args.skip_repair,
        "install_guide": args.skip_install_guide,
    }
    if args.variant == "one_shot_direct":
        skips["classify"] = True
        skips["validation_gate"] = True
        skips["repair"] = True
        skips["install_guide"] = True
    return skips


def _expected_validation_gate_enabled_for_variant(variant: str) -> bool | None:
    contract = {
        "one_shot_direct": False,
        "flat_baseline": False,
        "exploration": False,
        "synthesis": False,
        "validation": True,
        "full_system": True,
        "ab_retrieval_bm25": True,
        "ab_retrieval_neural_embedding": True,
        "ab_retrieval_one_shot_fingerprint": True,
        "ab_retrieval_iterative_react": True,
        "ab_prev_attempt_ctx_on": True,
        "ab_prev_attempt_ctx_off": True,
        "ab_stateful_tree_on": True,
        "ab_stateful_tree_off": True,
    }
    return contract.get(variant)


def enforce_variant_phase_contract(phase_skips: dict[str, bool]) -> bool | None:
    expected_enabled = _expected_validation_gate_enabled_for_variant(args.variant)
    if expected_enabled is None:
        return None

    current_enabled = not bool(phase_skips.get("validation_gate", False))
    if current_enabled != expected_enabled:
        log_info(
            "Variant %s selected: forcing post-generation validation gate %s for ablation-contract consistency."
            % (args.variant, "ON" if expected_enabled else "OFF")
        )
        phase_skips["validation_gate"] = not expected_enabled
        args.skip_validation_gate = not expected_enabled

    effective_enabled = not bool(phase_skips.get("validation_gate", False))
    if effective_enabled != expected_enabled:
        raise ValueError(
            "variant/phase contract violation: variant=%s expected validation_gate=%s but got %s"
            % (args.variant, expected_enabled, effective_enabled)
        )
    return expected_enabled


def resolve_classify_retrieval_strategy() -> str:
    retrieval_overrides = {
        "ab_retrieval_bm25": "bm25",
        "ab_retrieval_neural_embedding": "neural_embedding",
        "ab_retrieval_one_shot_fingerprint": "one_shot_fingerprint",
        "ab_retrieval_iterative_react": "iterative_react",
    }
    if args.retrieval_strategy:
        return args.retrieval_strategy
    if args.variant in retrieval_overrides:
        return retrieval_overrides[args.variant]
    return "iterative_react"


VARIANT_POLICY_TABLE: dict[str, dict] = {
    "ab_retrieval_iterative_react": {
        "phase2_anchor": False,
        "repo_context_source": "iterative_react_retrieval",
        "classification_required": True,
        "repair_enabled": True,
        "exploration_enabled": True,
        "synthesis_enabled": True,
        "validation_enabled": True,
        "scratchpads_enabled": True,
        "retrieval_strategy": "iterative_react",
    },
    "ab_retrieval_one_shot_fingerprint": {
        "phase2_anchor": False,
        "repo_context_source": "one_shot_fingerprint_retrieval",
        "classification_required": True,
        "repair_enabled": True,
        "exploration_enabled": True,
        "synthesis_enabled": True,
        "validation_enabled": True,
        "scratchpads_enabled": True,
        "retrieval_strategy": "one_shot_fingerprint",
    },
    "ab_retrieval_neural_embedding": {
        "phase2_anchor": False,
        "repo_context_source": "neural_embedding_retrieval",
        "classification_required": True,
        "repair_enabled": True,
        "exploration_enabled": True,
        "synthesis_enabled": True,
        "validation_enabled": True,
        "scratchpads_enabled": True,
        "retrieval_strategy": "neural_embedding",
    },
    "ab_retrieval_bm25": {
        "phase2_anchor": False,
        "repo_context_source": "bm25_retrieval",
        "classification_required": True,
        "repair_enabled": True,
        "exploration_enabled": True,
        "synthesis_enabled": True,
        "validation_enabled": True,
        "scratchpads_enabled": True,
        "retrieval_strategy": "bm25",
    },
    "ab_stateful_tree_off": {
        "phase2_anchor": False,
        "repo_context_source": "iterative_exploration_synthesis_validation_scratchpads",
        "classification_required": True,
        "repair_enabled": True,
        "exploration_enabled": True,
        "synthesis_enabled": True,
        "validation_enabled": True,
        "scratchpads_enabled": True,
        "retrieval_strategy": "iterative_exploration",
        "stateful_repair_enabled": True,
        "stateful_tree_enabled": False,
        "prev_attempt_context_enabled": True,
    },
    "ab_stateful_tree_on": {
        "phase2_anchor": False,
        "repo_context_source": "iterative_exploration_synthesis_validation_scratchpads",
        "classification_required": True,
        "repair_enabled": True,
        "exploration_enabled": True,
        "synthesis_enabled": True,
        "validation_enabled": True,
        "scratchpads_enabled": True,
        "retrieval_strategy": "iterative_exploration",
        "stateful_repair_enabled": True,
        "stateful_tree_enabled": True,
        "prev_attempt_context_enabled": True,
    },
    "ab_prev_attempt_ctx_on": {
        "phase2_anchor": False,
        "repo_context_source": "iterative_exploration_synthesis_validation_scratchpads",
        "classification_required": True,
        "repair_enabled": True,
        "exploration_enabled": True,
        "synthesis_enabled": True,
        "validation_enabled": True,
        "scratchpads_enabled": True,
        "retrieval_strategy": "iterative_exploration",
        "stateful_repair_enabled": True,
        "stateful_tree_enabled": False,
        "prev_attempt_context_enabled": True,
    },
    "ab_prev_attempt_ctx_off": {
        "phase2_anchor": False,
        "repo_context_source": "iterative_exploration_synthesis_validation_scratchpads",
        "classification_required": True,
        "repair_enabled": True,
        "exploration_enabled": True,
        "synthesis_enabled": True,
        "validation_enabled": True,
        "scratchpads_enabled": True,
        "retrieval_strategy": "iterative_exploration",
        "stateful_repair_enabled": True,
        "stateful_tree_enabled": False,
        "prev_attempt_context_enabled": False,
    },
    "full_system": {
        "phase2_anchor": False,
        "repo_context_source": "iterative_exploration_synthesis_validation_scratchpads",
        "classification_required": True,
        "repair_enabled": True,
        "exploration_enabled": True,
        "synthesis_enabled": True,
        "validation_enabled": True,
        "scratchpads_enabled": True,
        "retrieval_strategy": "iterative_exploration",
    },
    "validation": {
        "phase2_anchor": False,
        "repo_context_source": "iterative_exploration_synthesis_validation",
        "classification_required": True,
        "repair_enabled": True,
        "exploration_enabled": True,
        "synthesis_enabled": True,
        "validation_enabled": True,
        "scratchpads_enabled": False,
        "retrieval_strategy": "iterative_exploration",
    },
    "synthesis": {
        "phase2_anchor": False,
        "repo_context_source": "iterative_exploration_plus_synthesis",
        "classification_required": True,
        "repair_enabled": True,
        "exploration_enabled": True,
        "synthesis_enabled": True,
        "validation_enabled": False,
        "scratchpads_enabled": False,
        "retrieval_strategy": "iterative_exploration",
    },
    "exploration": {
        "phase2_anchor": False,
        "repo_context_source": "iterative_exploration",
        "classification_required": True,
        "repair_enabled": True,
        "exploration_enabled": True,
        "synthesis_enabled": False,
        "validation_enabled": False,
        "scratchpads_enabled": False,
        "retrieval_strategy": "iterative_exploration",
    },
    "one_shot_direct": {
        "phase2_anchor": False,
        "repo_context_source": "static_fingerprint_only",
        "classification_required": False,
        "repair_enabled": False,
        "exploration_enabled": False,
        "synthesis_enabled": False,
        "validation_enabled": False,
        "scratchpads_enabled": False,
        "retrieval_strategy": "none",
    },
}


def resolve_variant_policy() -> dict:
    if args.variant in VARIANT_POLICY_TABLE:
        return VARIANT_POLICY_TABLE[args.variant]

    return {
        "phase2_anchor": True,
        "repo_context_source": "staged_pipeline",
        "classification_required": True,
        "repair_enabled": True,
        "exploration_enabled": False,
        "synthesis_enabled": False,
        "validation_enabled": False,
        "scratchpads_enabled": False,
        "retrieval_strategy": "baseline_one_shot_fingerprint",
    }


def apply_stateful_contract_by_variant() -> None:
    contract: dict[str, tuple[bool, bool]] = {
        "flat_baseline": (False, False),
        "exploration": (False, False),
        "synthesis": (False, False),
        "validation": (False, False),
        "full_system": (False, False),
        "ab_retrieval_bm25": (False, False),
        "ab_retrieval_neural_embedding": (False, False),
        "ab_retrieval_one_shot_fingerprint": (False, False),
        "ab_retrieval_iterative_react": (False, False),
        "ab_prev_attempt_ctx_on": (True, False),
        "ab_prev_attempt_ctx_off": (True, False),
        "ab_stateful_tree_on": (True, True),
        "ab_stateful_tree_off": (True, False),
    }
    expected = contract.get(args.variant)
    if expected is None:
        return
    expected_stateful, expected_tree = expected

    if args.stateful_repair != expected_stateful or args.stateful_repair_tree != expected_tree:
        log_info(
            "Variant %s selected: forcing stateful_repair=%s stateful_repair_tree=%s for ablation-contract consistency."
            % (args.variant, expected_stateful, expected_tree)
        )
    args.stateful_repair = expected_stateful
    args.stateful_repair_tree = expected_tree


def main() -> int:
    run_id = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    pipeline_started_ts = time.perf_counter()
    pipeline_started_at = utc_now()
    src_root = Path(__file__).resolve().parents[2]
    workspace_root = _resolve_workspace_root(src_root)
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
    runtime_config_lock_path = run_dir / "runtime-config-lock.yaml"

    classify_script = src_root / "stages" / "stage_1_repository_installation_analysis" / "agent_classify.py"
    dockerfile_script = src_root / "stages" / "stage_2_dockerfile_generation" / "agent_dockerfile.py"
    repair_script = src_root / "stages" / "stage_3_iterative_dockerfile_repair" / "agent_dockerfile_repair.py"
    validation_gate_script = src_root / "stages" / "stage_2_dockerfile_generation" / "agent_validation_gate.py"
    install_guide_script = src_root / "stages" / "stage_4_install_guide" / "agent_install_guide.py"
    analysis_script = src_dir / "parse_results.py"

    phase_skips = resolve_phase_skips()
    apply_stateful_contract_by_variant()

    base_variant_policy = resolve_variant_policy()
    args.arch_exploration_enabled = bool(base_variant_policy.get("exploration_enabled", True))
    args.arch_synthesis_enabled = bool(base_variant_policy.get("synthesis_enabled", True))
    args.arch_validation_enabled = bool(base_variant_policy.get("validation_enabled", True))
    args.arch_scratchpads_enabled = bool(base_variant_policy.get("scratchpads_enabled", True))

    agent_config_path = _resolve_agent_config_path(workspace_root)
    agent_config_applied: dict = {}
    if agent_config_path is not None:
        try:
            agent_config = _load_agent_config(agent_config_path)
            agent_config_applied = _apply_agent_config_overrides(agent_config, phase_skips)
            log_info(f"Applied runtime agent config: {agent_config_path}")
        except ValueError as error:
            log_error(str(error))
            return 1

    try:
        expected_validation_gate_enabled = enforce_variant_phase_contract(phase_skips)
    except ValueError as error:
        log_error(str(error))
        return 1

    if args.print_summary:
        print_planned_summary()
        return 0

    phases_selected = not (
        phase_skips["classify"]
        and phase_skips["dockerfile"]
        and phase_skips["validation_gate"]
        and phase_skips["repair"]
        and phase_skips["install_guide"]
    )
    if not phases_selected and not args.run_analysis:
        log_error("Nothing to do. All pipeline phases were skipped and --run-analysis was not set.")
        return 1

    if phase_skips["classify"] and args.run_analysis:
        log_info("Classification is skipped: disabling --run-analysis because parse_results depends on classify outputs.")
        args.run_analysis = False

    variant_policy = resolve_variant_policy()

    summary: dict = {
        "run_id": run_id,
        "started_at": pipeline_started_at,
        "ended_at": None,
        "duration_seconds": None,
        "status": "failed",
        "canonical_execution_branch": CANONICAL_EXECUTION_BRANCH,
        "trace": {
            "enabled": args.trace,
            "forwarded_to_child_agents": args.trace,
        },
        "variant": args.variant,
        "variant_policy": variant_policy,
        "agent_config": {
            "path": str(agent_config_path) if agent_config_path is not None else None,
            "applied_overrides": agent_config_applied,
        },
        "effective_runtime_controls": {
            "phases": {
                "classify": not phase_skips["classify"],
                "dockerfile": not phase_skips["dockerfile"],
                "validation_gate": not phase_skips["validation_gate"],
                "repair": not phase_skips["repair"],
                "install_guide": not phase_skips["install_guide"],
            },
            "phase_contract": {
                "validation_gate_expected_for_variant": expected_validation_gate_enabled,
                "validation_gate_contract_enforced": expected_validation_gate_enabled is not None,
            },
            "classification": {
                "retrieval_strategy": resolve_classify_retrieval_strategy(),
                "embedding_model": args.embedding_model,
                "architecture": {
                    "exploration_enabled": args.arch_exploration_enabled,
                    "synthesis_enabled": args.arch_synthesis_enabled,
                    "validation_enabled": args.arch_validation_enabled,
                    "scratchpads_enabled": args.arch_scratchpads_enabled,
                    "synthesis_subagents_enabled": args.synthesis_subagents_enabled,
                    "synthesis_react_max_steps": args.synthesis_react_max_steps,
                    "synthesis_review_rounds": args.synthesis_review_rounds,
                    "validation_react_max_steps": args.validation_react_max_steps,
                },
                "artifact_patterns": {
                    "exploration": str(Path(args.summaries_dir) / "<repo>.exploration.yaml"),
                    "synthesis": str(Path(args.summaries_dir) / "<repo>.synthesis.yaml"),
                    "validation": str(Path(args.summaries_dir) / "<repo>.validation.yaml"),
                    "post_generation_validation": str(Path(args.summaries_dir) / "<repo>.postgen-validation.yaml"),
                    "scratchpad": str(Path(args.summaries_dir) / "<repo>.architecture-scratchpad.yaml"),
                },
                "react": {
                    "max_steps": args.react_max_steps,
                    "max_total_files": args.react_max_total_files,
                    "final_cap": args.react_final_cap,
                },
                "step2_token_budget": args.step2_token_budget,
            },
            "dockerfile": {
                "one_shot_direct": (
                    args.variant == "one_shot_direct"
                    if args.dockerfile_one_shot_direct is None
                    else args.dockerfile_one_shot_direct
                ),
            },
            "repair": {
                "stateful_repair": args.stateful_repair,
                "stateful_repair_tree": args.stateful_repair_tree,
                "history_window": args.stateful_history_window,
                "history_max_chars": args.stateful_history_max_chars,
                "tree_max_chars": args.stateful_tree_max_chars,
                "tree_max_children": args.stateful_tree_max_children,
            },
            "user_constraints": agent_config_applied.get("user_constraints", {}),
        },
        "prompt_profile": prompt_profile_metadata(PROMPT_PROFILE, EFFECTIVE_TEMPERATURE),
        "paths": {
            "run_dir": str(run_dir),
            "reports_dir": str(reports_dir),
            "run_logs_dir": str(run_logs_dir),
            "summary_path": str(summary_path),
            "runtime_config_lock_path": str(runtime_config_lock_path),
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

    runtime_config_lock = {
        "generated_at": utc_now(),
        "canonical_execution_branch": CANONICAL_EXECUTION_BRANCH,
        "variant": args.variant,
        "variant_policy": variant_policy,
        "prompt_profile": prompt_profile_metadata(PROMPT_PROFILE, EFFECTIVE_TEMPERATURE),
        "override_sources": {
            "cli_flags": {
                "skip_classify": args.skip_classify,
                "skip_dockerfile": args.skip_dockerfile,
                "skip_validation_gate": args.skip_validation_gate,
                "skip_repair": args.skip_repair,
                "skip_install_guide": args.skip_install_guide,
                "retrieval_strategy": args.retrieval_strategy,
                "stateful_repair": args.stateful_repair,
                "stateful_repair_tree": args.stateful_repair_tree,
            },
            "agent_config_path": str(agent_config_path) if agent_config_path is not None else None,
            "agent_config_applied": agent_config_applied,
            "validation_gate_contract_expected": expected_validation_gate_enabled,
        },
        "effective_runtime_controls": summary["effective_runtime_controls"],
    }
    write_summary(runtime_config_lock_path, runtime_config_lock)
    log_info(f"Runtime config lock written to {runtime_config_lock_path}")

    try:
        repo_urls_for_run = _resolve_repo_urls_for_run(workspace_root)
    except ValueError as error:
        log_error(str(error))
        return 1

    constraints_seeded = _seed_user_constraints(
        repo_urls_for_run,
        Path(args.summaries_dir),
        agent_config_applied.get("user_constraints", {}),
    )
    if constraints_seeded:
        log_info(f"Seeded user constraints into shared state for {constraints_seeded} repositories")

    try:
        if not phase_skips["classify"]:
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

        if not phase_skips["dockerfile"]:
            summary["phases"].append(
                run_step(
                    "dockerfile generation",
                    build_dockerfile_command(python_executable, dockerfile_script),
                    run_logs_dir / "dockerfile-generation.log",
                )
            )

        if not phase_skips["validation_gate"]:
            summary["phases"].append(
                run_step(
                    "post-generation validation gate",
                    build_validation_gate_command(python_executable, validation_gate_script),
                    run_logs_dir / "post-generation-validation-gate.log",
                )
            )

        if not phase_skips["repair"]:
            summary["phases"].append(
                run_step(
                    "dockerfile repair",
                    build_repair_command(python_executable, repair_script),
                    run_logs_dir / "dockerfile-repair.log",
                )
            )

        if not phase_skips["install_guide"]:
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

    if not phase_skips["repair"]:
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