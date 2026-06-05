import argparse
import asyncio
from pathlib import Path

import yaml
from tqdm import tqdm

try:
    from RepoBuilderAgent.src.core.log_utils import log_error, log_info, log_warn, set_tqdm_bar, set_trace_enabled
    from RepoBuilderAgent.src.core.common import (
        load_repo_urls,
        read_yaml_file,
        repo_name_from_url,
        should_use_progress,
        update_progress,
        upsert_shared_repository_state,
        validate_dockerfile_syntax,
    )
except ImportError:
    from core.log_utils import log_error, log_info, log_warn, set_tqdm_bar, set_trace_enabled
    from core.common import (
        load_repo_urls,
        read_yaml_file,
        repo_name_from_url,
        should_use_progress,
        update_progress,
        upsert_shared_repository_state,
        validate_dockerfile_syntax,
    )


parser = argparse.ArgumentParser(
    description="Run post-generation validation gate between Dockerfile generation and repair."
)
parser.add_argument("--input-file", default="repos.json", help="Path to input file containing repository URLs")
parser.add_argument(
    "--repo-url",
    action="append",
    default=[],
    help="Validate a specific repository URL (can be passed multiple times). Overrides --input-file when provided.",
)
parser.add_argument("--trace", action="store_true", help="Enable verbose trace logs")
parser.add_argument("--skip-hadolint-gate", action="store_true", help="Skip hadolint syntax check in validation gate")
parser.add_argument("--summaries-dir", default="summaries", help="Directory containing repository summary files")
parser.add_argument("--dockerfiles-dir", default="dockerfiles", help="Directory containing generated Dockerfiles")
args = parser.parse_args()

set_trace_enabled(args.trace)
sem = asyncio.Semaphore(8)


def _evaluate_gate(
    *,
    repo_url: str,
    repo_name: str,
    dockerfile_path: Path,
    verify_command_path: Path,
    classify_validation_artifact: dict | None,
    hadolint_result: tuple[bool, str] | None,
) -> dict:
    checks: dict[str, dict[str, str]] = {}

    checks["dockerfile_exists"] = {
        "status": "pass" if dockerfile_path.exists() else "fail",
        "detail": f"Dockerfile present at {dockerfile_path}" if dockerfile_path.exists() else f"Missing Dockerfile at {dockerfile_path}",
    }

    verify_exists = verify_command_path.exists()
    verify_non_empty = verify_exists and bool(verify_command_path.read_text(encoding="utf-8").strip())
    if verify_non_empty:
        verify_status = "pass"
        verify_detail = f"Verification command found at {verify_command_path}."
    elif verify_exists:
        verify_status = "warn"
        verify_detail = f"Verification command file exists but is empty at {verify_command_path}."
    else:
        verify_status = "warn"
        verify_detail = f"No generated verification command found at {verify_command_path}; downstream default command may be used."
    checks["verify_command_ready"] = {"status": verify_status, "detail": verify_detail}

    if hadolint_result is not None:
        hadolint_ok, hadolint_error = hadolint_result
        checks["dockerfile_hadolint"] = {
            "status": "pass" if hadolint_ok else "warn",
            "detail": "Dockerfile passed hadolint validation."
            if hadolint_ok
            else f"Hadolint reported issues: {(hadolint_error or '').strip()[:300]}",
        }

    classify_warn_count = 0
    classify_fail_count = 0
    if isinstance(classify_validation_artifact, dict):
        checks_payload = classify_validation_artifact.get("checks")
        if isinstance(checks_payload, dict):
            for payload in checks_payload.values():
                if not isinstance(payload, dict):
                    continue
                status = str(payload.get("status", "")).strip().lower()
                if status == "warn":
                    classify_warn_count += 1
                elif status == "fail":
                    classify_fail_count += 1
    checks["classify_validation_signal"] = {
        "status": "warn" if classify_fail_count or classify_warn_count else "pass",
        "detail": (
            f"Classify validation reported fail={classify_fail_count}, warn={classify_warn_count}."
            if (classify_fail_count or classify_warn_count)
            else "Classify validation reported no warnings/failures."
        ),
    }

    hard_fail = any(entry.get("status") == "fail" for entry in checks.values())
    decision = {
        "run_repair": not hard_fail,
        "reason": "gate_open" if not hard_fail else "hard_fail_in_post_generation_gate",
    }

    return {
        "schema_version": "1.0",
        "repo": repo_url,
        "repo_name": repo_name,
        "stage": "post_generation_validation_gate",
        "checks": checks,
        "decision": decision,
    }


async def validate_repository(repo_url: str, summaries_dir: Path, dockerfiles_dir: Path, progress_state: dict) -> None:
    async with sem:
        repo_name = repo_name_from_url(repo_url)
        dockerfile_path = dockerfiles_dir / f"{repo_name}.Dockerfile"
        verify_command_path = dockerfiles_dir / f"{repo_name}.verify-command"
        output_path = summaries_dir / f"{repo_name}.postgen-validation.yaml"

        classify_validation_artifact = read_yaml_file(summaries_dir / f"{repo_name}.validation.yaml")

        hadolint_result: tuple[bool, str] | None = None
        if not args.skip_hadolint_gate and dockerfile_path.exists():
            hadolint_result = await validate_dockerfile_syntax(dockerfile_path, repo_name)

        gate_artifact = _evaluate_gate(
            repo_url=repo_url,
            repo_name=repo_name,
            dockerfile_path=dockerfile_path,
            verify_command_path=verify_command_path,
            classify_validation_artifact=classify_validation_artifact,
            hadolint_result=hadolint_result,
        )

        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as output_file:
            yaml.dump(gate_artifact, output_file, sort_keys=False, allow_unicode=True)

        upsert_shared_repository_state(
            repo_name,
            summaries_dir,
            repo_url=repo_url,
            stage_name="validation_gate",
            stage_update={
                "status": "completed",
                "artifact_path": str(output_path),
                "decision": gate_artifact.get("decision", {}),
                "warn_checks": [
                    key
                    for key, value in gate_artifact.get("checks", {}).items()
                    if isinstance(value, dict) and value.get("status") == "warn"
                ],
                "fail_checks": [
                    key
                    for key, value in gate_artifact.get("checks", {}).items()
                    if isinstance(value, dict) and value.get("status") == "fail"
                ],
            },
        )

        log_info(
            f"Validation gate for {repo_name}: run_repair={gate_artifact['decision']['run_repair']} "
            f"({gate_artifact['decision']['reason']})"
        )
        await update_progress(progress_state, repo_name)


async def main() -> None:
    repos = load_repo_urls(args.input_file, args.repo_url)
    if not repos:
        log_error("No repositories to process. Provide --repo-url or a non-empty --input-file.")
        return

    workspace_root = Path(args.input_file).parent
    summaries_dir = workspace_root / args.summaries_dir
    dockerfiles_dir = workspace_root / args.dockerfiles_dir
    summaries_dir.mkdir(parents=True, exist_ok=True)

    progress_bar = None
    if should_use_progress(len(repos), args.trace):
        progress_bar = tqdm(total=len(repos), desc="Validation gate", unit="repo", dynamic_ncols=True)

    progress_state = {
        "lock": asyncio.Lock(),
        "bar": progress_bar,
    }
    set_tqdm_bar(progress_state["bar"])

    tasks = [
        validate_repository(repo, summaries_dir, dockerfiles_dir, progress_state)
        for repo in repos
    ]

    try:
        await asyncio.gather(*tasks, return_exceptions=True)
    finally:
        if progress_state["bar"] is not None:
            progress_state["bar"].close()
        set_tqdm_bar(None)


if __name__ == "__main__":
    asyncio.run(main())
