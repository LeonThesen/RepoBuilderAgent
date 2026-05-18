import argparse
import asyncio
import json
import os
import re
import shutil
import ssl
from pathlib import Path

import httpx
from openai import APIError, APITimeoutError, AsyncOpenAI
from tqdm import tqdm

try:
    from RepoBuilderAgent.src.config import OPENAI_API_KEY, OPENAI_BASE_URL, OPENAI_MODEL
    from RepoBuilderAgent.src.log_utils import log_error, log_info, log_trace, log_warn, set_tqdm_bar, set_trace_enabled
    from RepoBuilderAgent.src.common import (
        chat_completion_with_retries,
        ensure_repo_checkout,
        finalize_llm_metrics,
        init_llm_metrics,
        inject_ca_cert_into_dockerfile,
        load_repo_urls,
        load_summary,
        prompt_path,
        read_yaml_file,
        render_yaml,
        repo_name_from_url,
        should_use_progress,
        update_progress,
        validate_dockerfile_syntax,
    )
except ImportError:
    # Fallback for direct script execution from RepoBuilderAgent/src
    import config as _config
    from log_utils import log_error, log_info, log_trace, log_warn, set_tqdm_bar, set_trace_enabled
    from common import (
        chat_completion_with_retries,
        ensure_repo_checkout,
        finalize_llm_metrics,
        init_llm_metrics,
        inject_ca_cert_into_dockerfile,
        load_repo_urls,
        load_summary,
        prompt_path,
        read_yaml_file,
        render_yaml,
        repo_name_from_url,
        should_use_progress,
        update_progress,
        validate_dockerfile_syntax,
    )

    OPENAI_API_KEY = getattr(_config, "OPENAI_API_KEY", "")
    OPENAI_BASE_URL = getattr(_config, "OPENAI_BASE_URL", "https://api.openai.com/v1")
    OPENAI_MODEL = getattr(_config, "OPENAI_MODEL", "gpt-4o")


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
parser.add_argument("--temperature", type=float, default=0.0, help="Temperature for the model")
parser.add_argument("--timeout", type=int, default=120, help="Timeout for API requests in seconds")
parser.add_argument("--llm-max-retries", type=int, default=2, help="Maximum retries for transient LLM timeouts and retryable API errors")
parser.add_argument("--llm-retry-backoff-seconds", type=float, default=2.0, help="Base exponential backoff delay in seconds for LLM retries")
parser.add_argument("--repair-timeout", type=int, default=240, help="Timeout for Dockerfile repair LLM calls in seconds")
parser.add_argument("--verify-repair-timeout", type=int, default=180, help="Timeout for verification-command repair LLM calls in seconds")
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
parser.add_argument("--verify-timeout", type=int, default=30, help="Timeout in seconds for build verification container execution")
parser.add_argument("--force", action="store_true", help="Re-run repair even if a successful report.yaml already exists")
args = parser.parse_args()


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
    PROMPT_TEMPLATE = prompt_file.read()

with open(prompt_path("PROMPT_BUILD_VERIFICATION.md"), "r", encoding="utf-8") as prompt_file:
    VERIFY_PROMPT_TEMPLATE = prompt_file.read()

sem = asyncio.Semaphore(1)

set_trace_enabled(args.trace)


# Files/directories to remove from the repo before docker build so the build
# context is free of documentation and CI/CD configuration that can't affect
# an installation and may only inflate the context or mislead the build.
# Intentionally exhaustive: agents must not be able to read pre-existing docs
# or build instructions from the repo context.
_DELETE_DOCS_EXTENSIONS: tuple[str, ...] = (
    # Markup documentation
    ".md", ".rst", ".adoc", ".asciidoc", ".textile", ".wiki",
    # Office / print documents
    ".pdf", ".doc", ".docx", ".odt", ".rtf",
    # Other doc formats (avoid man-page style sources like .1-.8 that can be build inputs)
    ".tex", ".pod", ".man",
    # Notebook / presentation
    ".ipynb", ".pptx", ".ppt",
)
_DELETE_DOCS_FILE_NAMES: frozenset[str] = frozenset({
    # CI/CD pipeline files
    "Jenkinsfile",
    ".travis.yml", ".travis.yaml",
    ".gitlab-ci.yml", ".gitlab-ci.yaml",
    "appveyor.yml", "appveyor.yaml",
    "azure-pipelines.yml", "azure-pipelines.yaml",
    ".drone.yml", ".drone.yaml",
    "bitbucket-pipelines.yml", "bitbucket-pipelines.yaml",
    "CODEOWNERS",
})
_DELETE_DOCS_DIR_NAMES: frozenset[str] = frozenset({
    # CI/CD directories
    ".github", ".gitlab", ".circleci", ".buildkite", ".drone",
    ".woodpecker", ".jenkins", ".azure-pipelines",
    # Documentation directories
    "docs", "doc", "documentation",
    "website", "site", "gh-pages",
    "wiki",
    "javadoc", "apidoc", "apidocs",
    "man", "manpages",
    "sphinx", "docsrc",
})


def delete_docs_build_context(repo_path: Path, repo_name: str) -> None:
    """Remove documentation and CI/CD files from repo_path before building."""
    log_info(f"[delete-docs {repo_name}] Starting docs/CI deletion in {repo_path}")

    removed_files: list[str] = []
    removed_dirs: list[str] = []

    for item in list(repo_path.rglob("*")):
        if not item.exists():
            # Already deleted as part of a parent directory removal
            continue

        if item.is_dir():
            if item.name in _DELETE_DOCS_DIR_NAMES:
                rel = item.relative_to(repo_path)
                log_trace(f"[delete-docs {repo_name}] Removing CI/CD directory: {rel}")
                shutil.rmtree(item)
                removed_dirs.append(str(rel))
        elif item.is_file():
            if item.suffix.lower() in _DELETE_DOCS_EXTENSIONS:
                rel = item.relative_to(repo_path)
                log_trace(f"[delete-docs {repo_name}] Removing doc file: {rel}")
                item.unlink()
                removed_files.append(str(rel))
            elif item.name in _DELETE_DOCS_FILE_NAMES:
                rel = item.relative_to(repo_path)
                log_trace(f"[delete-docs {repo_name}] Removing CI/CD file: {rel}")
                item.unlink()
                removed_files.append(str(rel))

    log_info(
        f"[delete-docs {repo_name}] Done — removed {len(removed_files)} file(s), "
        f"{len(removed_dirs)} director{'ies' if len(removed_dirs) != 1 else 'y'}"
    )



def extract_dockerfile(raw: str) -> str:
    match = re.search(r"```(?:dockerfile)?\n(.*?)```", raw, re.DOTALL | re.IGNORECASE)
    content = match.group(1) if match else raw
    return content.strip() + "\n"


def get_base_template(classification: dict) -> str:
    """Select and load the appropriate base template based on programming language."""
    subrepo_templates_dir = Path(__file__).resolve().parent.parent / "templates"

    languages = classification.get("categories", {}).get("programming_language", {}).get("value", [])
    if not languages:
        log_warn("No programming language detected in classification; defaulting to C template")
        template_name = "Dockerfile.base-c"
    else:
        lang = languages[0].lower()
        if "python" in lang:
            template_name = "Dockerfile.base-python"
        elif "c++" in lang or "cpp" in lang:
            template_name = "Dockerfile.base-cpp"
        elif "c" in lang and "c++" not in lang:
            template_name = "Dockerfile.base-c"
        elif "typescript" in lang or "javascript" in lang:
            template_name = "Dockerfile.base-typescript"
        elif "rust" in lang:
            template_name = "Dockerfile.base-rust"
        elif "java" in lang or "kotlin" in lang:
            template_name = "Dockerfile.base-java"
        else:
            log_warn(f"Unknown language {lang}; defaulting to C template")
            template_name = "Dockerfile.base-c"

    template_path = subrepo_templates_dir / template_name
    if template_path.exists():
        with open(template_path, "r", encoding="utf-8") as f:
            return f.read()

    log_error(f"Base template not found at {template_path}")
    return ""


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


def write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as output_file:
        output_file.write(content)


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
        "if find /home/manualrepos/repo -maxdepth 6 -type f "
        "\\( -path '*/target/release/*' -o -path '*/build/*' -o -name '*.so*' -o -name '*.a' -o -name '*.jar' -o -name '*.whl' \\) "
        "| head -n 1 | grep -q .; then "
        "echo 'fallback: found build artifact in repository'; "
        "exit 0; "
        "fi; "
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


async def run_build_verification(image_tag: str, repo_name: str, attempt: int, smoke_command: str) -> tuple[int, str, list[str], bool]:
    user, workdir = await get_image_runtime_context(image_tag)
    command = [
        args.container_cli,
        "run",
        "--rm",
    ]

    if user:
        command.extend(["--user", user])
    if workdir:
        command.extend(["--workdir", workdir])

    command.extend([
        "--entrypoint",
        "/bin/sh",
        image_tag,
        "-lc",
        smoke_command,
    ])

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

    return returncode, "".join(output_chunks), command, timed_out


async def request_repair(
    repo_url: str,
    attempt_number: int,
    classification: dict,
    summary: str,
    dockerfile_content: str,
    build_log: str,
    llm_metrics: dict,
) -> str:
    base_template = get_base_template(classification)
    prompt = (
        PROMPT_TEMPLATE.replace("{{REPO_URL}}", repo_url)
        .replace("{{ATTEMPT_NUMBER}}", str(attempt_number))
        .replace("{{BASE_TEMPLATE_CONTENT}}", base_template)
        .replace("{{CLASSIFICATION_RESULT}}", render_yaml(classification))
        .replace("{{SUMMARY_CONTENT}}", summary)
        .replace("{{DOCKERFILE_CONTENT}}", dockerfile_content)
        .replace("{{BUILD_LOG}}", trim_log(build_log))
    )

    response = await chat_completion_with_retries(
        client=client,
        model=args.model,
        temperature=args.temperature,
        messages=[{"role": "user", "content": prompt}],
        repo_url=repo_url,
        phase="repair",
        metrics=llm_metrics,
        timeout_seconds=args.repair_timeout,
        max_retries=args.llm_max_retries,
        retry_backoff_seconds=args.llm_retry_backoff_seconds,
    )
    if response.usage:
        log_info(f"[TOKENS] {json.dumps({'phase': 'repair', 'repo': repo_url, 'attempt': attempt_number, 'prompt_tokens': response.usage.prompt_tokens, 'completion_tokens': response.usage.completion_tokens, 'total_tokens': response.usage.total_tokens})}")
    return extract_dockerfile((response.choices[0].message.content or "").strip())


async def request_verification_command_repair(
    repo_url: str,
    classification: dict,
    dockerfile_content: str,
    current_verify_command: str,
    verify_log: str,
    llm_metrics: dict,
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

    response = await chat_completion_with_retries(
        client=client,
        model=args.model,
        temperature=args.temperature,
        messages=[{"role": "user", "content": prompt}],
        repo_url=repo_url,
        phase="verify-repair",
        metrics=llm_metrics,
        timeout_seconds=args.verify_repair_timeout,
        max_retries=args.llm_max_retries,
        retry_backoff_seconds=args.llm_retry_backoff_seconds,
    )
    if response.usage:
        log_info(f"[TOKENS] {json.dumps({'phase': 'verify-repair', 'repo': repo_url, 'prompt_tokens': response.usage.prompt_tokens, 'completion_tokens': response.usage.completion_tokens, 'total_tokens': response.usage.total_tokens})}")
    return (response.choices[0].message.content or "").strip().strip("`")


async def request_verification_command_refresh(
    repo_url: str,
    classification: dict,
    dockerfile_content: str,
    current_verify_command: str,
    llm_metrics: dict,
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

    response = await chat_completion_with_retries(
        client=client,
        model=args.model,
        temperature=args.temperature,
        messages=[{"role": "user", "content": prompt}],
        repo_url=repo_url,
        phase="verify-refresh",
        metrics=llm_metrics,
        timeout_seconds=args.verify_repair_timeout,
        max_retries=args.llm_max_retries,
        retry_backoff_seconds=args.llm_retry_backoff_seconds,
    )
    if response.usage:
        log_info(f"[TOKENS] {json.dumps({'phase': 'verify-refresh', 'repo': repo_url, 'prompt_tokens': response.usage.prompt_tokens, 'completion_tokens': response.usage.completion_tokens, 'total_tokens': response.usage.total_tokens})}")
    return (response.choices[0].message.content or "").strip().strip("`")


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

        report: dict = {
            "repo": repo_url,
            "dockerfile": str(dockerfile_path),
            "max_attempts": args.max_attempts,
            "success": False,
            "attempts": [],
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
            current_dockerfile = dockerfile_path.read_text(encoding="utf-8")
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
                            llm_metrics=llm_metrics,
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
                        llm_metrics=llm_metrics,
                    )
                    current_dockerfile, stop = _apply_repair(
                        repo_url, repo_name, current_dockerfile, repaired_dockerfile,
                        dockerfile_path, report_dir, attempt,
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
                    llm_metrics=llm_metrics,
                )
                current_dockerfile, stop = _apply_repair(
                    repo_url, repo_name, current_dockerfile, repaired_dockerfile,
                    dockerfile_path, report_dir, attempt,
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
            write_text(llm_metrics_path, render_yaml(finalize_llm_metrics(llm_metrics)))
            log_info(f"LLM metrics saved at {llm_metrics_path}")

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