import argparse
import asyncio
import base64
import json
import os
import re
import shutil
import ssl
from pathlib import Path

import httpx
from openai import APIError, APITimeoutError, AsyncOpenAI
from tqdm import tqdm

from config import OPENAI_API_KEY, OPENAI_BASE_URL, OPENAI_MODEL
from log_utils import log_error, log_info, log_trace, log_warn, set_tqdm_bar, set_trace_enabled
from common import (
    ensure_repo_checkout,
    load_repo_urls,
    load_summary,
    read_yaml_file,
    render_yaml,
    repo_name_from_url,
    should_use_progress,
    update_progress,
)


def inject_ca_cert_into_dockerfile(dockerfile_content: str, ca_cert_b64: str | None = None) -> str:
    """
    Inject CA certificate setup into the Dockerfile if MANUALREPOS_CA_CERT_B64 is present.
    This ensures curl/git/wget/npm/pip/maven/rust can reach package registries behind TLS interception.
    
    Sets up:
    - System CA trust store (apt, curl, git, wget)
    - Environment variables for Python (pip), Node.js (npm/pnpm), Java (maven)
    - Java keystore import for Maven/Gradle
    - Rust CA configuration
    """
    if not ca_cert_b64:
        ca_cert_b64 = os.getenv("MANUALREPOS_CA_CERT_B64")
    if not ca_cert_b64:
        return dockerfile_content

    try:
        ca_cert_pem = base64.b64decode(ca_cert_b64).decode("utf-8")
    except Exception as e:
        log_warn(f"Failed to decode MANUALREPOS_CA_CERT_B64: {e}")
        return dockerfile_content

    ca_cert_path = "/usr/local/share/ca-certificates/custom-ca.crt"
    ca_setup_commands = f"""
RUN apt-get update -qq && apt-get install -y --no-install-recommends ca-certificates curl default-jre-headless 2>/dev/null || true
RUN mkdir -p /usr/local/share/ca-certificates
RUN cat > {ca_cert_path} << 'EOF'
{ca_cert_pem}
EOF
RUN update-ca-certificates
RUN if command -v keytool >/dev/null 2>&1; then keytool -import -alias custom-ca -file {ca_cert_path} -into /etc/ssl/certs/java/cacerts -storepass changeit -noprompt 2>/dev/null || true; fi
ENV SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt
ENV REQUESTS_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt
ENV CURL_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt
ENV NODE_TLS_REJECT_UNAUTHORIZED=0
ENV NODE_OPTIONS=--use-openssl-ca
ENV NODE_EXTRA_CA_CERTS=/etc/ssl/certs/ca-certificates.crt
ENV RUSTFLAGS=-Ctarget-feature=+crt-static
"""

    lines = dockerfile_content.split("\n")
    injected_lines = []
    inserted = False

    for i, line in enumerate(lines):
        injected_lines.append(line)
        # Insert CA setup after first FROM line
        if not inserted and line.strip().upper().startswith("FROM"):
            injected_lines.append("USER root")
            injected_lines.append(ca_setup_commands.strip())
            inserted = True

    return "\n".join(injected_lines)


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
parser.add_argument("--verify-command", default="echo build-ok", help="Shell command executed inside the built image to verify the build produced working software")
parser.add_argument("--verify-timeout", type=int, default=30, help="Timeout in seconds for build verification container execution")
parser.add_argument("--force", action="store_true", help="Re-run repair even if a successful report.yaml already exists")
args = parser.parse_args()


os.environ.setdefault("SSL_CERT_FILE", "/etc/ssl/certs/ca-certificates.crt")

client = AsyncOpenAI(
    base_url=args.endpoint,
    api_key=args.api_key,
    timeout=args.timeout,
)

with open(Path("prompts/PROMPT_DOCKERFILE_REPAIR.md"), "r", encoding="utf-8") as prompt_file:
    PROMPT_TEMPLATE = prompt_file.read()

sem = asyncio.Semaphore(1)

set_trace_enabled(args.trace)


# Files/directories to remove from the repo before docker build so the build
# context is free of documentation and CI/CD configuration that can't affect
# an installation and may only inflate the context or mislead the build.
_DELETE_DOCS_EXTENSIONS: tuple[str, ...] = (".md", ".rst", ".pdf", ".doc", ".docx")
_DELETE_DOCS_FILE_NAMES: frozenset[str] = frozenset({
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
    ".github", ".gitlab", ".circleci", ".buildkite", ".drone",
    ".woodpecker", ".jenkins", ".azure-pipelines",
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


def _apply_repair(
    repo_url: str,
    repo_name: str,
    current: str,
    repaired: str,
    dockerfile_path: Path,
    report_dir: Path,
    attempt: int,
) -> tuple[str | None, bool]:
    """Validate and write a repaired Dockerfile. Returns (updated_content, should_stop)."""
    if not repaired.strip():
        log_warn(f"Repair model returned an empty Dockerfile for {repo_url}; stopping retries.")
        return None, True
    if repaired.strip() == current.strip():
        log_warn(f"Repair model returned an unchanged Dockerfile for {repo_url}; stopping retries.")
        return None, True
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


async def run_build_verification(image_tag: str, repo_name: str, attempt: int, smoke_command: str) -> tuple[int, str, list[str], bool]:
    command = [
        args.container_cli,
        "run",
        "--rm",
        "--entrypoint",
        "/bin/sh",
        image_tag,
        "-lc",
        smoke_command,
    ]

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
) -> str:
    prompt = (
        PROMPT_TEMPLATE.replace("{{REPO_URL}}", repo_url)
        .replace("{{ATTEMPT_NUMBER}}", str(attempt_number))
        .replace("{{CLASSIFICATION_RESULT}}", render_yaml(classification))
        .replace("{{SUMMARY_CONTENT}}", summary)
        .replace("{{DOCKERFILE_CONTENT}}", dockerfile_content)
        .replace("{{BUILD_LOG}}", trim_log(build_log))
    )

    response = await client.chat.completions.create(
        model=args.model,
        temperature=args.temperature,
        messages=[{"role": "user", "content": prompt}],
    )
    if response.usage:
        log_info(f"[TOKENS] {json.dumps({'phase': 'repair', 'repo': repo_url, 'attempt': attempt_number, 'prompt_tokens': response.usage.prompt_tokens, 'completion_tokens': response.usage.completion_tokens, 'total_tokens': response.usage.total_tokens})}")
    return extract_dockerfile(response.choices[0].message.content.strip())


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

                log_info(f"Build attempt {attempt}/{args.max_attempts} for {repo_url}...")
                log_info(f"Streaming build output; full log will be written to {build_log_path}")
                exit_code, streamed_output = await run_build(build_command, repo_name, attempt)
                build_log = combine_build_output(build_command, exit_code, streamed_output)

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

                    log_warn(
                        f"Build verification failed for {repo_url} on attempt {attempt} with exit code {verify_exit_code}; verify log: {verify_log_path}"
                    )

                    build_log = build_log + "\n\nBUILD_VERIFICATION_LOG:\n" + verify_log

                    if attempt == args.max_attempts:
                        log_warn(f"Build verification still failing for {repo_url} after {args.max_attempts} attempts")
                        break

                    log_info(f"Diagnosing failed build verification for {repo_url} and rewriting Dockerfile...")
                    repaired_dockerfile = await request_repair(
                        repo_url=repo_url,
                        attempt_number=attempt,
                        classification=classification,
                        summary=summary,
                        dockerfile_content=current_dockerfile,
                        build_log=build_log,
                    )
                    current_dockerfile, stop = _apply_repair(
                        repo_url, repo_name, current_dockerfile, repaired_dockerfile,
                        dockerfile_path, report_dir, attempt,
                    )
                    if stop:
                        break

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
                )
                current_dockerfile, stop = _apply_repair(
                    repo_url, repo_name, current_dockerfile, repaired_dockerfile,
                    dockerfile_path, report_dir, attempt,
                )
                if stop:
                    break

            write_text(report_path, render_yaml(report))
            log_info(f"Repair report written to {report_path}")

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