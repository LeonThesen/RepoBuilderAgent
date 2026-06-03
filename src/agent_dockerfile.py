import argparse
import asyncio
import json
import os
import re
import ssl
from pathlib import Path

import httpx
from openai import APIError, APITimeoutError, AsyncOpenAI
from tqdm import tqdm

try:
    from RepoBuilderAgent.src.config import OPENAI_API_KEY, OPENAI_BASE_URL, OPENAI_MODEL
    from RepoBuilderAgent.src.log_utils import log_error, log_info, log_trace, log_warn, set_tqdm_bar, set_trace_enabled
    from RepoBuilderAgent.src.timeout_config import load_timeout_defaults
    from RepoBuilderAgent.src.prompt_profiles import (
        apply_prompt_profile,
        prompt_profile_metadata,
        resolve_prompt_profile,
        resolve_prompt_temperature,
    )
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
    from timeout_config import load_timeout_defaults
    from prompt_profiles import (
        apply_prompt_profile,
        prompt_profile_metadata,
        resolve_prompt_profile,
        resolve_prompt_temperature,
    )
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

TIMEOUTS = load_timeout_defaults(
    "agent_dockerfile",
    {
        "timeout": 120,
        "llm_max_retries": 2,
        "llm_retry_backoff_seconds": 2.0,
        "dockerfile_timeout": 240,
        "verify_cmd_timeout": 180,
    },
)


parser = argparse.ArgumentParser(
    description="Generate Dockerfiles for repositories using classification results produced by agent_classify.py."
)
parser.add_argument("--input-file", default="repos.json", help="Path to input file containing repository URLs")
parser.add_argument(
    "--repo-url",
    action="append",
    default=[],
    help="Generate a Dockerfile for a specific repository URL (can be passed multiple times). Overrides --input-file when provided.",
)
parser.add_argument("--endpoint", default=os.getenv("LLM_ENDPOINT", OPENAI_BASE_URL), help="Custom API endpoint URL")
parser.add_argument("--model", default=os.getenv("LLM_MODEL", OPENAI_MODEL), help="Model name")
parser.add_argument("--api-key", default=os.getenv("LLM_API_KEY", OPENAI_API_KEY), help="API key")
parser.add_argument("--prompt-profile", default=os.getenv("PROMPT_PROFILE", "P*"), help="Prompt profile name from RepoBuilderAgent/config/prompt_profiles.yaml (supports alias P*)")
parser.add_argument("--temperature", type=float, default=None, help="Temperature override for the model; defaults to selected prompt profile value")
parser.add_argument("--timeout", type=int, default=int(TIMEOUTS["timeout"]), help="Timeout for API requests in seconds")
parser.add_argument("--llm-max-retries", type=int, default=int(TIMEOUTS["llm_max_retries"]), help="Maximum retries for transient LLM timeouts and retryable API errors")
parser.add_argument("--llm-retry-backoff-seconds", type=float, default=float(TIMEOUTS["llm_retry_backoff_seconds"]), help="Base exponential backoff delay in seconds for LLM retries")
parser.add_argument("--dockerfile-timeout", type=int, default=int(TIMEOUTS["dockerfile_timeout"]), help="Timeout for Dockerfile generation calls in seconds")
parser.add_argument("--verify-cmd-timeout", type=int, default=int(TIMEOUTS["verify_cmd_timeout"]), help="Timeout for verification command generation calls in seconds")
parser.add_argument("--trace", action="store_true", help="Enable verbose trace logs")
parser.add_argument("--force", action="store_true", help="Overwrite existing generated Dockerfiles")
parser.add_argument("--skip-hadolint", action="store_true", help="Skip Dockerfile syntax validation via hadolint")
parser.add_argument("--results-dir", default="classification_results", help="Directory containing classification result YAML files")
parser.add_argument("--summaries-dir", default="summaries", help="Directory containing repository summary files")
parser.add_argument("--repos-dir", default="repos", help="Directory containing cloned repositories")
parser.add_argument("--output-dir", default="dockerfiles", help="Directory where generated Dockerfiles will be written")
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

with open(prompt_path("PROMPT_DOCKERFILE.md"), "r", encoding="utf-8") as prompt_file:
    PROMPT_TEMPLATE = apply_prompt_profile(prompt_file.read(), PROMPT_PROFILE, "dockerfile")

with open(prompt_path("PROMPT_BUILD_VERIFICATION.md"), "r", encoding="utf-8") as prompt_file:
    BUILD_VERIFY_PROMPT_TEMPLATE = prompt_file.read()

sem = asyncio.Semaphore(4)

set_trace_enabled(args.trace)



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


def render_failed_lint_attempts(failed_attempts: list[dict[str, str]]) -> str:
    if not failed_attempts:
        return ""

    recent_attempts = failed_attempts[-3:]
    skipped_count = len(failed_attempts) - len(recent_attempts)
    sections = [
        "\n\nPrevious Dockerfile generations failed Hadolint validation.",
        "\nDo not repeat these mistakes. Return only a valid Dockerfile.",
    ]
    if skipped_count > 0:
        sections.append(f"\n{skipped_count} earlier failed lint attempts are omitted for brevity.")
    for index, failed_attempt in enumerate(recent_attempts, start=len(failed_attempts) - len(recent_attempts) + 1):
        sections.append(
            "\n\n"
            f"Attempt {index} failed Hadolint with:\n"
            "```text\n"
            f"{failed_attempt['error']}\n"
            "```\n"
            "Generated Dockerfile was:\n"
            "```dockerfile\n"
            f"{failed_attempt['dockerfile'][:1200]}"
            "```"
        )
    return "".join(sections)


def render_repeated_lint_guardrail(failed_attempts: list[dict[str, str]]) -> str:
    if len(failed_attempts) < 3:
        return ""

    recent_attempts = failed_attempts[-3:]
    recent_errors = [attempt["error"] for attempt in recent_attempts]
    if len(set(recent_errors)) != 1:
        return ""

    match = re.search(r":(\d+):(\d+)\s+(.*)", recent_attempts[-1]["error"])
    if not match:
        return (
            "\n\nHadolint has produced the same syntax error three times in a row. "
            "Change your structure substantially and return only valid Dockerfile instructions."
        )

    line_number = int(match.group(1))
    error_summary = match.group(3).strip()
    dockerfile_lines = recent_attempts[-1]["dockerfile"].splitlines()
    offending_line = ""
    if 1 <= line_number <= len(dockerfile_lines):
        offending_line = dockerfile_lines[line_number - 1].strip()

    sections = [
        "\n\nHadolint has produced the same syntax error three times in a row.",
        f"\nThe current parser failure is at line {line_number}: {error_summary}",
    ]
    if offending_line:
        sections.append(
            "\nThe line currently at that position is:\n"
            "```text\n"
            f"{offending_line}\n"
            "```"
        )
    sections.append(
        "\nRewrite that part of the Dockerfile so every line is a valid Dockerfile instruction. "
        "Do not emit bullet points, prose, or raw certificate text. Return only the Dockerfile."
    )
    return "".join(sections)



async def generate_dockerfile(
    repo_url: str,
    repos_dir: Path,
    summaries_dir: Path,
    results_dir: Path,
    output_dir: Path,
    progress_state: dict,
) -> None:
    async with sem:
        repo_name = repo_name_from_url(repo_url)
        output_path = output_dir / f"{repo_name}.Dockerfile"
        llm_metrics_path = output_dir / f"{repo_name}.llm-metrics.yaml"
        llm_metrics = init_llm_metrics(repo_url, args.model, args.endpoint, args.timeout, args.llm_max_retries)
        llm_metrics["prompt_profile"] = prompt_profile_metadata(PROMPT_PROFILE, EFFECTIVE_TEMPERATURE)

        try:
            if output_path.exists() and not args.force:
                log_info(f"Skipping {repo_url}: existing Dockerfile found at {output_path}")
                return

            classification_path = results_dir / f"{repo_name}.yaml"
            classification = read_yaml_file(classification_path)
            if not classification:
                log_warn(
                    f"Skipping {repo_url}: classification result missing at {classification_path}. Run agent_classify.py first."
                )
                return

            repo_path = repos_dir / repo_name
            if not await ensure_repo_checkout(repo_url, repo_path, "skipping Dockerfile generation"):
                return

            summary = load_summary(repo_name, repo_path, summaries_dir)
            base_template = get_base_template(classification)

            # Regenerate until Hadolint accepts the Dockerfile. These retries do not
            # count as build attempts because no project build has run yet.
            dockerfile_content = None
            failed_lint_attempts: list[dict[str, str]] = []
            lint_attempt = 1

            while True:
                prompt = (
                    PROMPT_TEMPLATE.replace("{{REPO_URL}}", repo_url)
                    .replace("{{BASE_TEMPLATE_CONTENT}}", base_template)
                    .replace("{{CLASSIFICATION_RESULT}}", render_yaml(classification))
                    .replace("{{SUMMARY_CONTENT}}", summary)
                )

                if failed_lint_attempts:
                    prompt += render_failed_lint_attempts(failed_lint_attempts)
                    prompt += render_repeated_lint_guardrail(failed_lint_attempts)

                log_info(f"Generating Dockerfile for {repo_url} (lint attempt {lint_attempt})...")
                response = await chat_completion_with_retries(
                    client=client,
                    model=args.model,
                    temperature=EFFECTIVE_TEMPERATURE,
                    messages=[{"role": "user", "content": prompt}],
                    repo_url=repo_url,
                    phase="dockerfile",
                    metrics=llm_metrics,
                    timeout_seconds=args.dockerfile_timeout,
                    max_retries=args.llm_max_retries,
                    retry_backoff_seconds=args.llm_retry_backoff_seconds,
                )
                raw = response.choices[0].message.content.strip()
                dockerfile_content = extract_dockerfile(raw)
                if response.usage:
                    log_info(f"[TOKENS] {json.dumps({'phase': 'dockerfile', 'repo': repo_url, 'prompt_tokens': response.usage.prompt_tokens, 'completion_tokens': response.usage.completion_tokens, 'total_tokens': response.usage.total_tokens})}")

                if not dockerfile_content.strip():
                    log_warn(f"Empty Dockerfile output for {repo_url}; skipping write.")
                    return

                dockerfile_content = inject_ca_cert_into_dockerfile(dockerfile_content)

                # Validate Dockerfile syntax with hadolint
                if not args.skip_hadolint:
                    # Write temporary file for hadolint check
                    temp_dockerfile = output_dir / f".{repo_name}.Dockerfile.tmp"
                    with open(temp_dockerfile, "w", encoding="utf-8") as tmp_file:
                        tmp_file.write(dockerfile_content)

                    try:
                        is_valid, validation_error = await validate_dockerfile_syntax(temp_dockerfile, repo_name)
                    finally:
                        if temp_dockerfile.exists():
                            temp_dockerfile.unlink()

                    if is_valid:
                        log_info(f"[hadolint {repo_name}] Dockerfile syntax OK on attempt {lint_attempt}")
                        break

                    failed_lint_attempts.append(
                        {
                            "error": validation_error[:1500],
                            "dockerfile": dockerfile_content,
                        }
                    )
                    log_warn(
                        f"[hadolint {repo_name}] Dockerfile syntax error on attempt {lint_attempt}: "
                        f"{validation_error[:200]}"
                    )
                    lint_attempt += 1
                    continue
                else:
                    # skip_hadolint mode: just use first generation
                    log_info(f"[hadolint {repo_name}] Skipping validation (--skip-hadolint set)")
                    break

            # Write Dockerfile
            with open(output_path, "w", encoding="utf-8") as output_file:
                output_file.write(dockerfile_content)

            log_trace(f"Dockerfile written for {repo_name} at {output_path}")
            log_info(f"Saved Dockerfile to {output_path}")

            # Generate a repo-specific build verification command
            verify_prompt = (
                BUILD_VERIFY_PROMPT_TEMPLATE
                .replace("{{REPO_URL}}", repo_url)
                .replace("{{CLASSIFICATION_RESULT}}", render_yaml(classification))
                .replace("{{DOCKERFILE_CONTENT}}", dockerfile_content)
            )
            log_info(f"Generating build verification command for {repo_url}...")
            verify_response = await chat_completion_with_retries(
                client=client,
                model=args.model,
                temperature=EFFECTIVE_TEMPERATURE,
                messages=[{"role": "user", "content": verify_prompt}],
                repo_url=repo_url,
                phase="dockerfile-verify-cmd",
                metrics=llm_metrics,
                timeout_seconds=args.verify_cmd_timeout,
                max_retries=args.llm_max_retries,
                retry_backoff_seconds=args.llm_retry_backoff_seconds,
            )
            verify_command = verify_response.choices[0].message.content.strip().strip("`")
            if verify_response.usage:
                log_info(f"[TOKENS] {json.dumps({'phase': 'dockerfile-verify-cmd', 'repo': repo_url, 'prompt_tokens': verify_response.usage.prompt_tokens, 'completion_tokens': verify_response.usage.completion_tokens, 'total_tokens': verify_response.usage.total_tokens})}")
            verify_command_path = output_path.with_suffix(".verify-command")
            with open(verify_command_path, "w", encoding="utf-8") as verify_file:
                verify_file.write(verify_command + "\n")
            log_info(f"Saved build verification command to {verify_command_path}: {verify_command}")

            with open(llm_metrics_path, "w", encoding="utf-8") as metrics_file:
                metrics_file.write(render_yaml(finalize_llm_metrics(llm_metrics)))
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
            log_error(f"Unexpected error while generating Dockerfile for {repo_url}: {error}")
        finally:
            with open(llm_metrics_path, "w", encoding="utf-8") as metrics_file:
                metrics_file.write(render_yaml(finalize_llm_metrics(llm_metrics)))
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
    output_dir = workspace_root / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    progress_bar = None
    if should_use_progress(len(repos), args.trace):
        progress_bar = tqdm(total=len(repos), desc="Generating Dockerfiles", unit="repo", dynamic_ncols=True)

    progress_state = {
        "lock": asyncio.Lock(),
        "bar": progress_bar,
    }
    set_tqdm_bar(progress_state["bar"])
    log_info(f"Starting Dockerfile generation for {len(repos)} repositories")

    tasks = [
        generate_dockerfile(repo, repos_dir, summaries_dir, results_dir, output_dir, progress_state)
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