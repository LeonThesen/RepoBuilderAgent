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

from config import OPENAI_API_KEY, OPENAI_BASE_URL, OPENAI_MODEL
from log_utils import log_error, log_info, log_trace, log_warn, set_tqdm_bar, set_trace_enabled
from common import (
    ensure_repo_checkout,
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
parser.add_argument("--temperature", type=float, default=0.0, help="Temperature for the model")
parser.add_argument("--timeout", type=int, default=120, help="Timeout for API requests in seconds")
parser.add_argument("--trace", action="store_true", help="Enable verbose trace logs")
parser.add_argument("--force", action="store_true", help="Overwrite existing generated Dockerfiles")
parser.add_argument("--results-dir", default="classification_results", help="Directory containing classification result YAML files")
parser.add_argument("--summaries-dir", default="summaries", help="Directory containing repository summary files")
parser.add_argument("--repos-dir", default="repos", help="Directory containing cloned repositories")
parser.add_argument("--output-dir", default="dockerfiles", help="Directory where generated Dockerfiles will be written")
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

with open(prompt_path("PROMPT_DOCKERFILE.md"), "r", encoding="utf-8") as prompt_file:
    PROMPT_TEMPLATE = prompt_file.read()

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
        elif "java" in lang:
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
            
            # Generation loop with hadolint validation
            max_lint_attempts = 3
            dockerfile_content = None
            lint_errors = ""
            
            for lint_attempt in range(1, max_lint_attempts + 1):
                prompt = (
                    PROMPT_TEMPLATE.replace("{{REPO_URL}}", repo_url)
                    .replace("{{BASE_TEMPLATE_CONTENT}}", base_template)
                    .replace("{{CLASSIFICATION_RESULT}}", render_yaml(classification))
                    .replace("{{SUMMARY_CONTENT}}", summary)
                )
                
                # Add hadolint error feedback if this is a retry
                if lint_errors:
                    prompt += f"\n\n**Previous hadolint validation failed with:**\n```\n{lint_errors}\n```\n\nPlease fix the Dockerfile syntax errors above."
                
                log_info(f"Generating Dockerfile for {repo_url} (lint attempt {lint_attempt}/{max_lint_attempts})...")
                response = await client.chat.completions.create(
                    model=args.model,
                    temperature=args.temperature,
                    messages=[{"role": "user", "content": prompt}],
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
                    
                    is_valid, validation_error = await validate_dockerfile_syntax(temp_dockerfile, repo_name)
                    temp_dockerfile.unlink()  # Clean up temp file
                    
                    if is_valid:
                        log_info(f"[hadolint {repo_name}] Dockerfile syntax OK on attempt {lint_attempt}")
                        break
                    else:
                        lint_errors = validation_error[:1000]
                        log_warn(f"[hadolint {repo_name}] Dockerfile syntax error on attempt {lint_attempt}: {lint_errors[:200]}")
                        if lint_attempt < max_lint_attempts:
                            continue
                        else:
                            log_warn(f"[hadolint {repo_name}] Max lint attempts reached; skipping repo")
                            return
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
            verify_response = await client.chat.completions.create(
                model=args.model,
                temperature=args.temperature,
                messages=[{"role": "user", "content": verify_prompt}],
            )
            verify_command = verify_response.choices[0].message.content.strip().strip("`")
            if verify_response.usage:
                log_info(f"[TOKENS] {json.dumps({'phase': 'dockerfile-verify-cmd', 'repo': repo_url, 'prompt_tokens': verify_response.usage.prompt_tokens, 'completion_tokens': verify_response.usage.completion_tokens, 'total_tokens': verify_response.usage.total_tokens})}")
            verify_command_path = output_path.with_suffix(".verify-command")
            with open(verify_command_path, "w", encoding="utf-8") as verify_file:
                verify_file.write(verify_command + "\n")
            log_info(f"Saved build verification command to {verify_command_path}: {verify_command}")

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