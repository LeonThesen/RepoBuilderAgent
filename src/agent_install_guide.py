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
    load_repo_urls,
    load_summary,
    prompt_path,
    read_yaml_file,
    render_yaml,
    repo_name_from_url,
    should_use_progress,
    update_progress,
)


parser = argparse.ArgumentParser(
    description="Generate human-readable INSTALL.md guides from final Dockerfiles and repository evidence."
)
parser.add_argument("--input-file", default="repos.json", help="Path to input file containing repository URLs")
parser.add_argument(
    "--repo-url",
    action="append",
    default=[],
    help="Generate an install guide for a specific repository URL (can be passed multiple times). Overrides --input-file when provided.",
)
parser.add_argument("--endpoint", default=os.getenv("LLM_ENDPOINT", OPENAI_BASE_URL), help="Custom API endpoint URL")
parser.add_argument("--model", default=os.getenv("LLM_MODEL", OPENAI_MODEL), help="Model name")
parser.add_argument("--api-key", default=os.getenv("LLM_API_KEY", OPENAI_API_KEY), help="API key")
parser.add_argument("--temperature", type=float, default=0.0, help="Temperature for the model")
parser.add_argument("--timeout", type=int, default=120, help="Timeout for API requests in seconds")
parser.add_argument("--trace", action="store_true", help="Enable verbose trace logs")
parser.add_argument("--force", action="store_true", help="Overwrite existing generated install guides")
parser.add_argument("--results-dir", default="classification_results", help="Directory containing classification result YAML files")
parser.add_argument("--summaries-dir", default="summaries", help="Directory containing repository summary files")
parser.add_argument("--repos-dir", default="repos", help="Directory containing cloned repositories")
parser.add_argument("--dockerfiles-dir", default="dockerfiles", help="Directory containing generated Dockerfiles")
parser.add_argument("--output-dir", default="install-guides", help="Directory where generated INSTALL.md files will be written")
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

with open(prompt_path("PROMPT_INSTALL_GUIDE.md"), "r", encoding="utf-8") as prompt_file:
    PROMPT_TEMPLATE = prompt_file.read()

sem = asyncio.Semaphore(4)

set_trace_enabled(args.trace)



def extract_markdown(raw: str) -> str:
    stripped = raw.strip()
    outer_markdown_match = re.fullmatch(r"```(?:markdown|md)\n(.*?)\n```", stripped, re.DOTALL | re.IGNORECASE)
    content = outer_markdown_match.group(1) if outer_markdown_match else raw
    return content.strip() + "\n"


def is_install_guide_complete(content: str) -> bool:
    normalized = content.strip()
    if len(normalized) < 400:
        return False

    required_markers = [
        "# Install",
        "## Prerequisites",
        "## Build Steps",
        "## Install Artifacts",
        "## Verification",
        "```bash",
    ]
    return all(marker in normalized for marker in required_markers)



async def generate_install_guide(
    repo_url: str,
    repos_dir: Path,
    summaries_dir: Path,
    results_dir: Path,
    dockerfiles_dir: Path,
    output_dir: Path,
    progress_state: dict,
) -> None:
    async with sem:
        repo_name = repo_name_from_url(repo_url)
        output_path = output_dir / repo_name / "INSTALL.md"

        try:
            if output_path.exists() and not args.force:
                log_info(f"Skipping {repo_url}: existing install guide found at {output_path}")
                return

            classification_path = results_dir / f"{repo_name}.yaml"
            classification = read_yaml_file(classification_path)
            if not classification:
                log_warn(
                    f"Skipping {repo_url}: classification result missing at {classification_path}. Run agent_classify.py first."
                )
                return

            dockerfile_path = dockerfiles_dir / f"{repo_name}.Dockerfile"
            if not dockerfile_path.exists():
                log_warn(
                    f"Skipping {repo_url}: Dockerfile missing at {dockerfile_path}. Run Dockerfile generation/repair first."
                )
                return

            repo_path = repos_dir / repo_name
            if not await ensure_repo_checkout(repo_url, repo_path, "skipping install guide generation"):
                return

            summary = load_summary(repo_name, repo_path, summaries_dir)
            dockerfile_content = dockerfile_path.read_text(encoding="utf-8")
            verify_command_path = dockerfiles_dir / f"{repo_name}.verify-command"
            verify_command = verify_command_path.read_text(encoding="utf-8").strip() if verify_command_path.exists() else ""

            prompt = (
                PROMPT_TEMPLATE.replace("{{REPO_URL}}", repo_url)
                .replace("{{CLASSIFICATION_RESULT}}", render_yaml(classification))
                .replace("{{SUMMARY_CONTENT}}", summary)
                .replace("{{DOCKERFILE_CONTENT}}", dockerfile_content)
                .replace("{{VERIFY_COMMAND}}", verify_command)
            )

            log_info(f"Generating INSTALL.md for {repo_url}...")
            install_guide_content = ""
            generation_prompt = prompt
            for attempt in range(1, 3):
                response = await client.chat.completions.create(
                    model=args.model,
                    temperature=args.temperature,
                    messages=[{"role": "user", "content": generation_prompt}],
                )
                raw = response.choices[0].message.content or ""
                install_guide_content = extract_markdown(raw)
                if response.usage:
                    log_info(f"[TOKENS] {json.dumps({'phase': 'install-guide', 'repo': repo_url, 'attempt': attempt, 'prompt_tokens': response.usage.prompt_tokens, 'completion_tokens': response.usage.completion_tokens, 'total_tokens': response.usage.total_tokens})}")

                if is_install_guide_complete(install_guide_content):
                    break

                log_warn(
                    f"INSTALL.md output for {repo_url} was incomplete on generation attempt {attempt}; retrying with stricter instructions."
                )
                generation_prompt = (
                    prompt
                    + "\n\nReturn a complete INSTALL.md document with all required sections and a non-empty bash code block."
                )

            if not is_install_guide_complete(install_guide_content):
                log_warn(f"INSTALL.md output for {repo_url} remained incomplete; skipping write.")
                return

            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "w", encoding="utf-8") as output_file:
                output_file.write(install_guide_content)

            log_trace(f"INSTALL.md written for {repo_name} at {output_path}")
            log_info(f"Saved INSTALL.md to {output_path}")

        except httpx.HTTPError as error:
            log_warn(f"HTTP error for {repo_url}: {error}")
        except ssl.SSLError as error:
            log_warn(f"SSL error for {repo_url}: {error}")
        except APITimeoutError as error:
            log_warn(f"OpenAI timeout for {repo_url}: {error}")
        except APIError as error:
            log_warn(f"OpenAI API error for {repo_url}: {error}")
        except Exception as error:
            log_error(f"Unexpected error while generating INSTALL.md for {repo_url}: {error}")
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
    dockerfiles_dir = workspace_root / args.dockerfiles_dir
    output_dir = workspace_root / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    progress_bar = None
    if should_use_progress(len(repos), args.trace):
        progress_bar = tqdm(total=len(repos), desc="Generating INSTALL guides", unit="repo", dynamic_ncols=True)

    progress_state = {
        "lock": asyncio.Lock(),
        "bar": progress_bar,
    }
    set_tqdm_bar(progress_state["bar"])
    log_info(f"Starting INSTALL.md generation for {len(repos)} repositories")

    tasks = [
        generate_install_guide(repo, repos_dir, summaries_dir, results_dir, dockerfiles_dir, output_dir, progress_state)
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