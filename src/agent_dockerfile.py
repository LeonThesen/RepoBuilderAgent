import argparse
import asyncio
import json
import os
import re
import ssl
import subprocess
import sys
from pathlib import Path

import httpx
import yaml
from openai import APIError, APITimeoutError, AsyncOpenAI
from tqdm import tqdm

from config import OPENAI_API_KEY, OPENAI_BASE_URL, OPENAI_MODEL
from log_utils import log_error, log_info, log_trace, log_warn, set_tqdm_bar, set_trace_enabled
from repo_fingerprint import fingerprint


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


os.environ.setdefault("SSL_CERT_FILE", "/etc/ssl/certs/ca-certificates.crt")

client = AsyncOpenAI(
    base_url=args.endpoint,
    api_key=args.api_key,
    timeout=args.timeout,
)

with open(Path("prompts/PROMPT_DOCKERFILE.md"), "r", encoding="utf-8") as prompt_file:
    PROMPT_TEMPLATE = prompt_file.read()

with open(Path("prompts/PROMPT_BUILD_VERIFICATION.md"), "r", encoding="utf-8") as prompt_file:
    BUILD_VERIFY_PROMPT_TEMPLATE = prompt_file.read()

sem = asyncio.Semaphore(4)

set_trace_enabled(args.trace)


def should_use_progress(total_repos: int) -> bool:
    return total_repos > 1 and sys.stderr.isatty() and not args.trace


def repo_name_from_url(repo_url: str) -> str:
    return repo_url.rstrip("/").split("/")[-1].replace(".git", "")


def load_repo_urls() -> list[str]:
    if args.repo_url:
        repos = [url.strip() for url in args.repo_url if url and url.strip()]
        return list(dict.fromkeys(repos))

    with open(args.input_file, "r", encoding="utf-8") as input_file:
        return [item["url"] for item in json.load(input_file)]


def read_yaml_file(path: Path) -> dict | None:
    if not path.exists():
        return None

    with open(path, "r", encoding="utf-8") as yaml_file:
        return yaml.safe_load(yaml_file)


def load_summary(repo_name: str, repo_path: Path, summaries_dir: Path) -> str:
    reduced_summary_path = summaries_dir / f"{repo_name}.md"
    if reduced_summary_path.exists():
        with open(reduced_summary_path, "r", encoding="utf-8") as summary_file:
            return summary_file.read()

    selected_files_path = summaries_dir / f"{repo_name}.selected-files.yaml"
    selected_files_config = read_yaml_file(selected_files_path) or {}
    selected_files = selected_files_config.get("selected_files")
    if isinstance(selected_files, list) and selected_files:
        return fingerprint(
            format="md",
            repo_path=repo_path,
            selected_files=selected_files,
            include_tree=False,
            context="dockerfile-selected",
        )

    return fingerprint(
        format="md",
        repo_path=repo_path,
        selected_files=None,
        include_tree=True,
        context="dockerfile-baseline",
    )


def render_classification(classification: dict) -> str:
    return yaml.dump(classification, sort_keys=False, allow_unicode=True)


def extract_dockerfile(raw: str) -> str:
    match = re.search(r"```(?:dockerfile)?\n(.*?)```", raw, re.DOTALL | re.IGNORECASE)
    content = match.group(1) if match else raw
    return content.strip() + "\n"


async def update_progress(progress_state: dict, repo_name: str) -> None:
    if progress_state["bar"] is None:
        return
    async with progress_state["lock"]:
        progress_state["bar"].set_postfix_str(repo_name)
        progress_state["bar"].update(1)


async def ensure_repo_checkout(repo_url: str, repo_path: Path) -> bool:
    if repo_path.exists():
        return True

    log_info(f"Cloning {repo_url} -> {repo_path}")
    result = subprocess.run(["git", "clone", repo_url, str(repo_path)], check=False)
    if result.returncode != 0:
        log_warn(f"Failed to clone {repo_url}; skipping Dockerfile generation.")
        return False
    return True


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
            if not await ensure_repo_checkout(repo_url, repo_path):
                return

            summary = load_summary(repo_name, repo_path, summaries_dir)
            prompt = (
                PROMPT_TEMPLATE.replace("{{REPO_URL}}", repo_url)
                .replace("{{CLASSIFICATION_RESULT}}", render_classification(classification))
                .replace("{{SUMMARY_CONTENT}}", summary)
            )

            log_info(f"Generating Dockerfile for {repo_url}...")
            response = await client.chat.completions.create(
                model=args.model,
                temperature=args.temperature,
                messages=[{"role": "user", "content": prompt}],
            )
            raw = response.choices[0].message.content.strip()
            dockerfile_content = extract_dockerfile(raw)
            if response.usage:
                import json as _json
                log_info(f"[TOKENS] {_json.dumps({'phase': 'dockerfile', 'repo': repo_url, 'prompt_tokens': response.usage.prompt_tokens, 'completion_tokens': response.usage.completion_tokens, 'total_tokens': response.usage.total_tokens})}")

            if not dockerfile_content.strip():
                log_warn(f"Empty Dockerfile output for {repo_url}; skipping write.")
                return

            with open(output_path, "w", encoding="utf-8") as output_file:
                output_file.write(dockerfile_content)

            log_trace(f"Dockerfile written for {repo_name} at {output_path}")
            log_info(f"Saved Dockerfile to {output_path}")

            # Generate a repo-specific build verification command
            verify_prompt = (
                BUILD_VERIFY_PROMPT_TEMPLATE
                .replace("{{REPO_URL}}", repo_url)
                .replace("{{CLASSIFICATION_RESULT}}", render_classification(classification))
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
                import json as _json
                log_info(f"[TOKENS] {_json.dumps({'phase': 'dockerfile-verify-cmd', 'repo': repo_url, 'prompt_tokens': verify_response.usage.prompt_tokens, 'completion_tokens': verify_response.usage.completion_tokens, 'total_tokens': verify_response.usage.total_tokens})}")
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
    repos = load_repo_urls()
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
    if should_use_progress(len(repos)):
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