import json
from openai import AsyncOpenAI, APITimeoutError, APIError
import tiktoken
import argparse
from pathlib import Path
import asyncio
import os
import httpx
import ssl
import re
import subprocess
import sys
import yaml
from tqdm import tqdm
import glob
import shutil
from typing import Any

try:
    from RepoBuilderAgent.src.config import OPENAI_API_KEY, OPENAI_BASE_URL, OPENAI_MODEL
    from RepoBuilderAgent.src.common import (
        chat_completion_with_retries,
        finalize_llm_metrics,
        init_llm_metrics,
        prompt_path,
    )
    from RepoBuilderAgent.src.timeout_config import load_timeout_defaults
    from RepoBuilderAgent.src.prompt_profiles import (
        apply_prompt_profile,
        prompt_profile_metadata,
        resolve_prompt_profile,
        resolve_prompt_temperature,
    )
    from RepoBuilderAgent.src.repo_fingerprint import fingerprint, collect_manifest_files, collect_selected_files, learn_new_files, select_files_by_bm25
    from RepoBuilderAgent.src.log_utils import log_info, log_warn, log_error, log_trace, set_trace_enabled, set_tqdm_bar, log_file_delta
except ImportError:
    # Fallback for direct script execution from RepoBuilderAgent/src
    import config as _config
    from common import chat_completion_with_retries, finalize_llm_metrics, init_llm_metrics, prompt_path
    from timeout_config import load_timeout_defaults
    from prompt_profiles import (
        apply_prompt_profile,
        prompt_profile_metadata,
        resolve_prompt_profile,
        resolve_prompt_temperature,
    )
    from repo_fingerprint import fingerprint, collect_manifest_files, collect_selected_files, learn_new_files, select_files_by_bm25
    from log_utils import log_info, log_warn, log_error, log_trace, set_trace_enabled, set_tqdm_bar, log_file_delta

    OPENAI_API_KEY = getattr(_config, "OPENAI_API_KEY", "")
    OPENAI_BASE_URL = getattr(_config, "OPENAI_BASE_URL", "https://api.openai.com/v1")
    OPENAI_MODEL = getattr(_config, "OPENAI_MODEL", "gpt-4o")

TIMEOUTS = load_timeout_defaults(
    "agent_classify",
    {
        "timeout": 120,
        "llm_max_retries": 2,
        "llm_retry_backoff_seconds": 2.0,
        "selection_timeout": 120,
        "classification_timeout": 240,
    },
)

parser = argparse.ArgumentParser(description="Analyze and classify GitHub repositories based on a given schema file.")
parser.add_argument("--input-file", default="repos.json", help="Path to input file containing repository URLs")
parser.add_argument(
    "--repo-url",
    action="append",
    default=[],
    help="Analyze a specific repository URL (can be passed multiple times). Overrides --input-file when provided.",
)
parser.add_argument("--schema", default="schemas/schema.yaml", help="Path to the schema .yaml file")
parser.add_argument("--endpoint", default=os.getenv("LLM_ENDPOINT", OPENAI_BASE_URL), help="Custom API endpoint URL")
parser.add_argument("--model", default=os.getenv("LLM_MODEL", OPENAI_MODEL), help="Model name")
parser.add_argument("--api-key", default=os.getenv("LLM_API_KEY", OPENAI_API_KEY), help="API key")
parser.add_argument("--prompt-profile", default=os.getenv("PROMPT_PROFILE", "P*"), help="Prompt profile name from RepoBuilderAgent/config/prompt_profiles.yaml (supports alias P*)")
parser.add_argument("--temperature", type=float, default=None, help="Temperature override for the model; defaults to selected prompt profile value")
parser.add_argument("--timeout", type=int, default=int(TIMEOUTS["timeout"]), help="Timeout for API requests in seconds")
parser.add_argument("--llm-max-retries", type=int, default=int(TIMEOUTS["llm_max_retries"]), help="Maximum retries for transient LLM timeouts and retryable API errors")
parser.add_argument("--llm-retry-backoff-seconds", type=float, default=float(TIMEOUTS["llm_retry_backoff_seconds"]), help="Base exponential backoff delay in seconds for LLM retries")
parser.add_argument("--selection-timeout", type=int, default=int(TIMEOUTS["selection_timeout"]), help="Timeout for file-selection LLM requests in seconds")
parser.add_argument("--classification-timeout", type=int, default=int(TIMEOUTS["classification_timeout"]), help="Timeout for final classification LLM requests in seconds")
parser.add_argument("--trace", action="store_true", help="Enable verbose trace logs")
parser.add_argument("--force", action="store_true", help="Overwrite existing analysis results")
parser.add_argument("--learn", action="store_true", help="Learn new files from LLM selections and update config")
parser.add_argument("--preprocess", action="store_true", help="Remove docs and unnecessary files after cloning")
parser.add_argument("--deletion-patterns", default="config/deletion-patterns.yaml", help="Path to YAML file with deletion patterns for preprocessing")
parser.add_argument("--results-dir", default="classification_results", help="Directory containing classification result YAML files")
parser.add_argument("--summaries-dir", default="summaries", help="Directory containing repository summary files")
parser.add_argument("--repos-dir", default="repos", help="Directory containing cloned repositories")
parser.add_argument("--analysis-dir", default="analysis", help="Directory containing aggregated analysis outputs")
parser.add_argument("--no-analysis", action="store_true", help="Skip running the analysis script after completion")
parser.add_argument(
    "--retrieval-strategy",
    default="iterative_react",
    choices=["iterative_react", "bm25"],
    help="Step 1.1 repository evidence-selection strategy.",
)
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

# Load the prompt template from the agent subrepo.
with open(prompt_path("PROMPT.md"), "r") as f:
    PROMPT_TEMPLATE = apply_prompt_profile(f.read(), PROMPT_PROFILE, "classify-step2")

with open(prompt_path("PROMPT_SELECT_FILES.md"), "r") as f:
    SELECT_FILES_PROMPT_TEMPLATE = apply_prompt_profile(f.read(), PROMPT_PROFILE, "classify-step1-selection")

sem = asyncio.Semaphore(4)

set_trace_enabled(args.trace)


def should_use_progress(total_repos: int) -> bool:
    return total_repos > 1 and sys.stderr.isatty() and not args.trace

def estimate_tokens(string: str, model_name: str) -> int:
    """Returns the number of tokens in a text string."""

    try:
        encoding = tiktoken.encoding_for_model(model_name)
    except Exception:
        encoding = tiktoken.get_encoding("cl100k_base")
    num_tokens = len(encoding.encode(string))
    return num_tokens


def pct(part: int, whole: int) -> float:
    if whole <= 0:
        return 0.0
    return round((part / whole) * 100.0, 3)


def preprocess_repository(repo_path: Path, deletion_patterns_file: str) -> None:
    """Remove docs and unnecessary files from repository based on deletion patterns config."""
    
    if not Path(deletion_patterns_file).exists():
        log_warn(f"Deletion patterns file not found: {deletion_patterns_file}. Skipping preprocessing.")
        return
    
    with open(deletion_patterns_file, "r") as f:
        patterns_config = yaml.safe_load(f)
    
    if not patterns_config:
        log_warn("Deletion patterns config is empty. Skipping preprocessing.")
        return
    
    deleted_count = 0
    
    # Delete files matching extension patterns
    extension_patterns = patterns_config.get("extension_patterns", [])
    for ext_pattern in extension_patterns:
        # Convert extension pattern (e.g., "*.o") to glob pattern
        glob_pattern = f"**/{ext_pattern}"
        matches = glob.glob(str(repo_path / glob_pattern), recursive=True)
        for file_path in matches:
            try:
                if Path(file_path).is_file():
                    Path(file_path).unlink()
                    deleted_count += 1
                    log_trace(f"Deleted file: {file_path}")
            except Exception as e:
                log_warn(f"Failed to delete file {file_path}: {e}")
    
    # Delete files matching file patterns
    file_patterns = patterns_config.get("file_patterns", [])
    for pattern in file_patterns:
        matches = glob.glob(str(repo_path / pattern), recursive=True)
        for file_path in matches:
            try:
                if Path(file_path).is_file():
                    Path(file_path).unlink()
                    deleted_count += 1
                    log_trace(f"Deleted file: {file_path}")
            except Exception as e:
                log_warn(f"Failed to delete file {file_path}: {e}")
    
    # Delete directories matching directory patterns
    directory_patterns = patterns_config.get("directory_patterns", [])
    for pattern in directory_patterns:
        matches = glob.glob(str(repo_path / pattern), recursive=True)
        for dir_path in matches:
            try:
                if Path(dir_path).is_dir():
                    shutil.rmtree(dir_path)
                    deleted_count += 1
                    log_trace(f"Deleted directory: {dir_path}")
            except Exception as e:
                log_warn(f"Failed to delete directory {dir_path}: {e}")
    
    if deleted_count > 0:
        log_info(f"Preprocessing complete: deleted {deleted_count} files/directories from {repo_path.name}")


async def update_progress(progress_state: dict, repo_name: str) -> None:
    if progress_state["bar"] is None:
        return
    async with progress_state["lock"]:
        progress_state["bar"].set_postfix_str(repo_name)
        progress_state["bar"].update(1)

def parse_llm_yaml(raw: str) -> Any:
    match = re.search(r"```(?:yaml)?\n(.*?)```", raw, re.DOTALL)
    content = match.group(1) if match else raw
    return yaml.safe_load(content)


def extract_selected_files(raw: str) -> list[str]:
    """Parse file paths from step-1 model output (YAML preferred, text fallback)."""
    candidates: list[str] = []

    try:
        parsed = parse_llm_yaml(raw)
    except Exception:
        parsed = None

    if isinstance(parsed, dict):
        for key in ("selected_files", "relevant_files", "files", "paths"):
            values = parsed.get(key)
            if isinstance(values, list):
                candidates.extend(str(v) for v in values if isinstance(v, (str, int, float)))
    elif isinstance(parsed, list):
        candidates.extend(str(v) for v in parsed if isinstance(v, (str, int, float)))

    if not candidates:
        for line in raw.splitlines():
            s = line.strip().strip("`")
            if not s:
                continue
            s = re.sub(r"^[\-*\d\.\)\s]+", "", s).strip()
            if s.lower().startswith(("selected_files:", "relevant_files:", "files:", "paths:")):
                continue
            if s.lower().startswith("path:"):
                s = s.split(":", 1)[1].strip()
            if re.match(r"^[A-Za-z0-9._\-/\[\]*?]+$", s) and ("/" in s or "." in Path(s).name):
                candidates.append(s)

    cleaned: list[str] = []
    seen: set[str] = set()
    for item in candidates:
        p = item.strip().strip('"').strip("'").lstrip("./")
        if not p or p in seen:
            continue
        seen.add(p)
        cleaned.append(p)
    return cleaned


def build_bm25_query_terms(repo_name: str) -> list[str]:
    fixed_terms = [
        "install",
        "installation",
        "build",
        "setup",
        "dependency",
        "dependencies",
        "requirements",
        "package",
        "docker",
        "workflow",
        "ci",
        "compile",
        "test",
        "make",
        "cmake",
        "gradle",
        "maven",
        "cargo",
        "npm",
        "pip",
    ]
    repo_terms = re.findall(r"[a-z0-9]+", repo_name.lower())
    return fixed_terms + repo_terms

async def analyze_repository(repo_url: str, summary_dir: Path, output_dir: Path, results_dir: Path, progress_state: dict, force: bool = False, learn: bool = False) -> None:
    async with sem:
        repo_name = repo_url.split("/")[-1].replace(".git", "")
        llm_metrics_path = results_dir / f"{repo_name}.llm-metrics.yaml"
        llm_metrics = init_llm_metrics(repo_url, args.model, args.endpoint, args.timeout, args.llm_max_retries)
        llm_metrics["prompt_profile"] = prompt_profile_metadata(PROMPT_PROFILE, EFFECTIVE_TEMPERATURE)
        try:
            log_info(f"Processing {repo_url}...")
            output_path = results_dir / f"{repo_name}.yaml"
            metrics_path = results_dir / f"{repo_name}.token-metrics.yaml"

            if output_path.exists() and not force:
                log_info(f"Skipping {repo_url}: existing result found at {output_path}")
                return
            
            if output_path.exists() and force:
                log_info(f"Overwriting existing result for {repo_url}")

            repo_path = output_dir / repo_name
            if not repo_path.exists():
                log_info(f"Cloning {repo_url} -> {repo_path}")
                os.system(f"git clone {repo_url} {repo_path}")
            
            # Preprocess repository if flag is enabled
            if args.preprocess:
                log_info(f"Preprocessing {repo_name}...")
                preprocess_repository(repo_path, args.deletion_patterns)
            
            default_selected_files = [
                "README.md",
                "README.rst",
                "pyproject.toml",
                "requirements.txt",
                "package.json",
                "go.mod",
                "Cargo.toml",
                "Dockerfile",
                "docker-compose.yml",
                "docker-compose.yaml",
                ".env.example",
                ".github/workflows/*.yml",
                ".github/workflows/*.yaml",
            ]

            log_trace(f"Begin analysis for {repo_name}")
            # Step 1: structure-only context -> select relevant files.
            structure_summary = fingerprint(
                format="md",
                repo_path=str(repo_path),
                structure_only=True,
                include_tree=True,
                context="step1-structure",
            )
            structure_summary_path = summary_dir / f"{repo_name}.structure.md"
            with open(structure_summary_path, "w") as f:
                f.write(structure_summary)

            selected_files = default_selected_files.copy()
            step1_tokens = 0
            if args.retrieval_strategy == "bm25":
                selected_files = select_files_by_bm25(repo_path, build_bm25_query_terms(repo_name))
                if not selected_files:
                    log_warn(f"BM25 selection returned no files for {repo_url}; using defaults.")
                    selected_files = default_selected_files.copy()
                log_info(f"Selected {len(selected_files)} files for {repo_name} via BM25 retrieval.")
            else:
                selection_prompt = (
                    SELECT_FILES_PROMPT_TEMPLATE
                    .replace("{{REPO_URL}}", repo_url)
                    .replace("{{STRUCTURE_CONTENT}}", structure_summary)
                )
                step1_tokens = estimate_tokens(selection_prompt, args.model)
                log_info(f"Prompt tokens for {repo_name} step1-selection: {step1_tokens:,}")

                try:
                    log_info(f"Sending step1-selection prompt for {repo_url}...")
                    selection_response = await chat_completion_with_retries(
                        client=client,
                        model=args.model,
                        temperature=EFFECTIVE_TEMPERATURE,
                        messages=[{"role": "user", "content": selection_prompt}],
                        repo_url=repo_url,
                        phase="classify-step1-selection",
                        metrics=llm_metrics,
                        timeout_seconds=args.selection_timeout,
                        max_retries=args.llm_max_retries,
                        retry_backoff_seconds=args.llm_retry_backoff_seconds,
                    )
                    raw_selection = (selection_response.choices[0].message.content or "").strip()
                    log_trace(f"Received step1-selection response for {repo_name}")
                    if selection_response.usage:
                        log_info(f"[TOKENS] {json.dumps({'phase': 'classify-step1', 'repo': repo_url, 'prompt_tokens': selection_response.usage.prompt_tokens, 'completion_tokens': selection_response.usage.completion_tokens, 'total_tokens': selection_response.usage.total_tokens})}")
                    parsed_selection = extract_selected_files(raw_selection)
                    if parsed_selection:
                        selected_files = parsed_selection
                    else:
                        log_warn(f"No selected files returned for {repo_url}; using defaults.")
                except httpx.HTTPError as e:
                    log_warn(f"Selection HTTP error for {repo_url}: {e}. Using default file selection.")
                except ssl.SSLError as e:
                    log_warn(f"Selection SSL error for {repo_url}: {e}. Using default file selection.")
                except APITimeoutError as e:
                    log_warn(f"Selection timeout for {repo_url}: {e}. Using default file selection.")
                except APIError as e:
                    log_warn(f"Selection API error for {repo_url}: {e}. Using default file selection.")

            selected_files_path = summary_dir / f"{repo_name}.selected-files.yaml"
            with open(selected_files_path, "w") as f:
                yaml.dump({"retrieval_strategy": args.retrieval_strategy, "selected_files": selected_files}, f, sort_keys=False, allow_unicode=True)

            # Step 2: only selected file contents are provided to the classification prompt.
            summary = fingerprint(
                format="md",
                repo_path=str(repo_path),
                selected_files=selected_files,
                include_tree=False,
                context="step2-selected",
            )
            summary_path = summary_dir / f"{repo_name}.md"
            with open(summary_path, "w") as f:
                f.write(summary)
            log_info(f"Generated structure summary at {structure_summary_path}")
            log_info(f"Selected files list saved at {selected_files_path}")
            log_info(f"Generated selected-files summary at {summary_path}")

            # Token accounting for baseline vs two-step prompts.
            # Baseline: deterministic collection of all manifest files (no LLM filtering).
            baseline_summary = fingerprint(
                format="md",
                repo_path=str(repo_path),
                structure_only=False,
                selected_files=None,
                include_tree=True,
                context="baseline-full",
            )
            
            # Trace: file set delta between baseline and LLM-selected.
            baseline_files_tuples = collect_manifest_files(repo_path)
            selected_files_tuples = collect_selected_files(repo_path, selected_files)
            baseline_file_names = [name for name, _ in baseline_files_tuples]
            selected_file_names = [name for name, _ in selected_files_tuples]
            log_file_delta(repo_name, baseline_file_names, selected_file_names)
            
            baseline_prompt = PROMPT_TEMPLATE.replace("{{REPO_URL}}", repo_url).replace("{{SUMMARY_CONTENT}}", baseline_summary)

            prompt = PROMPT_TEMPLATE.replace("{{REPO_URL}}", repo_url).replace("{{SUMMARY_CONTENT}}", summary)
            step2_tokens = estimate_tokens(prompt, args.model)
            baseline_tokens = estimate_tokens(baseline_prompt, args.model)
            two_step_total_tokens = step1_tokens + step2_tokens
            log_info(f"Prompt tokens for {repo_name} step2-classification: {step2_tokens:,}")
            log_info(f"Prompt tokens for {repo_name} baseline-full: {baseline_tokens:,}")
            log_info(f"Prompt tokens for {repo_name} two-step-total: {two_step_total_tokens:,}")

            metrics = {
                "repo": repo_url,
                "model": args.model,
                "retrieval_strategy": args.retrieval_strategy,
                "prompt_profile": prompt_profile_metadata(PROMPT_PROFILE, EFFECTIVE_TEMPERATURE),
                "tokens": {
                    "baseline_full_classification": baseline_tokens,
                    "step1_selection_prompt": step1_tokens,
                    "step2_reduced_classification": step2_tokens,
                    "two_step_total": two_step_total_tokens,
                },
                "deltas": {
                    "step2_vs_baseline": step2_tokens - baseline_tokens,
                    "two_step_total_vs_baseline": two_step_total_tokens - baseline_tokens,
                },
                "reductions_percent": {
                    "step2_vs_baseline": round(100.0 - pct(step2_tokens, baseline_tokens), 3),
                    "two_step_total_vs_baseline": round(100.0 - pct(two_step_total_tokens, baseline_tokens), 3),
                },
                "selected_files_count": len(selected_files),
                "files": {
                    "structure_summary": str(structure_summary_path),
                    "selected_files": str(selected_files_path),
                    "reduced_summary": str(summary_path),
                },
            }

            with open(metrics_path, "w") as f:
                yaml.dump(metrics, f, sort_keys=False, allow_unicode=True)

            log_info(f"Token metrics saved at {metrics_path}")
            log_info(f"Sending step2-classification prompt for {repo_url}...")
            try:
                response = await chat_completion_with_retries(
                    client=client,
                    model=args.model,
                    temperature=EFFECTIVE_TEMPERATURE,
                    messages=[{"role": "user", "content": prompt}],
                    repo_url=repo_url,
                    phase="classify-step2",
                    metrics=llm_metrics,
                    timeout_seconds=args.classification_timeout,
                    max_retries=args.llm_max_retries,
                    retry_backoff_seconds=args.llm_retry_backoff_seconds,
                )
                raw = (response.choices[0].message.content or "").strip()
                log_trace(f"Received step2-classification response for {repo_name}")
                if response.usage:
                    log_info(f"[TOKENS] {json.dumps({'phase': 'classify-step2', 'repo': repo_url, 'prompt_tokens': response.usage.prompt_tokens, 'completion_tokens': response.usage.completion_tokens, 'total_tokens': response.usage.total_tokens})}")
                if args.trace:
                    print(raw)

                try:
                    parsed = parse_llm_yaml(raw)
                    with open(output_path, "w") as f:
                        yaml.dump(parsed, f, sort_keys=False, allow_unicode=True)
                except yaml.YAMLError as e:
                    log_warn(f"Failed to parse result for {repo_name}: {e}")
                    with open(output_path, "w") as f:
                        yaml.dump({"error": "parse_failed", "raw": raw}, f, sort_keys=False, allow_unicode=True)

                log_info(f"Saved result to {output_path}")

            except httpx.HTTPError as e:
                log_warn(f"HTTP error for {repo_url}: {e}")
            except ssl.SSLError as e:
                log_warn(f"SSL error for {repo_url}: {e}")
            except APITimeoutError as e:
                log_warn(f"OpenAI timeout for {repo_url}: {e}")
            except APIError as e:
                log_warn(f"OpenAI API error for {repo_url}: {e}")

            # Learning: identify and add new files from LLM selections if --learn is enabled
            if learn and selected_files and baseline_file_names:
                new_files = [f for f in selected_files if f not in baseline_file_names]
                if new_files:
                    learn_result = learn_new_files(new_files)
                    log_info(f"Learning: added {learn_result['added']} new file patterns for {repo_name}")
                    if args.trace:
                        log_trace(f"Added files: {learn_result.get('added_files', [])}")
                        log_trace(f"Added patterns: {learn_result.get('added_patterns', [])}")
                        log_trace(
                            f"Skipped project-specific paths: "
                            f"{learn_result.get('skipped_project_specific', [])}"
                        )

            with open(llm_metrics_path, "w", encoding="utf-8") as f:
                yaml.dump(finalize_llm_metrics(llm_metrics), f, sort_keys=False, allow_unicode=True)
            log_info(f"LLM metrics saved at {llm_metrics_path}")

        except Exception as e:
            log_error(f"Unexpected error while processing {repo_url}: {e}")
            with open(llm_metrics_path, "w", encoding="utf-8") as f:
                yaml.dump(finalize_llm_metrics(llm_metrics), f, sort_keys=False, allow_unicode=True)
        finally:
            await update_progress(progress_state, repo_name)

async def main():

    # Load repo URLs from CLI override or input file.
    if args.repo_url:
        repos = [url.strip() for url in args.repo_url if url and url.strip()]
        # Preserve order while removing duplicates.
        repos = list(dict.fromkeys(repos))
        log_info(f"Using {len(repos)} repository URL(s) from --repo-url")
    else:
        with open(args.input_file, "r") as f:
            repos = [item["url"] for item in json.load(f)]

    if not repos:
        log_error("No repositories to analyze. Provide --repo-url or a non-empty --input-file.")
        return

    workspace_root = Path(args.input_file).parent

    # Create output directory named 'repos' if it doesn't exist
    repos_dir = workspace_root / args.repos_dir
    repos_dir.mkdir(parents=True, exist_ok=True)

    # Create summaries directory containing Markdown files for each repository
    summaries_dir = workspace_root / args.summaries_dir
    summaries_dir.mkdir(parents=True, exist_ok=True)

    # Create classification results directory for storing the final classified YAML output
    results_dir = workspace_root / args.results_dir
    results_dir.mkdir(parents=True, exist_ok=True)

    # Run all analyses concurrently
    progress_bar = None
    if should_use_progress(len(repos)):
        progress_bar = tqdm(total=len(repos), desc="Analyzing repos", unit="repo", dynamic_ncols=True)

    progress_state = {
        "lock": asyncio.Lock(),
        "bar": progress_bar,
    }
    set_tqdm_bar(progress_state["bar"])
    log_info(f"Starting analysis for {len(repos)} repositories")
    tasks = [analyze_repository(repo, summaries_dir, repos_dir, results_dir, progress_state, force=args.force, learn=args.learn) for repo in repos]
    try:
        results = await asyncio.gather(*tasks, return_exceptions=True)
    finally:
        if progress_state["bar"] is not None:
            progress_state["bar"].close()
        set_tqdm_bar(None)

    failures = 0
    for repo, result in zip(repos, results):
        if isinstance(result, Exception):
            failures += 1
            log_error(f"Task failed for {repo}: {result}")

    if failures:
        log_warn(f"Completed with {failures} task-level failures.")

    log_info("Done.")

    # Run analysis script to aggregate results and generate metrics, unless --no-analysis is specified.
    if not args.no_analysis:
        analysis_script = Path(__file__).parent / "parse_results.py"
        log_info("Running analysis script...")
        subprocess.run(
            [
                sys.executable,
                str(analysis_script),
                "--results-dir", args.results_dir,
                "--summaries-dir", args.summaries_dir,
                "--analysis-dir", args.analysis_dir,
            ],
            check=False,
        )

if __name__ == "__main__":
    asyncio.run(main())