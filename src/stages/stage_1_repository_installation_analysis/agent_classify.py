import json
from openai import AsyncOpenAI, APITimeoutError, APIError
import tiktoken
import argparse
from pathlib import Path
import asyncio
import os
import dataclasses
import httpx
import ssl
import re
import subprocess
import sys
import yaml
import math
from tqdm import tqdm
from typing import Any, cast
from typing_extensions import TypedDict


try:
    from RepoBuilderAgent.src.core.config import OPENAI_API_KEY, OPENAI_BASE_URL, OPENAI_MODEL
    from RepoBuilderAgent.src.stages.stage_1_repository_installation_analysis.l1_relevant_file_discovery import select_files_by_iterative_react
    from RepoBuilderAgent.src.stages.stage_1_repository_installation_analysis.architecture_state_graph import run_architecture_state_graph as _run_architecture_state_graph_impl
    from RepoBuilderAgent.src.stages.stage_1_repository_installation_analysis.scratchpad_payloads import build_architecture_scratchpad_payload, LoopOutcome, TokenCounts, SummaryPaths
    from RepoBuilderAgent.src.core.common import (
        chat_completion_with_retries,
        finalize_llm_metrics,
        init_llm_metrics,
        load_shared_repository_state,
        upsert_shared_repository_state,
        prompt_path,
        should_use_progress,
        ensure_repo_checkout
    )
    from RepoBuilderAgent.src.core.chat_model_factory import make_prebuilt_chat_model_factory
    from RepoBuilderAgent.src.core.timeout_config import load_timeout_defaults
    from RepoBuilderAgent.src.core.prompt_profiles import (
        apply_prompt_profile,
        prompt_profile_metadata,
        resolve_prompt_profile,
        resolve_prompt_temperature,
    )
    from RepoBuilderAgent.src.retrieval.repo_fingerprint import fingerprint, collect_manifest_files, collect_selected_files, collect_retrieval_candidates, learn_new_files, select_files_by_bm25, select_files_by_bm25_budgeted
    from RepoBuilderAgent.src.core.log_utils import log_info, log_warn, log_error, log_trace, set_dump_prompts_dir, set_trace_enabled, set_tqdm_bar, log_file_delta
    from RepoBuilderAgent.src.core.repo_cleanup import delete_files_build_context, get_files_to_delete
    from RepoBuilderAgent.src.metrics.eval_metrics_lib import load_gt_for_repo
    from RepoBuilderAgent.src.core.llm_yaml import parse_llm_yaml
    from RepoBuilderAgent.src.core.agent_runtime import ClassifyConfig, ClassifyRuntime, RepoRef
    from RepoBuilderAgent.src.agent_tools.react_loop_tools import extract_finalize_answer, hit_step_limit
except ImportError:
    # Fallback for direct script execution from RepoBuilderAgent/src
    import core.config as _config
    from stages.stage_1_repository_installation_analysis.l1_relevant_file_discovery import select_files_by_iterative_react
    from stages.stage_1_repository_installation_analysis.architecture_state_graph import run_architecture_state_graph as _run_architecture_state_graph_impl
    from stages.stage_1_repository_installation_analysis.scratchpad_payloads import build_architecture_scratchpad_payload, LoopOutcome, TokenCounts, SummaryPaths
    from core.common import (
        chat_completion_with_retries,
        finalize_llm_metrics,
        init_llm_metrics,
        load_shared_repository_state,
        upsert_shared_repository_state,
        prompt_path,
        should_use_progress,
        ensure_repo_checkout
    )
    from core.chat_model_factory import make_prebuilt_chat_model_factory
    from core.timeout_config import load_timeout_defaults
    from core.prompt_profiles import (
        apply_prompt_profile,
        prompt_profile_metadata,
        resolve_prompt_profile,
        resolve_prompt_temperature,
    )
    from retrieval.repo_fingerprint import fingerprint, collect_manifest_files, collect_selected_files, collect_retrieval_candidates, learn_new_files, select_files_by_bm25, select_files_by_bm25_budgeted
    from core.log_utils import log_info, log_warn, log_error, log_trace, set_dump_prompts_dir, set_trace_enabled, set_tqdm_bar, log_file_delta
    from core.repo_cleanup import delete_files_build_context, get_files_to_delete
    from metrics.eval_metrics_lib import load_gt_for_repo
    from core.llm_yaml import parse_llm_yaml
    from core.agent_runtime import ClassifyConfig, ClassifyRuntime, RepoRef
    from agent_tools.react_loop_tools import extract_finalize_answer, hit_step_limit

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
parser.add_argument("--dump-prompts", default=None, metavar="PATH", help="Write each rendered prompt to PATH/<repo>/<phase>.<n>.txt before the LLM call")
parser.add_argument("--force", action="store_true", help="Overwrite existing analysis results")
parser.add_argument("--learn", action="store_true", help="Learn new files from LLM selections and update config")
parser.add_argument("--dataset-dir", default=None, help="Path to RepoBuilderDataset; the repo's declarative files_to_delete (docs + CI) is stripped after clone so the classifier never sees install docs.")
parser.add_argument("--results-dir", default="classification_results", help="Directory containing classification result YAML files")
parser.add_argument("--summaries-dir", default="summaries", help="Directory containing repository summary files")
parser.add_argument("--repos-dir", default="repos", help="Directory containing cloned repositories")
parser.add_argument("--analysis-dir", default="analysis", help="Directory containing aggregated analysis outputs")
parser.add_argument("--scratchpad-dir", default="", help="Optional directory to write per-repo architecture scratchpads")
parser.add_argument("--exploration-enabled", action=argparse.BooleanOptionalAction, default=True, help="Enable exploration-stage artifact generation")
parser.add_argument("--synthesis-enabled", action=argparse.BooleanOptionalAction, default=True, help="Enable synthesis-stage artifact generation")
parser.add_argument("--validation-enabled", action=argparse.BooleanOptionalAction, default=True, help="Enable validation-stage artifact generation")
parser.add_argument("--scratchpads-enabled", action=argparse.BooleanOptionalAction, default=True, help="Enable architecture scratchpad artifact generation")
parser.add_argument("--snippet-tools", action=argparse.BooleanOptionalAction, default=False, help="Enable Dockerfile snippet tools (get_dockerfile_snippet) in L2 synthesis agents")
parser.add_argument("--no-analysis", action="store_true", help="Skip running the analysis script after completion")
parser.add_argument(
    "--retrieval-strategy",
    default="iterative_react",
    choices=["iterative_react", "bm25", "neural_embedding", "one_shot_fingerprint", "one_shot_fingerprint_budgeted"],
    help="Step 1.1 repository evidence-selection strategy.",
)
parser.add_argument(
    "--embedding-model",
    default=os.getenv("LLM_EMBEDDING_MODEL", os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")),
    help="Embedding model used when --retrieval-strategy=neural_embedding.",
)
parser.add_argument(
    "--react-max-steps",
    type=int,
    default=4,
    help="Maximum selection iterations for --retrieval-strategy=iterative_react.",
)
parser.add_argument(
    "--react-max-total-files",
    type=int,
    default=24,
    help="Maximum total files retained across iterative_react steps.",
)
parser.add_argument(
    "--react-final-cap",
    type=int,
    default=12,
    help="Hard cap applied after L1 agent output normalization/reranking.",
)
parser.add_argument(
    "--synthesis-react-max-steps",
    type=int,
    default=3,
    help="Maximum L2 synthesis loop iterations.",
)
parser.add_argument(
    "--synthesis-review-rounds",
    type=int,
    default=1,
    help="Number of L2.5 reviewer rounds to run after initial L2 generator output.",
)
parser.add_argument(
    "--validation-react-max-steps",
    type=int,
    default=3,
    help="Maximum classify validation ReAct loop iterations.",
)
parser.add_argument(
    "--step2-token-budget",
    type=int,
    default=12000,
    help="Hard token budget target for Step 2 classification prompt (0 disables budget pruning).",
)
parser.add_argument(
    "--max-input-tokens",
    type=int,
    default=60000,
    help="Max input tokens the model endpoint accepts. Large repo-structure summaries are "
    "truncated to fit so classification doesn't 400 on huge repos (e.g. linux ~306k tokens). "
    "Set below the model's hard cap to leave room for the ReAct loop.",
)
parser.add_argument(
    "--synthesis-subagents-enabled",
    action=argparse.BooleanOptionalAction,
    default=True,
    help="Enable parallel synthesis sub-agent passes before iterative L2 convergence.",
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
if args.dump_prompts:
    set_dump_prompts_dir(args.dump_prompts)


_new_prebuilt_chat_model = make_prebuilt_chat_model_factory(
    model=args.model,
    temperature=EFFECTIVE_TEMPERATURE,
    api_key=args.api_key,
    base_url=args.endpoint,
    max_retries=args.llm_max_retries,
    http_async_client=_http_client,
)

def estimate_tokens(string: str, model_name: str) -> int:
    """Returns the number of tokens in a text string."""

    try:
        encoding = tiktoken.encoding_for_model(model_name)
    except Exception:
        encoding = tiktoken.get_encoding("cl100k_base")
    num_tokens = len(encoding.encode(string))
    return num_tokens


def _truncate_to_token_budget(text: str, model_name: str, max_tokens: int, *, label: str = "content") -> str:
    """Truncate text to at most max_tokens (exact, via the tokenizer) for LLM prompts.

    Huge repos (e.g. linux's ~306k-token structure summary) otherwise overflow the
    endpoint's input cap and 400 the whole classification. Keeps the head (top-of-tree
    and manifests come first in the fingerprint) and appends a notice; the agent can
    still reach the rest via list_tree/search_pattern/read_file. Callers retain the full
    text on disk — only the prompt copy is trimmed."""
    if max_tokens <= 0 or not text:
        return text
    try:
        encoding = tiktoken.encoding_for_model(model_name)
    except Exception:
        encoding = tiktoken.get_encoding("cl100k_base")
    tokens = encoding.encode(text)
    if len(tokens) <= max_tokens:
        return text
    notice = (
        f"\n\n... [{label} truncated to fit the {max_tokens}-token input budget; "
        f"use list_tree/search_pattern/read_file to inspect the rest]"
    )
    keep = max(1, max_tokens - len(encoding.encode(notice)))
    head = encoding.decode(tokens[:keep])
    newline = head.rfind("\n")
    if newline > 0:
        head = head[:newline]
    log_warn(
        f"[{label}] {len(tokens)} tokens exceeds the {max_tokens}-token budget; "
        f"truncated for the LLM prompt (full copy retained on disk)."
    )
    return head + notice


def pct(part: int, whole: int) -> float:
    if whole <= 0:
        return 0.0
    return round((part / whole) * 100.0, 3)


async def update_progress(progress_state: dict, repo_name: str) -> None:
    if progress_state["bar"] is None:
        return
    async with progress_state["lock"]:
        progress_state["bar"].set_postfix_str(repo_name)
        progress_state["bar"].update(1)

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


def build_bm25_query_terms(repo_name: str, extra_terms: list[str] | None = None) -> list[str]:
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
    merged = fixed_terms + repo_terms + (extra_terms or [])
    deduped: list[str] = []
    seen: set[str] = set()
    for term in merged:
        normalized = str(term).strip().lower()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        deduped.append(normalized)
    return deduped


def build_embedding_query_text(repo_name: str, extra_terms: list[str] | None = None) -> str:
    terms = build_bm25_query_terms(repo_name, extra_terms)
    return (
        "Find repository files that best explain how to install, build, verify, and package this project. "
        "Prefer manifests, READMEs, Dockerfiles, CI workflows, and dependency configuration. "
        f"Repository: {repo_name}. Keywords: {' '.join(terms)}"
    )


def build_prior_failure_retrieval_hints(shared_state: dict | None) -> tuple[list[str], str]:
    if not isinstance(shared_state, dict):
        return [], ""

    terms: list[str] = []
    summary_lines: list[str] = []

    stages = shared_state.get("stages")
    if isinstance(stages, dict):
        pipeline_stage = stages.get("pipeline")
        if isinstance(pipeline_stage, dict):
            user_constraints = pipeline_stage.get("user_constraints")
            if isinstance(user_constraints, dict) and user_constraints:
                constraints_blob = yaml.dump(user_constraints, sort_keys=False, allow_unicode=True)
                terms.extend(re.findall(r"[a-z0-9_\-]+", constraints_blob.lower()))
                summary_lines.append("- user_constraints present (pipeline contract)")

    signals = shared_state.get("signals")
    if not isinstance(signals, dict):
        dedup_terms: list[str] = []
        seen: set[str] = set()
        for term in terms:
            if len(term) < 3 or term in seen:
                continue
            seen.add(term)
            dedup_terms.append(term)
        return dedup_terms[:20], "\n".join(summary_lines[:8])

    failure_hints = signals.get("failure_hints")
    if not isinstance(failure_hints, list) or not failure_hints:
        dedup_terms: list[str] = []
        seen: set[str] = set()
        for term in terms:
            if len(term) < 3 or term in seen:
                continue
            seen.add(term)
            dedup_terms.append(term)
        return dedup_terms[:20], "\n".join(summary_lines[:8])
    for hint in failure_hints[-8:]:
        if not isinstance(hint, dict):
            continue
        category = str(hint.get("category", "")).strip().lower()
        confidence = str(hint.get("confidence", "")).strip().lower()
        evidence = hint.get("evidence")
        if category:
            terms.extend(re.findall(r"[a-z0-9_]+", category))
        if isinstance(evidence, list):
            for item in evidence[:3]:
                if isinstance(item, str):
                    terms.extend(re.findall(r"[a-z0-9_]+", item.lower()))
        if category:
            summary_lines.append(f"- {category} ({confidence or 'unknown'})")

    dedup_terms: list[str] = []
    seen: set[str] = set()
    for term in terms:
        if len(term) < 3 or term in seen:
            continue
        seen.add(term)
        dedup_terms.append(term)

    return dedup_terms[:20], "\n".join(summary_lines[:8])


def cosine_similarity(left: list[float], right: list[float]) -> float:
    numerator = sum(a * b for a, b in zip(left, right))
    left_norm = math.sqrt(sum(a * a for a in left))
    right_norm = math.sqrt(sum(b * b for b in right))
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    return numerator / (left_norm * right_norm)


def build_local_dense_embedding(text: str, dimensions: int = 256) -> list[float]:
    vector = [0.0] * dimensions
    tokens = re.findall(r"[a-z0-9]+", text.lower())

    for token in tokens:
        vector[hash(f"tok:{token}") % dimensions] += 1.0

    compact = text.lower().replace("\n", " ")
    for index in range(max(len(compact) - 2, 0)):
        trigram = compact[index:index + 3]
        if trigram.strip():
            vector[hash(f"tri:{trigram}") % dimensions] += 0.25

    norm = math.sqrt(sum(value * value for value in vector))
    if norm == 0.0:
        return vector
    return [value / norm for value in vector]


def select_files_by_local_embedding(query_text: str, candidates: list[tuple[str, str]], top_k: int = 12) -> list[str]:
    high_signal_names = {
        "readme.md",
        "readme.rst",
        "readme.txt",
        "install.md",
        "install.txt",
        "pyproject.toml",
        "requirements.txt",
        "package.json",
        "package-lock.json",
        "yarn.lock",
        "pnpm-lock.yaml",
        "go.mod",
        "cargo.toml",
        "dockerfile",
        "docker-compose.yml",
        "docker-compose.yaml",
        "makefile",
        "cmakelists.txt",
        "pom.xml",
        "build.gradle",
        "build.gradle.kts",
        "settings.gradle",
        "settings.gradle.kts",
        "gradlew",
        "gemfile",
        "composer.json",
        "setup.py",
        "setup.cfg",
        "pipfile",
        "pipfile.lock",
        ".env.example",
    }
    query_embedding = build_local_dense_embedding(query_text)
    ranked: list[tuple[float, str]] = []

    for rel, content in candidates:
        candidate_embedding = build_local_dense_embedding(f"Path: {rel}\n\nContent:\n{content}")
        score = cosine_similarity(query_embedding, candidate_embedding)
        rel_lower = rel.lower()
        basename = Path(rel_lower).name
        if basename in high_signal_names:
            score += 0.4
        elif basename.startswith(("readme", "install")):
            score += 0.2
        if rel_lower.startswith(".github/workflows/"):
            score += 0.05
        ranked.append((score, rel))

    ranked.sort(key=lambda item: (-item[0], item[1]))
    return [rel for _, rel in ranked[:top_k]]


async def select_files_by_neural_embedding(
    repo_name: str,
    repo_path: Path,
    top_k: int = 12,
    extra_terms: list[str] | None = None,
) -> list[str]:
    candidates = collect_retrieval_candidates(repo_path)
    if not candidates:
        return []

    query_text = build_embedding_query_text(repo_name, extra_terms)
    candidate_payloads = [f"Path: {rel}\n\nContent:\n{content}" for rel, content in candidates]

    try:
        query_response = await client.embeddings.create(model=args.embedding_model, input=[query_text])
        query_embedding = query_response.data[0].embedding

        ranked: list[tuple[float, str]] = []
        batch_size = 32
        for start in range(0, len(candidate_payloads), batch_size):
            batch_payloads = candidate_payloads[start:start + batch_size]
            batch_candidates = candidates[start:start + batch_size]
            batch_response = await client.embeddings.create(model=args.embedding_model, input=batch_payloads)
            for item, (rel, _) in zip(batch_response.data, batch_candidates):
                score = cosine_similarity(query_embedding, item.embedding)
                ranked.append((score, rel))

        ranked.sort(key=lambda item: (-item[0], item[1]))
        return [rel for _, rel in ranked[:top_k]]
    except httpx.HTTPError as e:
        log_warn(f"Embedding retrieval HTTP error for {repo_name}: {e}. Falling back to local dense retrieval.")
    except ssl.SSLError as e:
        log_warn(f"Embedding retrieval SSL error for {repo_name}: {e}. Falling back to local dense retrieval.")
    except APITimeoutError as e:
        log_warn(f"Embedding retrieval timeout for {repo_name}: {e}. Falling back to local dense retrieval.")
    except APIError as e:
        log_warn(f"Embedding retrieval API error for {repo_name}: {e}. Falling back to local dense retrieval.")
    except Exception as e:
        log_warn(f"Embedding retrieval unexpected error for {repo_name}: {e}. Falling back to local dense retrieval.")

    return select_files_by_local_embedding(query_text, candidates, top_k=top_k)


class AgentPayload(TypedDict, total=False):
    thought: str
    selected_files: list[str]
    hypothesis_updates: list[str]
    risk_updates: list[str]
    checks: dict[str, dict[str, str]]
    warnings: list[str]
    done: bool


def _extract_agent_payload(result: dict[str, Any]) -> AgentPayload:
    # (1) structured_response if dict.
    payload = result.get("structured_response")
    if isinstance(payload, dict):
        return cast(AgentPayload, payload)

    messages = result.get("messages") or []

    # (2) prefer the finalize tool-call's answer when present.
    finalize_answer = extract_finalize_answer(messages)
    if finalize_answer is not None:
        try:
            parsed = parse_llm_yaml(finalize_answer)
            if isinstance(parsed, dict):
                return cast(AgentPayload, parsed)
        except Exception:
            pass

    # (3) trailing-message YAML fallback.
    if messages:
        last = messages[-1]
        content = getattr(last, "content", "")
        if isinstance(content, str):
            try:
                parsed = parse_llm_yaml(content)
                if isinstance(parsed, dict):
                    return cast(AgentPayload, parsed)
            except Exception:
                pass

    # (4) {} — surface the silent recursion-limit failure.
    if hit_step_limit(result):
        log_warn(
            "[classify] ReAct agent yielded empty payload after hitting the LangGraph "
            "recursion_limit (placeholder 'Sorry, need more steps...'); returning {}."
        )
    return {}


def _extract_agent_trace(result: dict[str, Any]) -> list[dict[str, Any]]:
    trace: list[dict[str, Any]] = []
    for idx, message in enumerate(result.get("messages") or [], start=1):
        role = getattr(message, "type", "unknown")
        content = getattr(message, "content", "")
        tool_calls = getattr(message, "tool_calls", None)
        trace.append(
            {
                "step": idx,
                "role": role,
                "content": content if isinstance(content, str) else str(content),
                "tool_calls": tool_calls or [],
            }
        )
    return trace


def _path_matches_manifest_build(path: str) -> bool:
    lower = path.lower()
    return any(
        marker in lower
        for marker in (
            "package.json",
            "pyproject.toml",
            "requirements",
            "go.mod",
            "cargo.toml",
            "pom.xml",
            "build.gradle",
            "build.gradle.kts",
            "makefile",
            "cmakelists",
            "setup.py",
            "setup.cfg",
        )
    )


def _path_matches_verification_hint(path: str) -> bool:
    lower = path.lower()
    return any(
        marker in lower
        for marker in (
            "test",
            "spec",
            ".github/workflows",
            "ci",
            "dockerfile",
            "docker-compose",
            "verify",
        )
    )


def _estimate_step2_prompt_tokens_from_selected_files(repo_path: Path, repo_url: str, selected_files: list[str]) -> int:
    overhead = estimate_tokens(
        PROMPT_TEMPLATE.replace("{{REPO_URL}}", repo_url).replace("{{SUMMARY_CONTENT}}", ""),
        args.model,
    )
    selected_tuples = collect_selected_files(repo_path, selected_files)
    content_tokens = 0
    for _, content in selected_tuples:
        content_tokens += estimate_tokens(content, args.model) + 24
    return overhead + content_tokens


def _ensure_required_category(
    picked: list[str],
    ranked_rels: list[str],
    predicate,
) -> list[str]:
    if any(predicate(path) for path in picked):
        return picked

    required = next((rel for rel in ranked_rels if predicate(rel)), None)
    if not required:
        return picked

    if required in picked:
        return picked

    if not picked:
        return [required]

    replace_idx = len(picked) - 1
    for idx in range(len(picked) - 1, -1, -1):
        if not predicate(picked[idx]):
            replace_idx = idx
            break

    picked[replace_idx] = required
    return list(dict.fromkeys(picked))


def _relevance_score_for_path(path: str, position: int) -> int:
    lower = path.lower()
    score = 0
    if any(
        marker in lower
        for marker in (
            "package.json",
            "pyproject.toml",
            "requirements",
            "go.mod",
            "cargo.toml",
            "pom.xml",
            "build.gradle",
            "dockerfile",
            "docker-compose",
            ".github/workflows",
            "readme",
            "install",
            "makefile",
            "cmakelists",
        )
    ):
        score += 100
    if lower.endswith((".toml", ".json", ".yaml", ".yml", ".md", ".txt", ".xml")):
        score += 15
    if lower.startswith(("docs/", "examples/", "benchmark/", "benchmarks/")):
        score -= 20
    score += max(0, 20 - position)
    return max(1, score)


def _select_files_by_value_per_token(
    *,
    repo_path: Path,
    repo_url: str,
    selected_files: list[str],
    token_budget: int,
) -> list[str]:
    if token_budget <= 0 or not selected_files:
        return selected_files

    fixed_overhead = estimate_tokens(
        PROMPT_TEMPLATE.replace("{{REPO_URL}}", repo_url).replace("{{SUMMARY_CONTENT}}", ""),
        args.model,
    )
    # Keep a safety reserve so the final prompt remains under budget after formatting overhead.
    summary_budget = max(1000, int((token_budget - fixed_overhead) * 0.8))

    selected_tuples = collect_selected_files(repo_path, selected_files)
    if not selected_tuples:
        return selected_files

    ranked = _rank_selected_files_by_value_density(selected_tuples)
    ranked_rels = [rel for _, rel, _ in ranked]

    picked: list[str] = []
    used = 0
    for _, rel, token_cost in ranked:
        if used + token_cost > summary_budget and picked:
            continue
        picked.append(rel)
        used += token_cost

    if not picked:
        picked.append(ranked[0][1])

    # Quality guardrails: keep at least one build/manifest clue and one verification clue when available.
    picked = _ensure_required_category(picked, ranked_rels, _path_matches_manifest_build)
    picked = _ensure_required_category(picked, ranked_rels, _path_matches_verification_hint)

    return picked


def _rank_selected_files_by_value_density(selected_tuples: list[tuple[str, str]]) -> list[tuple[float, str, int]]:
    ranked: list[tuple[float, str, int]] = []
    for idx, (rel, content) in enumerate(selected_tuples):
        token_cost = max(50, estimate_tokens(content, args.model) + 24)
        relevance = _relevance_score_for_path(rel, idx)
        density = relevance / token_cost
        ranked.append((density, rel, token_cost))
    ranked.sort(key=lambda item: (-item[0], item[1]))
    return ranked


def _force_prune_by_measured_budget(repo_path: Path, selected_files: list[str], token_budget: int, measured_prompt_tokens: int) -> list[str]:
    """Single-shot hard prune using measured prompt overflow when soft packing is insufficient."""
    if measured_prompt_tokens <= 0 or token_budget <= 0 or len(selected_files) <= 1:
        return selected_files

    selected_tuples = collect_selected_files(repo_path, selected_files)
    if not selected_tuples:
        return selected_files[:1]

    ranked = _rank_selected_files_by_value_density(selected_tuples)
    ranked_rels = [rel for _, rel, _ in ranked]
    keep_ratio = min(1.0, max(0.05, float(token_budget) / float(measured_prompt_tokens)))
    forced_count = max(1, min(len(ranked_rels), int(len(ranked_rels) * keep_ratio * 0.9)))
    picked = ranked_rels[:forced_count]
    picked = _ensure_required_category(picked, ranked_rels, _path_matches_manifest_build)
    picked = _ensure_required_category(picked, ranked_rels, _path_matches_verification_hint)
    return picked


def _make_classify_config() -> ClassifyConfig:
    """Snapshot the run-constant Stage-1 tuning knobs from argv into one object,
    so each loop takes a single `config` rather than five-to-seven scalar params."""
    return ClassifyConfig(
        classification_timeout=args.classification_timeout,
        selection_timeout=args.selection_timeout,
        react_max_steps=args.react_max_steps,
        react_max_total_files=args.react_max_total_files,
        react_final_cap=args.react_final_cap,
        synthesis_react_max_steps=args.synthesis_react_max_steps,
        synthesis_review_rounds=args.synthesis_review_rounds,
        validation_react_max_steps=args.validation_react_max_steps,
        synthesis_subagents_enabled=args.synthesis_subagents_enabled,
        snippet_tools_enabled=args.snippet_tools,
        run_validation=args.validation_enabled,
    )


def _make_classify_runtime() -> ClassifyRuntime:
    """Bundle the per-run agent helpers (model factory + extractors/normalizers)
    that every Stage-1 loop needs, so they are passed as one object instead of
    threaded individually through each signature."""
    return ClassifyRuntime(
        model_name=args.model,
        new_prebuilt_chat_model=_new_prebuilt_chat_model,
        extract_agent_payload=_extract_agent_payload,
        extract_agent_trace=_extract_agent_trace,
        normalize_text_list=_normalize_text_list,
        normalize_validation_checks=_normalize_validation_checks,
    )


async def _run_architecture_state_graph(
    *,
    repo_url: str,
    repo_name: str,
    repo_path: "Path",
    summary: str,
    selected_files: list[str],
    exploration_artifact: dict[str, Any],
    file_context_by_path: dict[str, str],
    llm_metrics: dict[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]], str, dict[str, Any], list[dict[str, Any]], str]:
    return await _run_architecture_state_graph_impl(
        repo=RepoRef(url=repo_url, name=repo_name, path=str(repo_path)),
        summary=summary,
        selected_files=selected_files,
        exploration_artifact=exploration_artifact,
        file_context_by_path=file_context_by_path,
        config=_make_classify_config(),
        runtime=_make_classify_runtime(),
    )


def _normalize_text_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    items: list[str] = []
    for raw in value:
        if isinstance(raw, (str, int, float)):
            text = str(raw).strip()
            if text:
                items.append(text)
    return items


def _normalize_validation_checks(value: Any) -> dict[str, dict[str, str]]:
    if not isinstance(value, dict):
        return {}
    normalized: dict[str, dict[str, str]] = {}
    for key, payload in value.items():
        if not isinstance(key, str):
            continue
        status = "warn"
        detail = "No detail provided."
        if isinstance(payload, dict):
            raw_status = str(payload.get("status", "warn")).strip().lower()
            if raw_status in {"pass", "warn", "fail"}:
                status = raw_status
            raw_detail = payload.get("detail")
            if isinstance(raw_detail, str) and raw_detail.strip():
                detail = raw_detail.strip()
        elif isinstance(payload, str) and payload.strip():
            detail = payload.strip()
        normalized[key.strip()] = {"status": status, "detail": detail}
    return normalized


def _build_file_context_by_path(repo_path: Path) -> dict[str, str]:
    file_context: dict[str, str] = {}
    for rel, content in collect_retrieval_candidates(repo_path):
        normalized = rel.lstrip("./")
        if normalized and normalized not in file_context:
            file_context[normalized] = content
    return file_context


def _build_exploration_artifact(
    *,
    repo_url: str,
    retrieval_strategy: str,
    selected_files: list[str],
    react_trace: list[dict[str, Any]],
    react_stop_reason: str,
) -> dict[str, Any]:
    """Derive the L1 exploration artifact — evidence-gap flags, high-signal files,
    and the files the ReAct agent actually read — from the selected files and trace.
    Pure: the caller decides whether/where to persist the returned dict."""
    selected_lower = [str(path).lower() for path in selected_files]
    manifest_markers = (
        "package.json",
        "pyproject.toml",
        "requirements.txt",
        "go.mod",
        "cargo.toml",
        "pom.xml",
        "build.gradle",
        "build.gradle.kts",
    )
    has_manifest = any(any(marker in path for marker in manifest_markers) for path in selected_lower)
    has_ci = any(path.startswith(".github/workflows/") for path in selected_lower)
    has_docker = any("dockerfile" in path or "docker-compose" in path for path in selected_lower)
    has_tests = any(
        "/test" in path
        or path.startswith("test/")
        or path.startswith("tests/")
        or path.endswith(".spec.ts")
        or path.endswith("_test.go")
        for path in selected_lower
    )

    high_signal_files = [
        path
        for path in selected_files
        if any(
            marker in path.lower()
            for marker in (
                "package.json",
                "pyproject.toml",
                "requirements",
                "go.mod",
                "cargo.toml",
                "pom.xml",
                "build.gradle",
                "dockerfile",
                "docker-compose",
                ".github/workflows",
            )
        )
    ]

    # SP1: extract files the L1 agent actually read/listed from the trace.
    files_read_by_l1: list[dict[str, str]] = []
    for evt in react_trace:
        for tc in (evt.get("tool_calls") or []):
            if isinstance(tc, dict) and tc.get("name") in {"read_file", "list_tree"}:
                path = (tc.get("args") or {}).get("path")
                if path:
                    files_read_by_l1.append({"tool": tc["name"], "path": str(path)})

    return {
        "repo": repo_url,
        "stage": "exploration",
        "retrieval_strategy": retrieval_strategy,
        "react": {
            "steps": len(react_trace),
            "stop_reason": react_stop_reason,
        },
        "high_signal_files": high_signal_files,
        "files_read_by_l1": files_read_by_l1,
        "evidence_gaps": {
            "manifest_evidence_missing": not has_manifest,
            "ci_workflow_evidence_missing": not has_ci,
            "docker_evidence_missing": not has_docker,
            "test_evidence_missing": not has_tests,
        },
        "focus_questions": [
            "Which build tool and entrypoints are required for reproducible builds?",
            "Which system dependencies are likely required inside container builds?",
            "What is the minimal verification command that proves runtime correctness?",
        ],
    }


async def _select_relevant_files(
    *,
    repo_url: str,
    repo_name: str,
    repo_path: "Path",
    structure_summary: str,
    default_selected_files: list[str],
    prior_failure_terms: list[str],
    prior_failure_summary: str,
) -> tuple[list[str], str, int, list[dict[str, Any]], str, str]:
    """Run the configured retrieval strategy (Stage-1 file selection) and return
    (selected_files, baseline_summary, step1_tokens, react_trace, react_stop_reason,
    structure_summary). structure_summary is returned because the iterative_react
    path augments it with prior-failure signals before selecting."""
    selected_files = default_selected_files.copy()
    baseline_summary = ""
    step1_tokens = 0
    react_trace: list[dict[str, Any]] = []
    react_stop_reason = "not_applicable"
    if args.retrieval_strategy == "bm25":
        selected_files = select_files_by_bm25(repo_path, build_bm25_query_terms(repo_name, prior_failure_terms))
        if not selected_files:
            log_warn(f"BM25 selection returned no files for {repo_url}; using defaults.")
            selected_files = default_selected_files.copy()
        log_info(f"Selected {len(selected_files)} files for {repo_name} via BM25 retrieval.")
    elif args.retrieval_strategy == "neural_embedding":
        selected_files = await select_files_by_neural_embedding(
            repo_name,
            repo_path,
            extra_terms=prior_failure_terms,
        )
        if not selected_files:
            log_warn(f"Neural embedding selection returned no files for {repo_url}; using defaults.")
            selected_files = default_selected_files.copy()
        log_info(f"Selected {len(selected_files)} files for {repo_name} via neural embedding retrieval.")
    elif args.retrieval_strategy == "one_shot_fingerprint":
        baseline_summary = fingerprint(
            format="md",
            repo_path=str(repo_path),
            structure_only=False,
            selected_files=None,
            include_tree=True,
            context="step2-one-shot-fingerprint",
        )
        baseline_summary = _truncate_to_token_budget(
            baseline_summary, args.model, int(args.max_input_tokens * 0.8),
            label="one-shot fingerprint",
        )
        selected_files = []
        log_info(f"Using static repo fingerprint for {repo_name} via one-shot retrieval.")
    elif args.retrieval_strategy == "one_shot_fingerprint_budgeted":
        _budgeted_files = select_files_by_bm25_budgeted(
            repo_path, build_bm25_query_terms(repo_name, prior_failure_terms)
        )
        baseline_summary = fingerprint(
            format="md",
            repo_path=str(repo_path),
            structure_only=False,
            selected_files=_budgeted_files if _budgeted_files else None,
            include_tree=True,
            context="step2-one-shot-fingerprint-budgeted",
        )
        baseline_summary = _truncate_to_token_budget(
            baseline_summary, args.model, int(args.max_input_tokens * 0.8),
            label="one-shot fingerprint",
        )
        selected_files = []
        log_info(
            f"Using budgeted repo fingerprint for {repo_name} via one-shot-budgeted retrieval "
            f"({len(_budgeted_files)} files selected)."
        )
    elif args.retrieval_strategy == "iterative_react":
        if prior_failure_summary:
            structure_summary = (
                structure_summary
                + "\n\nPRIOR_FAILURE_SIGNALS (from shared state):\n"
                + prior_failure_summary
            )
        selected_files, step1_tokens, react_trace, react_stop_reason = await select_files_by_iterative_react(
            repo=RepoRef(url=repo_url, name=repo_name, path=repo_path),
            structure_summary=structure_summary,
            default_selected_files=default_selected_files,
            config=_make_classify_config(),
            runtime=_make_classify_runtime(),
            estimate_tokens=estimate_tokens,
        )
        log_info(
            f"Selected {len(selected_files)} files for {repo_name} via iterative ReAct retrieval "
            f"(stop_reason={react_stop_reason})."
        )
    else:
        log_warn(
            f"Unknown retrieval strategy '{args.retrieval_strategy}' for {repo_url}; using default file selection."
        )
        selected_files = default_selected_files.copy()

    return selected_files, baseline_summary, step1_tokens, react_trace, react_stop_reason, structure_summary


def _build_step2_summary(
    *,
    repo_url: str,
    repo_name: str,
    repo_path: "Path",
    selected_files: list[str],
    selected_files_before_budget: list[str],
    baseline_summary: str,
    budget_behavior: dict[str, Any],
) -> tuple[str, list[str]]:
    """Build the Step-2 classification summary — the one-shot baseline fingerprint,
    or a selected-files fingerprint with step2 token-budget enforcement — and return
    (summary, selected_files). budget_behavior is updated in place."""
    # Step 2: only selected file contents are provided to the classification prompt.
    if args.retrieval_strategy in {"one_shot_fingerprint", "one_shot_fingerprint_budgeted"}:
        summary = baseline_summary
    else:
        summary = fingerprint(
            format="md",
            repo_path=str(repo_path),
            selected_files=selected_files,
            include_tree=False,
            context="step2-selected",
        )
        if args.step2_token_budget > 0:
            budget_prompt = PROMPT_TEMPLATE.replace("{{REPO_URL}}", repo_url).replace("{{SUMMARY_CONTENT}}", summary)
            budget_tokens = estimate_tokens(budget_prompt, args.model)
            budget_behavior["measured_prompt_tokens_before_enforcement"] = budget_tokens
            if budget_tokens > args.step2_token_budget:
                hard_pruned = _force_prune_by_measured_budget(
                    repo_path,
                    selected_files,
                    args.step2_token_budget,
                    budget_tokens,
                )
                if hard_pruned != selected_files:
                    selected_files = hard_pruned
                    summary = fingerprint(
                        format="md",
                        repo_path=str(repo_path),
                        selected_files=selected_files,
                        include_tree=False,
                        context="step2-selected",
                    )
                    budget_prompt = PROMPT_TEMPLATE.replace("{{REPO_URL}}", repo_url).replace("{{SUMMARY_CONTENT}}", summary)
                    budget_tokens = estimate_tokens(budget_prompt, args.model)
                    budget_behavior["hard_enforcement_applied"] = True

                budget_behavior["measured_prompt_tokens_after_enforcement"] = budget_tokens
                if budget_tokens > args.step2_token_budget:
                    log_warn(
                        f"Step 2 prompt for {repo_name} still exceeded budget after hard enforcement "
                        f"(prompt_tokens={budget_tokens}, budget={args.step2_token_budget})."
                    )
                else:
                    log_info(
                        f"Hard budget enforcement succeeded for {repo_name}: "
                        f"prompt_tokens={budget_tokens}, selected_files={len(selected_files)}"
                    )
            else:
                budget_behavior["measured_prompt_tokens_after_enforcement"] = budget_tokens
                log_info(
                    f"Applied Step 2 token budget packing for {repo_name}: "
                    f"prompt_tokens={budget_tokens}, selected_files={len(selected_files)}"
                )

            budget_behavior["post_budget_selected_files_count"] = len(selected_files)
            budget_behavior["pruned_files_count"] = max(0, len(selected_files_before_budget) - len(selected_files))
            budget_behavior["applied"] = selected_files != selected_files_before_budget
            final_estimated = _estimate_step2_prompt_tokens_from_selected_files(
                repo_path,
                repo_url,
                selected_files,
            )
            budget_behavior["estimated_prompt_tokens_after_budget"] = final_estimated
            before_estimated = budget_behavior.get("estimated_prompt_tokens_before_budget")
            if isinstance(before_estimated, int):
                budget_behavior["estimated_saved_tokens"] = max(0, before_estimated - final_estimated)
    return summary, selected_files


async def analyze_repository(repo_url: str, repos_dir: Path, summary_dir: Path, output_dir: Path, results_dir: Path, progress_state: dict, force: bool = False, learn: bool = False) -> None:
    async with sem:
        repo_name = repo_url.split("/")[-1].replace(".git", "")
        llm_metrics_path = results_dir / f"{repo_name}.llm-metrics.yaml"
        llm_metrics = init_llm_metrics(repo_url, args.model, args.endpoint, args.timeout, args.llm_max_retries)
        llm_metrics["prompt_profile"] = prompt_profile_metadata(PROMPT_PROFILE, EFFECTIVE_TEMPERATURE)
        try:
            log_info(f"Processing {repo_url}...")
            output_path = results_dir / f"{repo_name}.yaml"
            metrics_path = results_dir / f"{repo_name}.token-metrics.yaml"
            shared_repository_state = load_shared_repository_state(repo_name, summary_dir)
            prior_failure_terms, prior_failure_summary = build_prior_failure_retrieval_hints(shared_repository_state)

            repo_path = repos_dir / repo_name
            if not await ensure_repo_checkout(repo_url, repo_path, "skipping classify"):
                return

            if output_path.exists() and not force:
                log_info(f"Skipping {repo_url}: existing result found at {output_path}")
                return
            
            if output_path.exists() and force:
                log_info(f"Overwriting existing result for {repo_url}")

            repo_path = output_dir / repo_name
            if not repo_path.exists():
                log_error(f"Cannot find directory for {repo_url} at {repo_path}")
            
            # Strip the repo's declarative files_to_delete (docs + CI) so the classifier
            # never reads install instructions — the same per-repo list the build strip
            # uses (no global heuristic). The repair stage re-strips after its hard reset.
            if args.dataset_dir:
                gt_doc = load_gt_for_repo(Path(args.dataset_dir), repo_url)
                delete_files_build_context(repo_path, repo_name, get_files_to_delete(gt_doc))
            # TODO: stripping docs is mandatory we need to abort the FULL pipeline if we cant load gt_doc
            # TODO: this code needs to be in agent_pipeline.py because one_shot_direct currently skips 
            # TODO: things like git checkout/clone and docs delete need to be done before any stage or agent config is run

            # Conservative default file selection used when a retrieval strategy returns
            # nothing (or is unknown). High-signal manifests/docs/CI across the dataset's languages.
            
            # TODO: this is way too small, prompt LLM (not dynamically but for code change) for more complete list for all 6 languages
            default_selected_files = list([
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
            ])

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

            # Cap the structure summary fed to LLM prompts (the full copy stays on disk).
            # The ReAct loop also accumulates tool output, so reserve ~half the input budget.
            structure_summary = _truncate_to_token_budget(
                structure_summary, args.model, int(args.max_input_tokens * 0.5),
                label="structure summary",
            )

            (
                selected_files,
                baseline_summary,
                step1_tokens,
                react_trace,
                react_stop_reason,
                structure_summary,
            ) = await _select_relevant_files(
                repo_url=repo_url,
                repo_name=repo_name,
                repo_path=repo_path,
                structure_summary=structure_summary,
                default_selected_files=default_selected_files,
                prior_failure_terms=prior_failure_terms,
                prior_failure_summary=prior_failure_summary,
            )

            selected_files_before_budget = selected_files.copy()
            budget_behavior = {
                "enabled": bool(args.step2_token_budget > 0 and args.retrieval_strategy not in {"one_shot_fingerprint", "one_shot_fingerprint_budgeted"}),
                "token_budget": int(args.step2_token_budget),
                "initial_selected_files_count": len(selected_files_before_budget),
                "post_budget_selected_files_count": len(selected_files_before_budget),
                "pruned_files_count": 0,
                "estimated_prompt_tokens_before_budget": None,
                "estimated_prompt_tokens_after_budget": None,
                "estimated_saved_tokens": 0,
                "applied": False,
                "measured_prompt_tokens_before_enforcement": None,
                "measured_prompt_tokens_after_enforcement": None,
                "hard_enforcement_applied": False,
            }

            if args.retrieval_strategy not in {"one_shot_fingerprint", "one_shot_fingerprint_budgeted"} and args.step2_token_budget > 0:
                estimated_before = _estimate_step2_prompt_tokens_from_selected_files(
                    repo_path,
                    repo_url,
                    selected_files_before_budget,
                )
                selected_files = _select_files_by_value_per_token(
                    repo_path=repo_path,
                    repo_url=repo_url,
                    selected_files=selected_files,
                    token_budget=args.step2_token_budget,
                )
                estimated_after = _estimate_step2_prompt_tokens_from_selected_files(
                    repo_path,
                    repo_url,
                    selected_files,
                )
                budget_behavior.update(
                    {
                        "post_budget_selected_files_count": len(selected_files),
                        "pruned_files_count": max(0, len(selected_files_before_budget) - len(selected_files)),
                        "estimated_prompt_tokens_before_budget": estimated_before,
                        "estimated_prompt_tokens_after_budget": estimated_after,
                        "estimated_saved_tokens": max(0, estimated_before - estimated_after),
                        "applied": selected_files != selected_files_before_budget,
                    }
                )

            if args.retrieval_strategy == "iterative_react":
                react_trace_path = summary_dir / f"{repo_name}.react-trace.yaml"
                with open(react_trace_path, "w") as f:
                    yaml.dump(
                        {
                            "retrieval_strategy": args.retrieval_strategy,
                            "stop_reason": react_stop_reason,
                            "steps": react_trace,
                        },
                        f,
                        sort_keys=False,
                        allow_unicode=True,
                    )
                log_info(f"ReAct retrieval trace saved at {react_trace_path}")

            if args.retrieval_strategy not in {"bm25", "neural_embedding", "one_shot_fingerprint", "one_shot_fingerprint_budgeted", "iterative_react"}:
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

            exploration_artifact = _build_exploration_artifact(
                repo_url=repo_url,
                retrieval_strategy=args.retrieval_strategy,
                selected_files=selected_files,
                react_trace=react_trace,
                react_stop_reason=react_stop_reason,
            )
            exploration_path = None
            if args.exploration_enabled:
                exploration_path = summary_dir / f"{repo_name}.exploration.yaml"
                with open(exploration_path, "w", encoding="utf-8") as exploration_file:
                    yaml.dump(exploration_artifact, exploration_file, sort_keys=False, allow_unicode=True)
                log_info(f"Exploration artifact saved at {exploration_path}")

            summary, selected_files = _build_step2_summary(
                repo_url=repo_url,
                repo_name=repo_name,
                repo_path=repo_path,
                selected_files=selected_files,
                selected_files_before_budget=selected_files_before_budget,
                baseline_summary=baseline_summary,
                budget_behavior=budget_behavior,
            )
            summary_path = summary_dir / f"{repo_name}.md"
            with open(summary_path, "w") as f:
                f.write(summary)
            log_info(f"Generated structure summary at {structure_summary_path}")
            log_info(f"Selected files list saved at {selected_files_path}")
            log_info(f"Generated selected-files summary at {summary_path}")

            file_context_by_path = _build_file_context_by_path(repo_path)
            _l1_escalation_count = 0
            _graph_selected_files = selected_files
            _graph_summary = summary
            while True:
                (
                    synthesis_artifact,
                    synthesis_loop_trace,
                    synthesis_stop_reason,
                    validation_artifact,
                    validation_loop_trace,
                    validation_stop_reason,
                ) = await _run_architecture_state_graph(
                    repo_url=repo_url,
                    repo_name=repo_name,
                    repo_path=repo_path,
                    summary=_graph_summary,
                    selected_files=_graph_selected_files,
                    exploration_artifact=exploration_artifact,
                    file_context_by_path=file_context_by_path,
                    llm_metrics=llm_metrics,
                )
                # L2→L1 escalation: re-run L1 with doubled steps when L2 found no stack signals.
                # Only triggers for iterative_react strategy (others have no L1 to re-run).
                if (
                    synthesis_artifact.get("stack_unknown")
                    and _l1_escalation_count < 1
                    and args.retrieval_strategy == "iterative_react"
                ):
                    _l1_escalation_count += 1
                    log_info(
                        f"Stack unknown for {repo_name} — escalating to L1 re-run "
                        f"(attempt {_l1_escalation_count}, react_max_steps={args.react_max_steps * 2})."
                    )
                    _rerun_files, _, _, _ = await select_files_by_iterative_react(
                        repo=RepoRef(url=repo_url, name=repo_name, path=repo_path),
                        structure_summary=structure_summary,
                        default_selected_files=default_selected_files,
                        config=dataclasses.replace(
                            _make_classify_config(),
                            react_max_steps=args.react_max_steps * 2,
                        ),
                        runtime=_make_classify_runtime(),
                        estimate_tokens=estimate_tokens,
                    )
                    if _rerun_files:
                        _graph_selected_files = _rerun_files
                        _graph_summary = fingerprint(
                            format="md",
                            repo_path=str(repo_path),
                            selected_files=_graph_selected_files,
                            include_tree=False,
                            context="l1-escalation-step2",
                        )
                    continue
                break
            selected_files = _graph_selected_files
            summary = _graph_summary
            synthesis_path = None
            if args.synthesis_enabled:
                synthesis_path = summary_dir / f"{repo_name}.synthesis.yaml"
                with open(synthesis_path, "w", encoding="utf-8") as synthesis_file:
                    yaml.dump(synthesis_artifact, synthesis_file, sort_keys=False, allow_unicode=True)
                log_info(f"Synthesis artifact saved at {synthesis_path}")

            validation_checks = validation_artifact["checks"]
            validation_warnings = validation_artifact["warnings"]
            validation_path = None
            if args.validation_enabled:
                validation_path = summary_dir / f"{repo_name}.validation.yaml"
                with open(validation_path, "w", encoding="utf-8") as validation_file:
                    yaml.dump(validation_artifact, validation_file, sort_keys=False, allow_unicode=True)
                log_info(f"Validation artifact saved at {validation_path}")

            # Token accounting for baseline vs two-step prompts.
            # Baseline: deterministic collection of all manifest files (no LLM filtering).
            if not baseline_summary:
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
                "budget_behavior": budget_behavior,
                "retrieval_trace": {
                    "react_steps": len(react_trace),
                    "react_stop_reason": react_stop_reason,
                },
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

            if args.scratchpads_enabled and args.scratchpad_dir:
                scratchpad_dir = Path(args.scratchpad_dir)
                scratchpad_dir.mkdir(parents=True, exist_ok=True)
                scratchpad_path = scratchpad_dir / f"{repo_name}.architecture-scratchpad.yaml"
                scratchpad_payload = build_architecture_scratchpad_payload(
                    repo_url=repo_url,
                    retrieval_strategy=args.retrieval_strategy,
                    selected_files=selected_files,
                    exploration=LoopOutcome(
                        path=exploration_path, artifact=exploration_artifact,
                        loop_trace=react_trace, stop_reason=react_stop_reason,
                    ),
                    synthesis=LoopOutcome(
                        path=synthesis_path, artifact=synthesis_artifact,
                        loop_trace=synthesis_loop_trace, stop_reason=synthesis_stop_reason,
                    ),
                    validation=LoopOutcome(
                        path=validation_path, artifact=validation_artifact,
                        loop_trace=validation_loop_trace, stop_reason=validation_stop_reason,
                    ),
                    tokens=TokenCounts(
                        step1_selection=step1_tokens,
                        step2_classification=step2_tokens,
                        two_step_total=two_step_total_tokens,
                    ),
                    paths=SummaryPaths(
                        structure_summary=structure_summary_path,
                        selected_summary=Path(summary_path),
                        selected_files=selected_files_path,
                    ),
                    subagents_enabled=args.synthesis_subagents_enabled,
                    budget_behavior=budget_behavior,
                )
                with open(scratchpad_path, "w", encoding="utf-8") as scratchpad_file:
                    yaml.dump(scratchpad_payload, scratchpad_file, sort_keys=False, allow_unicode=True)
                log_info(f"Architecture scratchpad saved at {scratchpad_path}")

            upsert_shared_repository_state(
                repo_name,
                summary_dir,
                repo_url=repo_url,
                stage_name="classify",
                stage_update={
                    "status": "loops_completed",
                    "retrieval_strategy": args.retrieval_strategy,
                    "selected_files_count": len(selected_files),
                    "selected_files_path": str(selected_files_path),
                    "summary_path": str(summary_path),
                    "exploration_artifact_path": str(exploration_path) if exploration_path else "",
                    "synthesis_artifact_path": str(synthesis_path) if synthesis_path else "",
                    "validation_artifact_path": str(validation_path) if validation_path else "",
                    "synthesis_transition": synthesis_artifact.get("transition_policy", {}),
                    "validation_outcome": validation_artifact.get("outcome_state", "unknown"),
                },
            )

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
                    upsert_shared_repository_state(
                        repo_name,
                        summary_dir,
                        repo_url=repo_url,
                        stage_name="classify",
                        stage_update={
                            "status": "completed",
                            "classification_output_path": str(output_path),
                        },
                    )
                except yaml.YAMLError as e:
                    log_warn(f"Failed to parse result for {repo_name}: {e}")
                    with open(output_path, "w") as f:
                        yaml.dump({"error": "parse_failed", "raw": raw}, f, sort_keys=False, allow_unicode=True)
                    upsert_shared_repository_state(
                        repo_name,
                        summary_dir,
                        repo_url=repo_url,
                        stage_name="classify",
                        stage_update={
                            "status": "parse_failed",
                            "classification_output_path": str(output_path),
                        },
                    )

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
            log_error(f"Unexpected error during classification while processing {repo_url}: {e}")
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
    if should_use_progress(len(repos), args.trace):
        progress_bar = tqdm(total=len(repos), desc="Analyzing repos", unit="repo", dynamic_ncols=True)

    progress_state = {
        "lock": asyncio.Lock(),
        "bar": progress_bar,
    }
    set_tqdm_bar(progress_state["bar"])
    log_info(f"Starting analysis for {len(repos)} repositories")
    tasks = [analyze_repository(repo, repos_dir, summaries_dir, repos_dir, results_dir, progress_state, force=args.force, learn=args.learn) for repo in repos]
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
        analysis_script = Path(__file__).resolve().parents[2] / "metrics" / "parse_results.py"
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