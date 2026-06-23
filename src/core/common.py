"""Shared utilities for the RepoBuilderAgent pipeline scripts."""

from __future__ import annotations

import asyncio
import base64
import functools
import json
import os
import re
import ssl
import subprocess
import sys
import time
import types
from datetime import datetime, timezone
from pathlib import Path

import httpx
import yaml
from openai import APIConnectionError, APIError, APITimeoutError

try:
    from RepoBuilderAgent.src.core.log_utils import dump_prompt, log_info, log_warn
    from RepoBuilderAgent.src.retrieval.repo_fingerprint import fingerprint
except ImportError:
    # Fallback for direct script execution from RepoBuilderAgent/src
    from core.log_utils import dump_prompt, log_info, log_warn
    from retrieval.repo_fingerprint import fingerprint


ARCHITECTURE_SCRATCHPAD_SCHEMA_VERSION = "1.0"
SHARED_REPOSITORY_STATE_SCHEMA_VERSION = "1.0"


def init_llm_metrics(repo_url: str, model: str, endpoint: str, timeout_seconds: int, max_retries: int) -> dict:
    return {
        "repo": repo_url,
        "model": model,
        "endpoint": endpoint,
        "timeout_seconds": timeout_seconds,
        "max_retries": max_retries,
        "phases": {},
    }


def _phase_bucket(metrics: dict, phase: str) -> dict:
    phases = metrics.setdefault("phases", {})
    return phases.setdefault(
        phase,
        {
            "calls": 0,
            "success": 0,
            "timeout": 0,
            "connection_error": 0,
            "api_error": 0,
            "http_error": 0,
            "ssl_error": 0,
            "other_error": 0,
            "retries": 0,
            "latencies_seconds": [],
            "attempts": [],
        },
    )


def finalize_llm_metrics(metrics: dict) -> dict:
    for phase_data in metrics.get("phases", {}).values():
        latencies = phase_data.get("latencies_seconds", [])
        if latencies:
            phase_data["latency_summary_seconds"] = {
                "min": round(min(latencies), 3),
                "avg": round(sum(latencies) / len(latencies), 3),
                "max": round(max(latencies), 3),
            }
    return metrics


def is_retryable_api_error(error: APIError) -> bool:
    status_code = getattr(error, "status_code", None)
    if status_code is None:
        return False
    return status_code in {408, 409, 429} or status_code >= 500


_RESPONSES_API_MODELS = {"gpt-5-codex"}


def _normalize_responses_output(raw) -> types.SimpleNamespace:
    """Wrap a responses API result to the same interface as chat.completions."""
    text = getattr(raw, "output_text", "") or ""
    return types.SimpleNamespace(
        choices=[types.SimpleNamespace(message=types.SimpleNamespace(content=text))]
    )


def build_async_http_client(timeout_seconds: float) -> httpx.AsyncClient:
    """Shared httpx client for AsyncOpenAI / ChatOpenAI across all stages.

    Uses ssl.create_default_context() to pull in the OS trust store — httpx defaults
    to certifi's CA bundle, which omits corporate / internal CAs. The explicit
    httpx.Timeout bounds connect/read/write/pool per operation; the hard wall-clock
    ceiling is enforced separately in chat_completion_with_retries via asyncio.wait_for,
    because httpx read timeouts reset on each received chunk and never bound a
    trickling response.
    """
    return httpx.AsyncClient(
        verify=ssl.create_default_context(),
        timeout=httpx.Timeout(timeout_seconds),
    )


async def chat_completion_with_retries(
    *,
    client,
    model: str,
    temperature: float,
    messages: list[dict[str, str]],
    repo_url: str,
    phase: str,
    metrics: dict,
    timeout_seconds: int,
    max_retries: int,
    retry_backoff_seconds: float,
    max_tokens: int | None = None,
):
    bucket = _phase_bucket(metrics, phase)
    last_error: Exception | None = None
    dump_prompt(repo_url, phase, messages)
    use_responses_api = model in _RESPONSES_API_MODELS

    for attempt in range(1, max_retries + 2):
        bucket["calls"] += 1
        started = time.perf_counter()
        try:
            # asyncio.wait_for is the hard wall-clock ceiling. The httpx/openai
            # `timeout` is a per-socket-read deadline that resets on every chunk
            # received, so a server that trickles bytes (e.g. a slow generation for
            # a very large prompt) never trips it and the call hangs indefinitely.
            c = client.with_options(timeout=timeout_seconds)
            if use_responses_api:
                responses_kwargs = {"model": model, "input": messages}
                if max_tokens is not None:
                    responses_kwargs["max_output_tokens"] = max_tokens
                raw = await asyncio.wait_for(
                    c.responses.create(**responses_kwargs),
                    timeout=timeout_seconds,
                )
                response = _normalize_responses_output(raw)
            else:
                completion_kwargs = {
                    "model": model,
                    "temperature": temperature,
                    "messages": messages,
                }
                if max_tokens is not None:
                    completion_kwargs["max_tokens"] = max_tokens
                response = await asyncio.wait_for(
                    c.chat.completions.create(**completion_kwargs),
                    timeout=timeout_seconds,
                )
            latency = round(time.perf_counter() - started, 3)
            bucket["success"] += 1
            bucket["latencies_seconds"].append(latency)
            bucket["attempts"].append(
                {
                    "attempt": attempt,
                    "status": "success",
                    "latency_seconds": latency,
                }
            )
            return response
        except (APITimeoutError, asyncio.TimeoutError) as error:
            latency = round(time.perf_counter() - started, 3)
            bucket["timeout"] += 1
            bucket["attempts"].append(
                {
                    "attempt": attempt,
                    "status": "timeout",
                    "latency_seconds": latency,
                    "error": str(error),
                }
            )
            last_error = error
            if attempt > max_retries:
                break
            delay_seconds = round(retry_backoff_seconds * (2 ** (attempt - 1)), 3)
            bucket["retries"] += 1
            log_warn(
                f"OpenAI timeout for {repo_url} [{phase}] attempt {attempt}/{max_retries + 1}; retrying in {delay_seconds}s"
            )
            await asyncio.sleep(delay_seconds)
        except APIConnectionError as error:
            latency = round(time.perf_counter() - started, 3)
            bucket["connection_error"] += 1
            bucket["attempts"].append(
                {
                    "attempt": attempt,
                    "status": "connection_error",
                    "latency_seconds": latency,
                    "error": str(error),
                }
            )
            last_error = error
            if attempt > max_retries:
                break
            delay_seconds = round(retry_backoff_seconds * (2 ** (attempt - 1)), 3)
            bucket["retries"] += 1
            log_warn(
                f"OpenAI connection error for {repo_url} [{phase}] attempt {attempt}/{max_retries + 1}; retrying in {delay_seconds}s"
            )
            await asyncio.sleep(delay_seconds)
        except APIError as error:
            latency = round(time.perf_counter() - started, 3)
            bucket["api_error"] += 1
            bucket["attempts"].append(
                {
                    "attempt": attempt,
                    "status": "api_error",
                    "latency_seconds": latency,
                    "error": str(error),
                    "status_code": getattr(error, "status_code", None),
                }
            )
            last_error = error
            if attempt > max_retries or not is_retryable_api_error(error):
                break
            delay_seconds = round(retry_backoff_seconds * (2 ** (attempt - 1)), 3)
            bucket["retries"] += 1
            log_warn(
                f"Retryable API error for {repo_url} [{phase}] attempt {attempt}/{max_retries + 1}; retrying in {delay_seconds}s"
            )
            await asyncio.sleep(delay_seconds)
        except Exception as error:
            latency = round(time.perf_counter() - started, 3)
            bucket["other_error"] += 1
            bucket["attempts"].append(
                {
                    "attempt": attempt,
                    "status": "other_error",
                    "latency_seconds": latency,
                    "error": str(error),
                    "error_type": type(error).__name__,
                }
            )
            last_error = error
            break

    if last_error is not None:
        raise last_error
    raise RuntimeError(f"Unexpected completion retry flow for {repo_url} [{phase}]")


# Sentinel comments that bracket the auto-injected corporate-CA bootstrap block.
# They make the block trivially strippable before it is shown to a repair LLM (so
# the model never has to regurgitate the multi-kilobyte base64 cert payload) and
# serve as the idempotency guard so a re-injection on a repair pass is a no-op.
CA_BLOCK_BEGIN_MARKER = "# >>> manualrepos-ca-bootstrap (auto-injected at build time) >>>"
CA_BLOCK_END_MARKER = "# <<< manualrepos-ca-bootstrap <<<"


def inject_ca_cert_into_dockerfile(dockerfile_content: str, ca_cert_b64: str | None = None) -> str:
    """
    Inject CA certificate setup into the Dockerfile if MANUALREPOS_CA_CERT_B64 is present.
    This ensures curl/git/wget/npm/pip/maven/rust/cargo can reach package registries behind TLS interception.

    Sets up:
    - System CA trust store (apt, curl, git, wget)
    - Environment variables for Python (pip), Node.js (npm/pnpm), Java (maven)
    - Java keystore import into the DEFAULT OpenJDK keystore (preserves public root CAs)
    - Rust/cargo CA configuration
    """
    def ensure_root_for_sensitive_runs(content: str) -> str:
        """Ensure privileged RUN commands execute as root, then restore prior user."""
        lines_local = content.split("\n")
        rewritten: list[str] = []
        current_user = "root"

        for raw_line in lines_local:
            stripped = raw_line.strip()
            upper = stripped.upper()

            if upper.startswith("USER "):
                parts = stripped.split(None, 1)
                if len(parts) == 2 and parts[1].strip():
                    current_user = parts[1].strip()
                rewritten.append(raw_line)
                continue

            is_run = upper.startswith("RUN ")
            is_chown_run = is_run and "CHOWN" in upper
            uses_root_scoped_path = is_run and any(
                p in upper for p in ("/ROOT/", "/USR/BIN/", "/USR/LOCAL/BIN/", "/BIN/", "/SBIN/", "/USR/SBIN/", "/OPT/")
            )
            needs_root = is_chown_run or uses_root_scoped_path

            if needs_root and current_user not in {"root", "0"}:
                rewritten.append("USER root")
                rewritten.append(raw_line)
                rewritten.append(f"USER {current_user}")
            else:
                rewritten.append(raw_line)

        return "\n".join(rewritten)

    if not ca_cert_b64:
        ca_cert_b64 = os.getenv("MANUALREPOS_CA_CERT_B64")
    if not ca_cert_b64:
        return ensure_root_for_sensitive_runs(dockerfile_content)

    # Avoid repeatedly injecting very large CA blocks on retries/repair passes.
    if CA_BLOCK_BEGIN_MARKER in dockerfile_content or "manualrepos-refresh-ca-certificates" in dockerfile_content:
        return ensure_root_for_sensitive_runs(dockerfile_content)

    decoded_bundle = base64.b64decode(ca_cert_b64).decode("utf-8")
    cert_blocks = []
    for block in decoded_bundle.split("-----END CERTIFICATE-----"):
        block = block.strip()
        if "-----BEGIN CERTIFICATE-----" not in block:
            continue
        cert_blocks.append(f"{block}\n-----END CERTIFICATE-----\n")
    if not cert_blocks:
        cert_blocks = [decoded_bundle]

    ca_cert_path = "/usr/local/share/ca-certificates/custom-ca.crt"
    corporate_ca_cert_path = "/usr/local/share/ca-certificates/corporate-ca.crt"
    split_cert_paths = [f"/usr/local/share/ca-certificates/manualrepos-ca-{index}.crt" for index, _ in enumerate(cert_blocks, start=1)]
    split_cert_commands = " ".join(
        f"printf '%s' '{base64.b64encode(cert.encode('utf-8')).decode('ascii')}' | base64 -d > {path};"
        for cert, path in zip(cert_blocks, split_cert_paths, strict=True)
    )
    split_cert_loop = " ".join(split_cert_paths)
    ca_refresh_script_path = "/usr/local/bin/manualrepos-refresh-ca-certificates"
    ca_refresh_script = f"""#!/bin/sh
set -eu
update-ca-certificates
if command -v keytool >/dev/null 2>&1; then
    for CERT_PATH in {split_cert_loop}; do
        if [ -f \"$CERT_PATH\" ]; then
            CERT_ALIAS=$(basename \"$CERT_PATH\" .crt)
            for JAVA_KEYSTORE in $(find /usr/lib/jvm -name cacerts -path '*/security/*' 2>/dev/null) /etc/ssl/certs/java/cacerts; do
                if [ -f \"$JAVA_KEYSTORE\" ]; then
                    if ! keytool -list -keystore \"$JAVA_KEYSTORE\" -storepass changeit -alias \"$CERT_ALIAS\" >/dev/null 2>&1; then
                        keytool -importcert -trustcacerts -alias \"$CERT_ALIAS\" -file \"$CERT_PATH\" -keystore \"$JAVA_KEYSTORE\" -storepass changeit -noprompt
                    fi
                fi
            done
        fi
    done
    for JAVA_KEYSTORE in $(find /usr/lib/jvm -name cacerts -path '*/security/*' 2>/dev/null) /etc/ssl/certs/java/cacerts; do
        if [ -f \"$JAVA_KEYSTORE\" ]; then
            for CA_ALIAS in custom-ca corporate-ca; do
                if ! keytool -list -keystore \"$JAVA_KEYSTORE\" -storepass changeit -alias \"$CA_ALIAS\" >/dev/null 2>&1; then
                    keytool -importcert -trustcacerts -alias \"$CA_ALIAS\" -file {ca_cert_path} -keystore \"$JAVA_KEYSTORE\" -storepass changeit -noprompt
                fi
            done
        fi
    done
fi
"""
    ca_refresh_script_b64 = base64.b64encode(ca_refresh_script.encode("utf-8")).decode("ascii")
    ca_setup_commands = f"""
RUN apt-get update -qq && apt-get install -y --no-install-recommends ca-certificates curl default-jre-headless 2>/dev/null || true; \
    mkdir -p /usr/local/share/ca-certificates; \
    printf '%s' '{ca_cert_b64}' | base64 -d > {ca_cert_path}; \
    cp {ca_cert_path} {corporate_ca_cert_path}; \
    {split_cert_commands} \
    printf '%s' '{ca_refresh_script_b64}' | base64 -d > {ca_refresh_script_path}; \
    chmod +x {ca_refresh_script_path}; \
    {ca_refresh_script_path}
ENV SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt
ENV REQUESTS_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt
ENV CURL_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt
ENV NODE_EXTRA_CA_CERTS=/etc/ssl/certs/ca-certificates.crt
# pnpm/npm do not reliably honour NODE_EXTRA_CA_CERTS for registry TLS under a
# corporate TLS-intercepting CA, and stall on the handshake (this hung vite's
# build for hours). Mirror the dataset validation harness (helper.py) and skip
# Node TLS verification whenever a corporate CA is present.
ENV NODE_TLS_REJECT_UNAUTHORIZED=0
ENV CARGO_HTTP_CAINFO=/etc/ssl/certs/ca-certificates.crt
ENV SSL_CERT_DIR=/etc/ssl/certs
ENV JAVA_TOOL_OPTIONS="-Dcom.sun.jndi.ldap.connect.pool=false -Djavax.net.ssl.trustStore=/etc/ssl/certs/java/cacerts -Djavax.net.ssl.trustStorePassword=changeit"
ENV GRADLE_OPTS="-Djavax.net.ssl.trustStore=/etc/ssl/certs/java/cacerts -Djavax.net.ssl.trustStorePassword=changeit"
ENV MAVEN_OPTS="-Djavax.net.ssl.trustStore=/etc/ssl/certs/java/cacerts -Djavax.net.ssl.trustStorePassword=changeit"
"""

    lines = dockerfile_content.split("\n")
    injected_lines = []
    inserted = False

    for line in lines:
        # Insert CA setup after first FROM line, before anything else
        if not inserted and line.strip().upper().startswith("FROM"):
            injected_lines.append(line)
            injected_lines.append(CA_BLOCK_BEGIN_MARKER)
            injected_lines.append("USER root")
            injected_lines.append(ca_setup_commands.strip())
            injected_lines.append(CA_BLOCK_END_MARKER)
            inserted = True
        else:
            injected_lines.append(line)

    return ensure_root_for_sensitive_runs("\n".join(injected_lines))


def strip_ca_cert_from_dockerfile(dockerfile_content: str) -> str:
    """Remove the auto-injected corporate-CA bootstrap block (the sentinel-bracketed
    region added by ``inject_ca_cert_into_dockerfile``), replacing it with a one-line
    comment. The block is re-added at build time by injection, so an LLM editing the
    Dockerfile never needs to see or reproduce the multi-kilobyte base64 cert payload.
    Idempotent and a no-op when no injected block is present."""
    if CA_BLOCK_BEGIN_MARKER not in dockerfile_content:
        return dockerfile_content
    out: list[str] = []
    skipping = False
    for line in dockerfile_content.split("\n"):
        if not skipping and line.strip().startswith(CA_BLOCK_BEGIN_MARKER):
            skipping = True
            out.append("# [corporate CA-cert bootstrap auto-injected at build time; omitted here]")
            continue
        if skipping:
            if CA_BLOCK_END_MARKER in line:
                skipping = False
            continue
        out.append(line)
    return "\n".join(out)


# A run of base64 long enough to be a cert/script blob rather than an incidental token.
_LONG_BASE64_RE = re.compile(r"[A-Za-z0-9+/]{120,}={0,2}")


def sanitize_build_log_for_prompt(text: str) -> str:
    """Strip the two kinds of high-volume noise that dominate failing-build logs before
    they are embedded in a repair prompt: (1) multi-kilobyte base64 blobs echoed by
    BuildKit (e.g. the injected CA cert), and (2) carriage-return progress-bar spam
    (apt/dpkg ``\\r``-redrawn lines). Both inflate the prompt and bait the model into
    degenerate repetition. Returns the de-noised log; signal lines are preserved."""
    collapsed_lines = []
    for line in text.split("\n"):
        # A terminal renders only what follows the last carriage return on a line.
        if "\r" in line:
            line = line.split("\r")[-1]
        collapsed_lines.append(line)
    collapsed = "\n".join(collapsed_lines)
    return _LONG_BASE64_RE.sub("<base64 blob omitted>", collapsed)


def repo_name_from_url(repo_url: str) -> str:
    return repo_url.rstrip("/").split("/")[-1].replace(".git", "")


def load_repo_urls(input_file: str, repo_urls: list[str]) -> list[str]:
    """Return deduplicated repo URLs from CLI overrides or input file."""
    if repo_urls:
        repos = [url.strip() for url in repo_urls if url and url.strip()]
        return list(dict.fromkeys(repos))
    with open(input_file, "r", encoding="utf-8") as f:
        return [item["url"] for item in json.load(f)]


def read_yaml_file(path: Path) -> dict | None:
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_summary(repo_name: str, repo_path: Path, summaries_dir: Path) -> str:
    """Return a prompt-friendly summary of the repository.

    Prefers the pre-generated reduced summary written by agent_classify.py;
    falls back to reconstructing from the selected-files list, then a full
    baseline fingerprint.
    """
    reduced_path = summaries_dir / f"{repo_name}.md"
    if reduced_path.exists():
        with open(reduced_path, "r", encoding="utf-8") as f:
            return f.read()

    selected_files_config = read_yaml_file(summaries_dir / f"{repo_name}.selected-files.yaml") or {}
    selected_files = selected_files_config.get("selected_files")
    if isinstance(selected_files, list) and selected_files:
        return fingerprint(
            format="md",
            repo_path=str(repo_path),
            selected_files=selected_files,
            include_tree=False,
            context="summary-selected",
        )

    return fingerprint(
        format="md",
        repo_path=str(repo_path),
        selected_files=None,
        include_tree=True,
        context="summary-baseline",
    )


# ---------------------------------------------------------------------------
# Input-token bounding
#
# The model endpoint enforces a hard input-token cap (e.g. 64000 for gpt-4o).
# The repository summary — dominated by the directory tree for large repos —
# is the one unbounded part of every stage prompt, so a giant repo (airflow:
# ~1700 tree entries) overflows the cap and the call 400s. These helpers bound
# the summary so no prompt can exceed the cap, trimming the lowest-value part
# (the tree) first and logging every cut (no silent truncation).
#
# We can only measure the user prompt we assemble, but the actual payload the
# endpoint counts also includes the ReAct agent's system prompt + tool schemas
# (~6k tokens in stage 3 repair) plus tokenizer drift vs the endpoint. So the
# clamp targets (cap - PROMPT_OVERHEAD_RESERVE_TOKENS), not the raw cap.
# ---------------------------------------------------------------------------

# --max-input-tokens is the endpoint's real input cap (gpt-4o = 64000).
DEFAULT_MAX_INPUT_TOKENS = 64000
# Headroom left below the cap for the parts of the payload we cannot measure
# from the assembled user prompt (ReAct system prompt + tool schemas + drift).
PROMPT_OVERHEAD_RESERVE_TOKENS = 10000

_TREE_HEADER = "## Directory structure"


@functools.lru_cache(maxsize=8)
def _token_encoder(model: str):
    import tiktoken

    try:
        return tiktoken.encoding_for_model(model)
    except KeyError:
        return tiktoken.get_encoding("cl100k_base")


def count_tokens(text: str, model: str = "gpt-4o") -> int:
    """Exact input-token count for ``text`` under ``model``'s encoding."""
    if not text:
        return 0
    return len(_token_encoder(model).encode(text))


def _shrink_tree_block(summary: str, budget_tokens: int, model: str) -> str:
    """Keep the largest prefix of the directory-tree block that lets the whole
    summary fit ``budget_tokens``; drop the rest behind an explicit marker."""
    header_idx = summary.find(_TREE_HEADER)
    if header_idx == -1:
        return summary
    fence_open = summary.find("```", header_idx)
    if fence_open == -1:
        return summary
    body_start = summary.find("\n", fence_open) + 1
    fence_close = summary.find("```", body_start)
    if fence_close == -1:
        return summary

    tree_lines = summary[body_start:fence_close].splitlines()
    prefix, suffix = summary[:body_start], summary[fence_close:]
    # Reserve headroom for the fixed parts plus the "omitted" marker line so the
    # reconstructed summary lands at or under budget (the marker is added after).
    marker_reserve = 64
    fixed_tokens = count_tokens(prefix + suffix, model) + marker_reserve

    kept: list[str] = []
    running = fixed_tokens
    for line in tree_lines:
        line_tokens = count_tokens(line + "\n", model)
        if running + line_tokens > budget_tokens:
            break
        kept.append(line)
        running += line_tokens

    if len(kept) == len(tree_lines):
        return summary
    omitted = len(tree_lines) - len(kept)
    marker = f"… [{omitted} of {len(tree_lines)} directory-tree entries omitted to fit the input-token budget]"
    return prefix + "\n".join(kept + [marker]) + "\n" + suffix


def _hard_truncate(text: str, budget_tokens: int, model: str) -> str:
    """Last-resort token clamp: slice ``text`` to ``budget_tokens`` and mark it."""
    enc = _token_encoder(model)
    marker = "\n… [content truncated to fit the input-token budget]"
    marker_tokens = count_tokens(marker, model)
    keep = max(0, budget_tokens - marker_tokens)
    return enc.decode(enc.encode(text)[:keep]) + marker


def bound_summary(summary: str, budget_tokens: int, model: str = "gpt-4o") -> str:
    """Return ``summary`` trimmed to at most ``budget_tokens``.

    Trims the directory-tree block first (lowest value per token); if that is
    not enough, hard-truncates the remainder. Warns whenever it cuts.
    """
    if budget_tokens <= 0:
        log_warn(f"[token-bound] summary budget is {budget_tokens}; dropping summary content")
        return _hard_truncate(summary, max(0, budget_tokens), model)
    if count_tokens(summary, model) <= budget_tokens:
        return summary
    shrunk = _shrink_tree_block(summary, budget_tokens, model)
    if count_tokens(shrunk, model) <= budget_tokens:
        log_warn(f"[token-bound] directory tree trimmed to fit {budget_tokens}-token summary budget")
        return shrunk
    log_warn(f"[token-bound] summary still over {budget_tokens} tokens after tree trim; hard-truncating")
    return _hard_truncate(shrunk, budget_tokens, model)


def clamp_summary_in_prompt(
    prompt: str,
    summary: str,
    max_input_tokens: int,
    model: str = "gpt-4o",
    *,
    phase: str = "",
) -> str:
    """Final guard: if ``prompt`` exceeds the effective cap, trim the embedded
    ``summary`` (the one unbounded region) just enough to fit. No-op when under.

    The effective cap is ``max_input_tokens - PROMPT_OVERHEAD_RESERVE_TOKENS`` so
    there is room for payload parts we cannot see here (ReAct system prompt + tool
    schemas + tokenizer drift); without that reserve the call still 400s."""
    effective_cap = max(0, max_input_tokens - PROMPT_OVERHEAD_RESERVE_TOKENS)
    total = count_tokens(prompt, model)
    if total <= effective_cap:
        return prompt
    summary_tokens = count_tokens(summary, model)
    budget = effective_cap - (total - summary_tokens)
    bounded = bound_summary(summary, budget, model)
    label = f"[token-bound]{(' ' + phase) if phase else ''}"
    log_warn(
        f"{label}: prompt {total} > effective cap {effective_cap} "
        f"(endpoint cap {max_input_tokens} - reserve {PROMPT_OVERHEAD_RESERVE_TOKENS}); "
        f"summary {summary_tokens} -> {count_tokens(bounded, model)} tokens"
    )
    return prompt.replace(summary, bounded, 1)


def load_architecture_scratchpad(repo_name: str, summaries_dir: Path) -> dict | None:
    scratchpad_path = summaries_dir / f"{repo_name}.architecture-scratchpad.yaml"
    if not scratchpad_path.exists():
        return None

    try:
        with open(scratchpad_path, "r", encoding="utf-8") as scratchpad_file:
            loaded = yaml.safe_load(scratchpad_file)
    except Exception:
        return None

    if not isinstance(loaded, dict):
        return None

    schema_version = str(loaded.get("schema_version", "")).strip()
    if schema_version and schema_version != ARCHITECTURE_SCRATCHPAD_SCHEMA_VERSION:
        log_warn(
            f"Skipping architecture scratchpad for {repo_name}: unsupported schema version {schema_version}"
        )
        return None

    if not schema_version:
        loaded["schema_version"] = ARCHITECTURE_SCRATCHPAD_SCHEMA_VERSION
    return loaded


def _now_utc_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _shared_repository_state_path(repo_name: str, summaries_dir: Path) -> Path:
    return summaries_dir / f"{repo_name}.shared-state.yaml"


def _new_shared_repository_state(repo_name: str, repo_url: str = "") -> dict:
    return {
        "schema_version": SHARED_REPOSITORY_STATE_SCHEMA_VERSION,
        "repo": repo_url,
        "repo_name": repo_name,
        "updated_at": _now_utc_iso(),
        "stages": {
            "classify": {},
            "dockerfile": {},
            "repair": {},
            "install_guide": {},
        },
        "signals": {
            "failure_hints": [],
        },
    }


def load_shared_repository_state(repo_name: str, summaries_dir: Path) -> dict | None:
    state_path = _shared_repository_state_path(repo_name, summaries_dir)
    if not state_path.exists():
        return None

    try:
        with open(state_path, "r", encoding="utf-8") as state_file:
            loaded = yaml.safe_load(state_file)
    except Exception:
        return None

    if not isinstance(loaded, dict):
        return None

    schema_version = str(loaded.get("schema_version", "")).strip()
    if schema_version and schema_version != SHARED_REPOSITORY_STATE_SCHEMA_VERSION:
        log_warn(
            f"Skipping shared repository state for {repo_name}: unsupported schema version {schema_version}"
        )
        return None

    if not schema_version:
        loaded["schema_version"] = SHARED_REPOSITORY_STATE_SCHEMA_VERSION
    return loaded


def upsert_shared_repository_state(
    repo_name: str,
    summaries_dir: Path,
    *,
    repo_url: str = "",
    stage_name: str | None = None,
    stage_update: dict | None = None,
    failure_hint: dict | None = None,
) -> dict:
    shared_state = load_shared_repository_state(repo_name, summaries_dir) or _new_shared_repository_state(
        repo_name,
        repo_url=repo_url,
    )

    if repo_url and not shared_state.get("repo"):
        shared_state["repo"] = repo_url

    stages = shared_state.setdefault("stages", {})
    signals = shared_state.setdefault("signals", {})
    failure_hints = signals.setdefault("failure_hints", [])
    if not isinstance(failure_hints, list):
        failure_hints = []
        signals["failure_hints"] = failure_hints

    if stage_name:
        stage_bucket = stages.setdefault(stage_name, {})
        if isinstance(stage_bucket, dict) and isinstance(stage_update, dict):
            stage_bucket.update(stage_update)
            stage_bucket["updated_at"] = _now_utc_iso()

    if isinstance(failure_hint, dict) and failure_hint:
        hint_record = dict(failure_hint)
        hint_record.setdefault("recorded_at", _now_utc_iso())
        failure_hints.append(hint_record)
        if len(failure_hints) > 30:
            del failure_hints[:-30]

    shared_state["updated_at"] = _now_utc_iso()

    state_path = _shared_repository_state_path(repo_name, summaries_dir)
    state_path.parent.mkdir(parents=True, exist_ok=True)
    with open(state_path, "w", encoding="utf-8") as state_file:
        yaml.dump(shared_state, state_file, sort_keys=False, allow_unicode=True)

    return shared_state


def render_shared_repository_state_for_prompt(shared_state: dict | None, *, max_chars: int = 8000) -> str:
    if not shared_state:
        return ""

    rendered = yaml.dump(shared_state, sort_keys=False, allow_unicode=True)
    safe_max = max(256, int(max_chars))
    if len(rendered) > safe_max:
        head_chars = max(128, safe_max // 2)
        tail_chars = max(128, safe_max - head_chars)
        rendered = (
            rendered[:head_chars]
            + "\n... [shared repository state truncated] ...\n"
            + rendered[-tail_chars:]
        )

    return (
        "\n\nSHARED_REPOSITORY_STATE:\n"
        "Use this as cross-phase memory, including prior stage outputs and failure signals.\n"
        f"{rendered}\n"
    )


def render_architecture_scratchpad_for_prompt(scratchpad: dict | None, *, max_chars: int = 8000) -> str:
    if not scratchpad:
        return ""

    rendered = yaml.dump(scratchpad, sort_keys=False, allow_unicode=True)
    safe_max = max(256, int(max_chars))
    if len(rendered) > safe_max:
        head_chars = max(128, safe_max // 2)
        tail_chars = max(128, safe_max - head_chars)
        rendered = (
            rendered[:head_chars]
            + "\n... [architecture scratchpad truncated] ...\n"
            + rendered[-tail_chars:]
        )

    return (
        "\n\nARCHITECTURE_SCRATCHPAD:\n"
        "Use this cross-phase exploration/synthesis/validation context when making decisions.\n"
        f"{rendered}\n"
    )


def render_validation_findings_for_prompt(validation_artifact: dict | None) -> str:
    if not validation_artifact:
        return ""

    warnings = validation_artifact.get("warnings") or []
    checks = validation_artifact.get("checks") or {}
    if not warnings and not checks:
        return ""

    rendered = yaml.dump(validation_artifact, sort_keys=False, allow_unicode=True)
    return (
        "\n\nVALIDATION_FINDINGS:\n"
        "Treat warnings as evidence gaps and prefer conservative, reproducible build assumptions when uncertain.\n"
        f"{rendered}\n"
    )


def build_initial_user_request(config_hint: str, language: str = "") -> dict:
    """Structured internal representation of the per-repo initial user request
    (TODO 18): the build configuration the user wants plus the stated language.
    Returns {} when no config_hint was provided so callers can skip seeding."""
    hint = (config_hint or "").strip()
    if not hint:
        return {}
    request = {"config_hint": hint}
    lang = (language or "").strip()
    if lang:
        request["language"] = lang
    return request


def render_initial_user_request_for_prompt(shared_state: dict | None) -> str:
    """Surface the seeded initial user request (TODO 18) as a first-class prompt
    block. Reads it from shared repository state so every consuming stage shows
    the same target the user asked for. Returns '' when none was seeded."""
    if not isinstance(shared_state, dict):
        return ""
    stages = shared_state.get("stages")
    pipeline_stage = stages.get("pipeline") if isinstance(stages, dict) else None
    constraints = pipeline_stage.get("user_constraints") if isinstance(pipeline_stage, dict) else None
    request = constraints.get("initial_user_request") if isinstance(constraints, dict) else None
    if not isinstance(request, dict):
        return ""
    config_hint = str(request.get("config_hint", "")).strip()
    if not config_hint:
        return ""
    language = str(request.get("language", "")).strip()
    lang_clause = f"This is a {language} project. " if language else ""
    return (
        "\n\nINITIAL_USER_REQUEST:\n"
        "The user asked for this specific build outcome. Treat it as the goal your "
        "classification and Dockerfile must serve.\n"
        f"{lang_clause}{config_hint}\n"
    )


def render_yaml(data: dict) -> str:
    return yaml.dump(data, sort_keys=False, allow_unicode=True)


def prompts_dir() -> Path:
    # Preferred layout: RepoBuilderAgent/prompts
    root_prompts = Path(__file__).resolve().parents[2] / "prompts"
    if root_prompts.exists():
        return root_prompts

    # Backward-compatible fallback for older layouts.
    return Path(__file__).resolve().parent.parent / "prompts"


# Active prompt verbosity. "detailed" loads the full PROMPT_X.md; "concise" loads a
# pre-generated, caveman-compressed PROMPT_X.concise.md sibling when it exists. This is
# how the AB-03 verbosity arm differs from the rest of Phase 1 — by real prompt content,
# not a directive sentence. Stages set this once at startup (after resolving their
# prompt profile, before reading any prompt) via set_prompt_length_mode(). Mirrors the
# existing set_dump_prompts_dir() module-global pattern.
_PROMPT_LENGTH_MODE = "detailed"


def set_prompt_length_mode(mode: str | None) -> None:
    global _PROMPT_LENGTH_MODE
    _PROMPT_LENGTH_MODE = "concise" if str(mode or "").strip().lower() == "concise" else "detailed"


def prompt_path(name: str) -> Path:
    base = prompts_dir()
    if _PROMPT_LENGTH_MODE == "concise" and name.endswith(".md"):
        concise = base / f"{name[:-len('.md')]}.concise.md"
        if concise.exists():
            return concise
    return base / name


def should_use_progress(total_repos: int, trace: bool) -> bool:
    return total_repos > 1 and sys.stderr.isatty() and not trace


async def update_progress(progress_state: dict, repo_name: str) -> None:
    if progress_state["bar"] is None:
        return
    async with progress_state["lock"]:
        progress_state["bar"].set_postfix_str(repo_name)
        progress_state["bar"].update(1)


async def ensure_repo_checkout(repo_url: str, repo_path: Path, skip_reason: str = "skipping") -> bool:
    if repo_path.exists():
        return True
    log_info(f"Cloning {repo_url} -> {repo_path}")
    result = await asyncio.to_thread(
        subprocess.run, ["git", "clone", repo_url, str(repo_path)], check=False
    )
    if result.returncode != 0:
        log_warn(f"Failed to clone {repo_url}; {skip_reason}.")
        return False
    return True


def resolve_repo_checkout_dir(repos_dir: Path, repo_name: str) -> Path:
    """Resolve ``repos_dir/<repo_name>`` against existing directories case-insensitively.

    Checkouts are prepared (by eval.py) under language subfolders using the
    repository's canonical name, whose casing can differ from the URL-derived
    name (e.g. on-disk ``cjson`` vs URL segment ``cJSON``). Returning the actual
    existing directory keeps every stage pointed at the prepared checkout and
    prevents re-cloning into a mis-cased path. Falls back to the literal
    ``repos_dir/<repo_name>`` when nothing matches.
    """
    candidate = repos_dir / repo_name
    if candidate.exists():
        return candidate
    if repos_dir.exists():
        target = repo_name.lower()
        for child in repos_dir.iterdir():
            if child.is_dir() and child.name.lower() == target:
                return child
    return candidate


async def validate_dockerfile_syntax(dockerfile_path: Path, repo_name: str = "") -> tuple[bool, str]:
    """
    Validate Dockerfile syntax using hadolint.
    Returns (is_valid, error_message).
    """
    hadolint_path = Path("/usr/local/bin/hadolint")
    if not hadolint_path.exists():
        log_warn(f"[hadolint {repo_name}] hadolint not found at {hadolint_path}; skipping validation")
        return True, ""

    repo_root = Path(__file__).resolve().parents[2]
    hadolint_config = repo_root / ".hadolint.yaml"
    command = [str(hadolint_path)]
    if hadolint_config.exists():
        command.extend(["--config", str(hadolint_config)])
    command.append(str(dockerfile_path))

    try:
        result = await asyncio.to_thread(
            subprocess.run,
            command,
            capture_output=True,
            text=True,
            timeout=10
        )
        if result.returncode == 0:
            log_info(f"[hadolint {repo_name}] Dockerfile syntax OK")
            return True, ""
        else:
            error_msg = result.stdout + result.stderr
            log_warn(f"[hadolint {repo_name}] Dockerfile syntax error: {error_msg[:500]}")
            return False, error_msg
    except subprocess.TimeoutExpired:
        error_msg = "hadolint timeout after 10s"
        log_warn(f"[hadolint {repo_name}] {error_msg}")
        return False, error_msg
    except Exception as e:
        log_warn(f"[hadolint {repo_name}] Validation failed: {e}")
        return False, str(e)
