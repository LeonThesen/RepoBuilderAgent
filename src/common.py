"""Shared utilities for the RepoBuilderAgent pipeline scripts."""

from __future__ import annotations

import asyncio
import base64
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import yaml
from openai import APIConnectionError, APIError, APITimeoutError

try:
    from RepoBuilderAgent.src.log_utils import log_info, log_warn
    from RepoBuilderAgent.src.repo_fingerprint import fingerprint
except ImportError:
    # Fallback for direct script execution from RepoBuilderAgent/src
    from log_utils import log_info, log_warn
    from repo_fingerprint import fingerprint


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
):
    bucket = _phase_bucket(metrics, phase)
    last_error: Exception | None = None

    for attempt in range(1, max_retries + 2):
        bucket["calls"] += 1
        started = time.perf_counter()
        try:
            response = await client.with_options(timeout=timeout_seconds).chat.completions.create(
                model=model,
                temperature=temperature,
                messages=messages,
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
        except APITimeoutError as error:
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
    def ensure_root_for_chown(content: str) -> str:
        """Ensure RUN chown commands execute as root, then restore prior user."""
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

            is_chown_run = upper.startswith("RUN ") and "CHOWN" in upper
            if is_chown_run and current_user not in {"root", "0"}:
                rewritten.append("USER root")
                rewritten.append(raw_line)
                rewritten.append(f"USER {current_user}")
            else:
                rewritten.append(raw_line)

        return "\n".join(rewritten)

    if not ca_cert_b64:
        ca_cert_b64 = os.getenv("MANUALREPOS_CA_CERT_B64")
    if not ca_cert_b64:
        return ensure_root_for_chown(dockerfile_content)

    # Avoid repeatedly injecting very large CA blocks on retries/repair passes.
    if "manualrepos-refresh-ca-certificates" in dockerfile_content:
        return ensure_root_for_chown(dockerfile_content)

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
            injected_lines.append("USER root")
            injected_lines.append(ca_setup_commands.strip())
            inserted = True
        else:
            injected_lines.append(line)

    return ensure_root_for_chown("\n".join(injected_lines))


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


def render_yaml(data: dict) -> str:
    return yaml.dump(data, sort_keys=False, allow_unicode=True)


def prompts_dir() -> Path:
    return Path(__file__).resolve().parent.parent / "prompts"


def prompt_path(name: str) -> Path:
    return prompts_dir() / name


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
        error_msg = f"hadolint timeout after 10s"
        log_warn(f"[hadolint {repo_name}] {error_msg}")
        return False, error_msg
    except Exception as e:
        log_warn(f"[hadolint {repo_name}] Validation failed: {e}")
        return False, str(e)
