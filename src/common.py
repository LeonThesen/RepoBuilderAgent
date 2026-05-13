"""Shared utilities for the RepoBuilderAgent pipeline scripts."""

from __future__ import annotations

import asyncio
import base64
import json
import os
import subprocess
import sys
from pathlib import Path

import yaml

from log_utils import log_info, log_warn
from repo_fingerprint import fingerprint


def inject_ca_cert_into_dockerfile(dockerfile_content: str, ca_cert_b64: str | None = None) -> str:
    """
    Inject CA certificate setup into the Dockerfile if MANUALREPOS_CA_CERT_B64 is present.
    This ensures curl/git/wget/npm/pip/maven/rust/cargo can reach package registries behind TLS interception.

    Sets up:
    - System CA trust store (apt, curl, git, wget)
    - Environment variables for Python (pip), Node.js (npm/pnpm), Java (maven)
    - Java keystore import for Maven/Gradle
    - Rust/cargo CA configuration
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
RUN if command -v keytool >/dev/null 2>&1; then keytool -import -alias custom-ca -file {ca_cert_path} -keystore /etc/ssl/certs/java/cacerts -storepass changeit -noprompt 2>/dev/null || true; fi
ENV SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt
ENV REQUESTS_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt
ENV CURL_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt
ENV NODE_EXTRA_CA_CERTS=/etc/ssl/certs/ca-certificates.crt
ENV CARGO_HTTP_CAINFO=/etc/ssl/certs/ca-certificates.crt
ENV SSL_CERT_DIR=/etc/ssl/certs
"""

    lines = dockerfile_content.split("\n")
    injected_lines = []
    inserted = False

    for line in lines:
        injected_lines.append(line)
        # Insert CA setup after first FROM line
        if not inserted and line.strip().upper().startswith("FROM"):
            injected_lines.append("USER root")
            injected_lines.append(ca_setup_commands.strip())
            inserted = True

    return "\n".join(injected_lines)


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
    
    try:
        result = await asyncio.to_thread(
            subprocess.run,
            [str(hadolint_path), str(dockerfile_path)],
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
