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
    - Java keystore import into the DEFAULT OpenJDK keystore (preserves public root CAs)
    - Rust/cargo CA configuration
    """
    if not ca_cert_b64:
        ca_cert_b64 = os.getenv("MANUALREPOS_CA_CERT_B64")
    if not ca_cert_b64:
        return dockerfile_content

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
    current_user_is_root = True

    for line in lines:
        stripped_line = line.strip()
        lower_line = stripped_line.lower()
        if inserted and (
            (stripped_line.upper().startswith("USER") and lower_line != "user root")
            or (
                stripped_line.upper().startswith("RUN")
                and current_user_is_root
                and ca_refresh_script_path not in lower_line
                and any(token in lower_line for token in ("gradle", "gradlew", "mvn", "maven", " java"))
            )
        ):
            injected_lines.append(f"RUN {ca_refresh_script_path}")
        injected_lines.append(line)
        # Insert CA setup after first FROM line
        if not inserted and line.strip().upper().startswith("FROM"):
            injected_lines.append("USER root")
            injected_lines.append(ca_setup_commands.strip())
            inserted = True
        if stripped_line.upper().startswith("USER"):
            current_user_is_root = lower_line == "user root"

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
