"""Shared utilities for the RepoBuilderAgent pipeline scripts."""

from __future__ import annotations

import asyncio
import json
import subprocess
import sys
from pathlib import Path

import yaml

from log_utils import log_info, log_warn
from repo_fingerprint import fingerprint


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
