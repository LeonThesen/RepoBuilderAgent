from pathlib import Path
import glob
import shutil

import yaml

try:
    from RepoBuilderAgent.src.core.log_utils import log_info, log_trace, log_warn
except ImportError:
    from core.log_utils import log_info, log_trace, log_warn


_DELETE_DOCS_EXTENSIONS: tuple[str, ...] = (
    ".md", ".rst", ".adoc", ".asciidoc", ".textile", ".wiki",
    ".pdf", ".doc", ".docx", ".odt", ".rtf",
    ".tex", ".pod", ".man",
    ".ipynb", ".pptx", ".ppt",
)

# Build-metadata files that happen to use a doc extension but are NOT install
# documentation — build systems reference/embed them (CMake globbing ChangeLog.md,
# packaging steps reading LICENSE.md). Keep these even though they look like docs.
# These are legal/metadata files, not "how to build" instructions, so preserving
# them does not leak the install procedure to the agent. Matched on the filename
# stem (case-insensitive), so LICENSE.md / ChangeLog.rst / NOTICE.adoc all qualify.
_KEEP_DOC_FILE_STEMS: frozenset[str] = frozenset({
    "license", "licence", "copying", "copyright", "notice", "unlicense",
    "changelog", "changes", "history", "news",
    "authors", "contributors", "credits",
})

_DELETE_DOCS_FILE_NAMES: frozenset[str] = frozenset({
    "Jenkinsfile",
    ".travis.yml", ".travis.yaml",
    ".gitlab-ci.yml", ".gitlab-ci.yaml",
    "appveyor.yml", "appveyor.yaml",
    "azure-pipelines.yml", "azure-pipelines.yaml",
    ".drone.yml", ".drone.yaml",
    "bitbucket-pipelines.yml", "bitbucket-pipelines.yaml",
    "CODEOWNERS",
})

# CI/CD directories — stripped universally: uniform across repos and never a build
# input, so they need no per-repo curation and can't break a build.
_CI_DIR_NAMES: frozenset[str] = frozenset({
    ".github", ".gitlab", ".circleci", ".buildkite", ".drone",
    ".woodpecker", ".jenkins", ".azure-pipelines",
})

# Documentation directory names — used ONLY by the legacy fallback for repos that
# don't yet declare an explicit docs_to_delete list. The blanket rmtree of these is
# what broke builds whose source lives under a doc dir (e.g. curl's docs/examples,
# add_subdirectory'd by CMake). The dataset-driven path replaces it: each repo lists
# the exact doc paths safe to strip, so build-referenced files survive.
_LEGACY_DOC_DIR_NAMES: frozenset[str] = frozenset({
    "docs", "doc", "documentation",
    "website", "site", "gh-pages",
    "wiki",
    "javadoc", "apidoc", "apidocs",
    "man", "manpages",
    "sphinx", "docsrc",
})


def preprocess_repository(repo_path: Path, deletion_patterns_file: str) -> None:
    """Remove files/dirs using YAML-configured deletion patterns."""
    if not Path(deletion_patterns_file).exists():
        log_warn(f"Deletion patterns file not found: {deletion_patterns_file}. Skipping preprocessing.")
        return

    with open(deletion_patterns_file, "r", encoding="utf-8") as f:
        patterns_config = yaml.safe_load(f)

    if not patterns_config:
        log_warn("Deletion patterns config is empty. Skipping preprocessing.")
        return

    deleted_count = 0

    extension_patterns = patterns_config.get("extension_patterns", [])
    for ext_pattern in extension_patterns:
        glob_pattern = f"**/{ext_pattern}"
        matches = glob.glob(str(repo_path / glob_pattern), recursive=True)
        for file_path in matches:
            try:
                if Path(file_path).is_file():
                    Path(file_path).unlink()
                    deleted_count += 1
                    log_trace(f"Deleted file: {file_path}")
            except Exception as error:
                log_warn(f"Failed to delete file {file_path}: {error}")

    file_patterns = patterns_config.get("file_patterns", [])
    for pattern in file_patterns:
        matches = glob.glob(str(repo_path / pattern), recursive=True)
        for file_path in matches:
            try:
                if Path(file_path).is_file():
                    Path(file_path).unlink()
                    deleted_count += 1
                    log_trace(f"Deleted file: {file_path}")
            except Exception as error:
                log_warn(f"Failed to delete file {file_path}: {error}")

    directory_patterns = patterns_config.get("directory_patterns", [])
    for pattern in directory_patterns:
        matches = glob.glob(str(repo_path / pattern), recursive=True)
        for dir_path in matches:
            try:
                if Path(dir_path).is_dir():
                    shutil.rmtree(dir_path)
                    deleted_count += 1
                    log_trace(f"Deleted directory: {dir_path}")
            except Exception as error:
                log_warn(f"Failed to delete directory {dir_path}: {error}")

    if deleted_count > 0:
        log_info(f"Preprocessing complete: deleted {deleted_count} files/directories from {repo_path.name}")


def _is_build_metadata_doc(item: Path) -> bool:
    """A doc-extension file that is build metadata (LICENSE/CHANGELOG/…), not
    install documentation — preserved so builds that reference it don't break."""
    return item.stem.lower() in _KEEP_DOC_FILE_STEMS


def get_docs_to_delete(gt_doc: "dict | None") -> list[str]:
    """Extract the per-repo documentation strip-list from a dataset YAML doc.

    Returns the ``docs_to_delete`` list (glob paths relative to the repo root) or
    [] when absent/malformed. [] signals 'not curated' to the deleter, which then
    falls back to the legacy heuristic.
    """
    if not isinstance(gt_doc, dict):
        return []
    value = gt_doc.get("docs_to_delete")
    if not isinstance(value, list):
        return []
    return [str(p).strip() for p in value if isinstance(p, str) and p.strip()]


def _delete_ci_configs(repo_path: Path, repo_name: str, removed_files: list[str], removed_dirs: list[str]) -> None:
    """Strip CI/CD config files + directories. Universal (uniform, never build inputs)."""
    for item in list(repo_path.rglob("*")):
        if not item.exists():
            continue
        if item.is_dir() and item.name in _CI_DIR_NAMES:
            rel = item.relative_to(repo_path)
            shutil.rmtree(item)
            removed_dirs.append(str(rel))
        elif item.is_file() and item.name in _DELETE_DOCS_FILE_NAMES:
            rel = item.relative_to(repo_path)
            item.unlink()
            removed_files.append(str(rel))


def delete_docs_build_context(
    repo_path: Path,
    repo_name: str,
    docs_to_delete: "list[str] | None" = None,
) -> None:
    """Remove documentation and CI/CD files from repo_path before image builds.

    The benchmark's premise is that the agent builds WITHOUT install documentation,
    so docs are stripped. Two layers:

      * CI/CD configs (.github, Jenkinsfile, …) are always removed — uniform across
        repos and never build inputs.
      * Documentation is removed from the per-repo ``docs_to_delete`` glob list,
        sourced from the repo's dataset YAML. This lists exactly the doc paths that
        are safe to strip, so build-referenced files that happen to live under a doc
        dir (e.g. curl's docs/examples, add_subdirectory'd by CMake) are preserved by
        simply not being on the list. The committed list is auditable and explains,
        per repo, what was stripped and why.

    When ``docs_to_delete`` is empty/None (repo not yet curated) it falls back to the
    legacy heuristic (doc-extension sweep + doc-dir rmtree, minus build-metadata
    files) and logs a warning, so an un-migrated repo still gets stripped.
    """
    log_info(f"[delete-docs {repo_name}] Starting docs/CI deletion in {repo_path}")

    removed_files: list[str] = []
    removed_dirs: list[str] = []

    _delete_ci_configs(repo_path, repo_name, removed_files, removed_dirs)

    if docs_to_delete:
        for pattern in docs_to_delete:
            # Patterns are repo-root-relative globs; recursive so ** works.
            for match in repo_path.glob(pattern):
                if not match.exists():
                    continue
                rel = match.relative_to(repo_path)
                if match.is_dir():
                    shutil.rmtree(match)
                    removed_dirs.append(str(rel))
                else:
                    match.unlink()
                    removed_files.append(str(rel))
        log_info(
            f"[delete-docs {repo_name}] Done (dataset docs_to_delete) — removed "
            f"{len(removed_files)} file(s), {len(removed_dirs)} "
            f"director{'ies' if len(removed_dirs) != 1 else 'y'}"
        )
        return

    # ── Legacy fallback: repo has no curated docs_to_delete ──────────────────────
    log_warn(
        f"[delete-docs {repo_name}] No docs_to_delete in dataset YAML; using legacy "
        f"heuristic (may over-strip and break builds that reference doc dirs)."
    )
    kept_metadata: list[str] = []
    for item in list(repo_path.rglob("*")):
        if not item.exists():
            continue
        if item.is_dir():
            if item.name in _LEGACY_DOC_DIR_NAMES:
                rel = item.relative_to(repo_path)
                if len(rel.parts) <= 2:
                    shutil.rmtree(item)
                    removed_dirs.append(str(rel))
        elif item.is_file() and item.suffix.lower() in _DELETE_DOCS_EXTENSIONS:
            if _is_build_metadata_doc(item):
                kept_metadata.append(str(item.relative_to(repo_path)))
                continue
            item.unlink()
            removed_files.append(str(item.relative_to(repo_path)))

    log_info(
        f"[delete-docs {repo_name}] Done (legacy) — removed {len(removed_files)} file(s), "
        f"{len(removed_dirs)} director{'ies' if len(removed_dirs) != 1 else 'y'}; "
        f"kept {len(kept_metadata)} build-metadata file(s)"
    )
