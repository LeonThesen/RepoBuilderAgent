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

_DELETE_DOCS_DIR_NAMES: frozenset[str] = frozenset({
    ".github", ".gitlab", ".circleci", ".buildkite", ".drone",
    ".woodpecker", ".jenkins", ".azure-pipelines",
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


def delete_docs_build_context(repo_path: Path, repo_name: str) -> None:
    """Remove documentation and CI/CD files from repo_path before image builds."""
    log_info(f"[delete-docs {repo_name}] Starting docs/CI deletion in {repo_path}")

    removed_files: list[str] = []
    removed_dirs: list[str] = []

    for item in list(repo_path.rglob("*")):
        if not item.exists():
            continue

        if item.is_dir():
            if item.name in _DELETE_DOCS_DIR_NAMES:
                rel = item.relative_to(repo_path)
                if len(rel.parts) <= 2:
                    log_trace(f"[delete-docs {repo_name}] Removing CI/CD directory: {rel}")
                    shutil.rmtree(item)
                    removed_dirs.append(str(rel))
                else:
                    log_trace(f"[delete-docs {repo_name}] Skipping deep source dir: {rel}")
        elif item.is_file():
            if item.suffix.lower() in _DELETE_DOCS_EXTENSIONS:
                rel = item.relative_to(repo_path)
                log_trace(f"[delete-docs {repo_name}] Removing doc file: {rel}")
                item.unlink()
                removed_files.append(str(rel))
            elif item.name in _DELETE_DOCS_FILE_NAMES:
                rel = item.relative_to(repo_path)
                log_trace(f"[delete-docs {repo_name}] Removing CI/CD file: {rel}")
                item.unlink()
                removed_files.append(str(rel))

    log_info(
        f"[delete-docs {repo_name}] Done — removed {len(removed_files)} file(s), "
        f"{len(removed_dirs)} director{'ies' if len(removed_dirs) != 1 else 'y'}"
    )
