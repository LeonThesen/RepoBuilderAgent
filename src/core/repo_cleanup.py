from pathlib import Path
import shutil

try:
    from RepoBuilderAgent.src.core.log_utils import log_info, log_trace, log_warn
except ImportError:
    from core.log_utils import log_info, log_trace, log_warn


def get_files_to_delete(gt_doc: "dict | None") -> list[str]:
    """Extract the per-repo strip-list from a dataset YAML doc.

    Returns the ``files_to_delete`` list (repo-root-relative globs) or [] when
    absent/malformed. This is the single, fully declarative source of truth for what
    is removed before the build — documentation AND CI/CD config. There is no implicit
    deletion and no implicit keep: exactly the listed globs are removed, nothing else.
    """
    if not isinstance(gt_doc, dict):
        return []
    value = gt_doc.get("files_to_delete")
    if not isinstance(value, list):
        return []
    return [str(p).strip() for p in value if isinstance(p, str) and p.strip()]


def delete_files_build_context(
    repo_path: Path,
    repo_name: str,
    files_to_delete: "list[str] | None" = None,
) -> None:
    """Remove or overwrite the repo's declared ``files_to_delete`` globs before the image build.

    Pattern prefixes:
    - "!" protects matching files/directories from deletion.
    - ">" overwrites matching files with an empty file instead of deleting them.

    All other patterns are deleted.
    """
    print(f"[delete-files {repo_name}] Starting declarative file deletion in {repo_path}")

    if not files_to_delete:
        print(
            f"[delete-files {repo_name}] No files_to_delete in dataset YAML; nothing "
            f"removed. Curate the repo's files_to_delete list to strip docs/CI."
        )
        return

    removed_files: list[str] = []
    removed_dirs: list[str] = []
    overwritten_files: list[str] = []

    delete_patterns: list[str] = []
    protected_patterns: list[str] = []
    overwrite_patterns: list[str] = []

    for pattern in files_to_delete:
        if pattern.startswith("!"):
            protected_patterns.append(pattern[1:])
        elif pattern.startswith(">"):
            overwrite_patterns.append(pattern[1:])
        else:
            delete_patterns.append(pattern)

    protected_paths: set[Path] = set()
    protected_dirs: set[Path] = set()
    overwrite_paths: set[Path] = set()

    # Expand protected glob patterns first.
    for pattern in protected_patterns:
        for protected_match in repo_path.glob(pattern):
            if not protected_match.exists():
                continue

            rel = protected_match.relative_to(repo_path)
            protected_paths.add(rel)

            if protected_match.is_dir():
                protected_dirs.add(rel)

    # Expand overwrite glob patterns next.
    # Overwritten files are also protected from later delete patterns.
    for pattern in overwrite_patterns:
        for overwrite_match in repo_path.glob(pattern):
            if not overwrite_match.exists():
                continue

            rel = overwrite_match.relative_to(repo_path)
            overwrite_paths.add(rel)
            protected_paths.add(rel)

            if overwrite_match.is_dir():
                protected_dirs.add(rel)

    def is_protected(match: Path) -> bool:
        rel = match.relative_to(repo_path)

        # Exact protected file/dir match.
        if rel in protected_paths:
            return True

        # Anything inside a protected directory is protected.
        for protected_dir in protected_dirs:
            if rel == protected_dir or protected_dir in rel.parents:
                return True

        # Do not delete a directory if it contains a protected file/dir.
        if match.is_dir():
            for protected_path in protected_paths:
                if protected_path == rel or rel in protected_path.parents:
                    return True

        return False

    # Apply overwrite patterns.
    for rel in sorted(overwrite_paths):
        match = repo_path / rel

        if not match.exists():
            continue

        if match.is_dir():
            log_trace(f"[delete-files {repo_name}] skipped overwrite for directory {rel}")
            continue

        match.write_text("", encoding="utf-8")
        overwritten_files.append(str(rel))

        log_trace(f"[delete-files {repo_name}] overwritten {rel}")

    # Apply delete patterns.
    for pattern in delete_patterns:
        for match in repo_path.glob(pattern):
            if not match.exists():
                continue

            if is_protected(match):
                log_trace(
                    f"[delete-files {repo_name}] protected "
                    f"{match.relative_to(repo_path)}"
                )
                continue

            rel = match.relative_to(repo_path)

            if match.is_dir():
                shutil.rmtree(match)
                removed_dirs.append(str(rel))
            else:
                match.unlink()
                removed_files.append(str(rel))

            log_trace(f"[delete-files {repo_name}] removed {rel}")

    log_trace(
        f"[delete-files {repo_name}] Done — removed {len(removed_files)} file(s), "
        f"{len(removed_dirs)} director{'ies' if len(removed_dirs) != 1 else 'y'}, "
        f"overwritten {len(overwritten_files)} file(s) "
        f"({len(files_to_delete)} pattern(s), "
        f"{len(protected_patterns)} protection pattern(s), "
        f"{len(overwrite_patterns)} overwrite pattern(s))"
    )