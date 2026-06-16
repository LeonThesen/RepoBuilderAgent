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
    """Remove the repo's declared ``files_to_delete`` globs before the image build.

    The benchmark's premise is that the agent builds WITHOUT install documentation, so
    docs (and CI/CD config) are stripped here. The strip-list is fully declarative and
    per-repo, sourced from the dataset YAML:

      * Every path removed is on the repo's ``files_to_delete`` list — no implicit CI
        deletion, no legacy doc-dir heuristic, no build-metadata carve-out. What you
        see in the YAML is exactly what is removed.
      * Globs are file-scoped by convention so build-referenced files survive (e.g.
        curl's docs/examples, add_subdirectory'd by CMake): list ``docs/**/*.md``, not
        ``docs``. A glob that matches a directory rmtree's it — used deliberately for
        CI dirs like ``.github``.
      * Build-input docs (fmt's README.md in add_library, curl's man-page sources,
        git's command-list .adoc) are simply NOT listed, so they remain for the build.

    When ``files_to_delete`` is empty/None nothing is removed (and a warning is logged):
    an un-curated repo is left intact rather than silently over-stripped.
    """
    log_info(f"[delete-files {repo_name}] Starting declarative file deletion in {repo_path}")

    if not files_to_delete:
        log_warn(
            f"[delete-files {repo_name}] No files_to_delete in dataset YAML; nothing "
            f"removed. Curate the repo's files_to_delete list to strip docs/CI."
        )
        return

    removed_files: list[str] = []
    removed_dirs: list[str] = []
    for pattern in files_to_delete:
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
            log_trace(f"[delete-files {repo_name}] removed {rel}")

    log_info(
        f"[delete-files {repo_name}] Done — removed {len(removed_files)} file(s), "
        f"{len(removed_dirs)} director{'ies' if len(removed_dirs) != 1 else 'y'} "
        f"({len(files_to_delete)} pattern(s))"
    )
