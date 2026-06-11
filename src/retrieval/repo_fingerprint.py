#!/usr/bin/env python3
"""
repo_fingerprint.py

Condenses a repository into a structured, prompt-friendly fingerprint
for LLM-based analysis (build tools, runtime, language, CI, etc.)
"""

import os
import re
import math
import fnmatch
import yaml
from pathlib import Path
from typing import Optional

try:
    from RepoBuilderAgent.src.core.log_utils import log_info, log_trace
except ImportError:
    # Fallback for direct script execution from RepoBuilderAgent/src
    from core.log_utils import log_info, log_trace

# ---------------------------------------------------------------------------
# Config Loading
# ---------------------------------------------------------------------------

def _find_config_dir() -> Path:
    """Walk ancestor directories until config/manifest-files.yaml is found."""
    current = Path(__file__).resolve().parent
    for _ in range(8):
        if (current / "config" / "manifest-files.yaml").exists():
            return current / "config"
        current = current.parent
    return Path(__file__).parent.parent / "config"  # original fallback


def load_config() -> tuple[set, list, dict]:
    """Load manifest patterns from config YAML files."""
    config_dir = _find_config_dir()
    
    # Load manifest-files.yaml
    manifest_path = config_dir / "manifest-files.yaml"
    patterns_path = config_dir / "patterns.yaml"
    
    full_read_files = set()
    full_read_patterns = []
    partial_read = {}
    
    if manifest_path.exists():
        with open(manifest_path, "r") as f:
            manifest_config = yaml.safe_load(f) or {}
            full_read_files = set(manifest_config.get("exact_matches", []))
            # Keep glob patterns as set for consistency with original code
            full_read_files.update(manifest_config.get("glob_patterns", []))
            partial_read = manifest_config.get("partial_read_files", {})
    
    if patterns_path.exists():
        with open(patterns_path, "r") as f:
            patterns_config = yaml.safe_load(f) or {}
            full_read_patterns = patterns_config.get("patterns", [])
    
    return full_read_files, full_read_patterns, partial_read


# Load config on module import
FULL_READ_FILES, FULL_READ_PATTERNS, PARTIAL_READ = load_config()

# Directories to skip entirely
SKIP_DIRS = {
    ".git", "node_modules", "__pycache__", ".venv", "venv", "env",
    ".tox", "dist", "build", "target", ".next", ".nuxt", "out",
    ".gradle", ".idea", ".vscode", "coverage", ".nyc_output",
    "vendor", "third_party", ".terraform", ".serverless",
    ".eggs", "*.egg-info",
}

# Hidden directories that often contain high-signal build/deploy metadata.
INCLUDE_HIDDEN_DIRS = {".github", ".circleci", ".devcontainer"}

MAX_TREE_DEPTH = 4
MAX_TREE_FILES_PER_DIR = 30    # truncate busy dirs
MAX_FILE_SIZE = 64 * 1024      # skip files > 64 KB


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def should_skip_dir(name: str) -> bool:
    if name in INCLUDE_HIDDEN_DIRS:
        return False
    return name in SKIP_DIRS or name.startswith(".")


def normalize_relative_path(path: str) -> str:
    """Normalize a user/model-provided relative path for safe lookup under root."""
    return str(Path(path.strip()).as_posix()).lstrip("/")


def read_file_safe(path: Path, max_bytes: Optional[int] = None) -> str:
    try:
        if path.stat().st_size > MAX_FILE_SIZE and max_bytes is None:
            return f"[skipped: file too large ({path.stat().st_size // 1024} KB)]"
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            content = f.read(max_bytes) if max_bytes else f.read()
        if max_bytes and len(content) == max_bytes:
            content += "\n... [truncated]"
        return content.strip()
    except Exception as e:
        return f"[error reading file: {e}]"


def build_tree(root: Path, depth: int = 0) -> list[str]:
    if depth > MAX_TREE_DEPTH:
        return []
    lines = []
    try:
        entries = sorted(root.iterdir(), key=lambda p: (p.is_file(), p.name.lower()))
    except PermissionError:
        return []

    dirs = [e for e in entries if e.is_dir() and not should_skip_dir(e.name)]
    files = [e for e in entries if e.is_file()]

    for d in dirs:
        lines.append("  " * depth + f"📁 {d.name}/")
        lines.extend(build_tree(d, depth + 1))

    shown = files[:MAX_TREE_FILES_PER_DIR]
    for f in shown:
        lines.append("  " * depth + f"  {f.name}")
    if len(files) > MAX_TREE_FILES_PER_DIR:
        lines.append("  " * depth + f"  ... ({len(files) - MAX_TREE_FILES_PER_DIR} more files)")

    return lines


def match_glob_patterns(root: Path) -> list[tuple[str, str]]:
    """Returns (relative_path, content) for files matching FULL_READ_PATTERNS."""
    results = []
    for pattern in FULL_READ_PATTERNS:
        for match in root.glob(pattern):
            if match.is_file():
                rel = str(match.relative_to(root))
                results.append((rel, read_file_safe(match)))
    return results


def _append_file_result(
    results: list[tuple[str, str]],
    seen: set[str],
    rel: str,
    file_path: Path,
    max_bytes: Optional[int] = None,
) -> None:
    if rel in seen:
        return
    seen.add(rel)
    results.append((rel, read_file_safe(file_path, max_bytes)))


def _iter_selected_file_matches(root: Path, pattern: str):
    for file_path in root.rglob("*"):
        if not file_path.is_file():
            continue
        file_rel = str(file_path.relative_to(root).as_posix())
        if fnmatch.fnmatch(file_rel, pattern):
            yield file_rel, file_path


def collect_retrieval_candidates(root: Path, max_bytes: int = 4096) -> list[tuple[str, str]]:
    results: list[tuple[str, str]] = []
    exact_names = {
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
    relevant_tokens = (
        "readme",
        "install",
        "build",
        "package",
        "requirement",
        "docker",
        "workflow",
        "cargo",
        "gradle",
        "maven",
        "cmake",
        "makefile",
        "setup",
        "pipfile",
        "compose",
    )
    allowed_suffixes = {
        ".md",
        ".rst",
        ".txt",
        ".toml",
        ".json",
        ".yaml",
        ".yml",
        ".xml",
        ".cfg",
        ".ini",
        ".properties",
        ".gradle",
        ".kts",
        ".mk",
        ".sh",
    }

    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [dirname for dirname in dirnames if not should_skip_dir(dirname)]
        current_dir = Path(dirpath)

        for filename in sorted(filenames):
            file_path = current_dir / filename
            try:
                if file_path.stat().st_size > MAX_FILE_SIZE:
                    continue
            except OSError:
                continue

            rel = str(file_path.relative_to(root).as_posix())
            rel_lower = rel.lower()
            basename = file_path.name.lower()
            in_docs_dir = rel_lower.startswith("docs/") or "/docs/" in rel_lower

            include_candidate = False
            if basename in exact_names:
                include_candidate = True
            elif rel_lower.startswith(".github/workflows/"):
                include_candidate = True
            elif any(token in rel_lower for token in relevant_tokens):
                include_candidate = True
            elif file_path.suffix.lower() in allowed_suffixes and not in_docs_dir:
                include_candidate = True

            if in_docs_dir and not basename.startswith(("readme", "install")) and basename not in exact_names:
                include_candidate = False

            if not include_candidate:
                continue

            results.append((rel, read_file_safe(file_path, max_bytes=max_bytes)))

    return results


def collect_manifest_files(root: Path) -> list[tuple[str, str]]:
    """Walk repo and collect full-read manifest files (non-pattern)."""
    results: list[tuple[str, str]] = []
    seen: set[str] = set()

    wildcard_patterns = [pattern for pattern in FULL_READ_FILES if "*" in pattern]

    def _walk(path: Path):
        try:
            for entry in path.iterdir():
                if entry.is_dir():
                    if not should_skip_dir(entry.name):
                        _walk(entry)
                elif entry.is_file():
                    rel = str(entry.relative_to(root))
                    if rel in seen:
                        continue
                    if entry.name in FULL_READ_FILES:
                        _append_file_result(results, seen, rel, entry)
                    elif any(entry.match(pattern) for pattern in wildcard_patterns):
                        _append_file_result(results, seen, rel, entry)
                    elif entry.name in PARTIAL_READ:
                        _append_file_result(results, seen, rel, entry, PARTIAL_READ[entry.name])
        except PermissionError:
            pass

    _walk(root)
    # also check patterns
    for rel, content in match_glob_patterns(root):
        if rel not in seen:
            seen.add(rel)
            results.append((rel, content))

    return sorted(results, key=lambda x: x[0])


def learn_new_files(new_files: list[str]) -> dict:
    """
    Learn new files and patterns from LLM-selected files.
    Returns summary of what was added.
    """
    config_dir = Path(__file__).parent.parent / "config"
    manifest_path = config_dir / "manifest-files.yaml"
    
    if not manifest_path.exists():
        return {"added": 0, "details": "Config file not found"}
    
    with open(manifest_path, "r") as f:
        config = yaml.safe_load(f) or {}
    
    exact_matches = set(config.get("exact_matches", []))
    glob_patterns = set(config.get("glob_patterns", []))
    original_count = len(exact_matches) + len(glob_patterns)
    
    added_files = []
    added_patterns = []
    skipped_project_specific = []
    
    for file_path in new_files:
        normalized = file_path.strip()
        normalized_lower = normalized.lower()
        
        # Skip if already in either exact_matches or glob_patterns
        if normalized_lower in [x.lower() for x in exact_matches]:
            continue
        if any(fnmatch.fnmatch(normalized_lower, p.lower()) for p in glob_patterns):
            continue
        
        # Classify: is this a glob pattern or exact filename?
        is_glob = "*" in normalized or "?" in normalized or "[" in normalized

        # Keep global config clean: don't learn literal nested repo paths.
        # Example skipped: "src/foo/CMakeLists.txt".
        if "/" in normalized and not is_glob:
            skipped_project_specific.append(normalized)
            continue
        
        if is_glob:
            # It's a pattern (e.g., .github/workflows/*.yml)
            if normalized not in glob_patterns:
                added_patterns.append(normalized)
        else:
            # It's an exact filename or path
            if normalized not in exact_matches:
                added_files.append(normalized)
    
    # Update config
    if added_files:
        exact_matches.update(added_files)
        config["exact_matches"] = sorted(list(exact_matches))
    
    if added_patterns:
        glob_patterns.update(added_patterns)
        config["glob_patterns"] = sorted(list(glob_patterns))
    
    # Write back
    with open(manifest_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    
    total_added = len(added_files) + len(added_patterns)
    return {
        "added": total_added,
        "added_files": added_files,
        "added_patterns": added_patterns,
        "skipped_project_specific": skipped_project_specific,
        "total_in_config": original_count + total_added,
    }


def collect_selected_files(root: Path, selected_files: list[str]) -> list[tuple[str, str]]:
    """Collect only pre-selected files (safe relative paths under root)."""
    results: list[tuple[str, str]] = []
    seen: set[str] = set()

    for candidate in selected_files:
        rel = normalize_relative_path(candidate)
        if not rel or rel in seen:
            continue

        # Allow wildcard patterns from model output, e.g. .github/workflows/*.yml
        if "*" in rel or "?" in rel or "[" in rel:
            for file_rel, file_path in _iter_selected_file_matches(root, rel):
                _append_file_result(results, seen, file_rel, file_path)
            continue

        file_path = (root / rel).resolve()
        try:
            file_path.relative_to(root)
        except ValueError:
            continue

        if file_path.is_file():
            _append_file_result(results, seen, rel, file_path)

    return sorted(results, key=lambda x: x[0])


def _tokenize_text(value: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", value.lower())


def select_files_by_bm25(root: Path, query_terms: list[str], top_k: int = 12) -> list[str]:
    """Rank manifest/config files using a lightweight BM25 scorer."""
    candidates = collect_retrieval_candidates(root)
    if not candidates:
        return []

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

    normalized_terms = []
    seen_terms: set[str] = set()
    for term in query_terms:
        for token in _tokenize_text(term):
            if token and token not in seen_terms:
                seen_terms.add(token)
                normalized_terms.append(token)

    if not normalized_terms:
        return [rel for rel, _ in candidates[:top_k]]

    documents: list[tuple[str, list[str]]] = []
    document_frequency: dict[str, int] = {}
    total_length = 0

    for rel, content in candidates:
        tokens = _tokenize_text(f"{rel}\n{content}") or _tokenize_text(rel)
        documents.append((rel, tokens))
        total_length += len(tokens)
        for token in set(tokens):
            document_frequency[token] = document_frequency.get(token, 0) + 1

    avg_doc_length = total_length / len(documents) if documents else 0.0
    k1 = 1.5
    b = 0.75
    ranked: list[tuple[float, str]] = []

    for rel, tokens in documents:
        if not tokens:
            continue
        token_counts: dict[str, int] = {}
        for token in tokens:
            token_counts[token] = token_counts.get(token, 0) + 1

        doc_length = len(tokens)
        norm = k1 * (1 - b + b * (doc_length / avg_doc_length)) if avg_doc_length else k1
        score = 0.0

        for term in normalized_terms:
            term_frequency = token_counts.get(term, 0)
            if term_frequency <= 0:
                continue
            matching_docs = document_frequency.get(term, 0)
            inverse_doc_frequency = math.log(1 + ((len(documents) - matching_docs + 0.5) / (matching_docs + 0.5)))
            score += inverse_doc_frequency * ((term_frequency * (k1 + 1)) / (term_frequency + norm))

        rel_lower = rel.lower()
        basename = Path(rel_lower).name
        if basename in high_signal_names:
            score += 2.5
        elif basename.startswith(("readme", "install")):
            score += 1.5
        if rel_lower.startswith(".github/workflows/"):
            score += 0.25

        if score > 0:
            ranked.append((score, rel))

    if not ranked:
        return [rel for rel, _ in candidates[:top_k]]

    ranked.sort(key=lambda item: (-item[0], item[1]))
    return [rel for _, rel in ranked[:top_k]]


def select_files_by_bm25_budgeted(
    root: Path, query_terms: list[str], token_budget: int = 6000
) -> list[str]:
    """BM25-rank candidates then fill a token budget top-down.

    Uses character length / 4 as a conservative token estimate so the resulting
    fingerprint stays within model context limits regardless of repo size.
    """
    candidates = collect_retrieval_candidates(root)
    if not candidates:
        return []

    high_signal_names = {
        "readme.md", "readme.rst", "readme.txt", "install.md", "install.txt",
        "pyproject.toml", "requirements.txt", "package.json", "package-lock.json",
        "yarn.lock", "pnpm-lock.yaml", "go.mod", "cargo.toml", "dockerfile",
        "docker-compose.yml", "docker-compose.yaml", "makefile", "cmakelists.txt",
        "pom.xml", "build.gradle", "build.gradle.kts", "settings.gradle",
        "settings.gradle.kts", "gradlew", "gemfile", "composer.json",
        "setup.py", "setup.cfg", "pipfile", "pipfile.lock", ".env.example",
    }

    normalized_terms: list[str] = []
    seen_terms: set[str] = set()
    for term in query_terms:
        for token in _tokenize_text(term):
            if token and token not in seen_terms:
                seen_terms.add(token)
                normalized_terms.append(token)

    def _fill_budget(ordered: list[tuple[str, str]]) -> list[str]:
        selected: list[str] = []
        tokens_used = 0
        for rel, content in ordered:
            cost = max(1, len(content) // 4)
            if tokens_used + cost > token_budget and selected:
                break
            selected.append(rel)
            tokens_used += cost
        return selected

    if not normalized_terms:
        return _fill_budget(candidates)

    documents: list[tuple[str, list[str]]] = []
    content_by_rel: dict[str, str] = {}
    document_frequency: dict[str, int] = {}
    total_length = 0

    for rel, content in candidates:
        tokens = _tokenize_text(f"{rel}\n{content}") or _tokenize_text(rel)
        documents.append((rel, tokens))
        content_by_rel[rel] = content
        total_length += len(tokens)
        for token in set(tokens):
            document_frequency[token] = document_frequency.get(token, 0) + 1

    avg_doc_length = total_length / len(documents) if documents else 0.0
    k1 = 1.5
    b = 0.75
    ranked: list[tuple[float, str]] = []

    for rel, tokens in documents:
        if not tokens:
            continue
        token_counts: dict[str, int] = {}
        for token in tokens:
            token_counts[token] = token_counts.get(token, 0) + 1

        doc_length = len(tokens)
        norm = k1 * (1 - b + b * (doc_length / avg_doc_length)) if avg_doc_length else k1
        score = 0.0
        for term in normalized_terms:
            tf = token_counts.get(term, 0)
            if tf <= 0:
                continue
            n_docs = document_frequency.get(term, 0)
            idf = math.log(1 + ((len(documents) - n_docs + 0.5) / (n_docs + 0.5)))
            score += idf * ((tf * (k1 + 1)) / (tf + norm))

        rel_lower = rel.lower()
        basename = Path(rel_lower).name
        if basename in high_signal_names:
            score += 2.5
        elif basename.startswith(("readme", "install")):
            score += 1.5
        if rel_lower.startswith(".github/workflows/"):
            score += 0.25

        if score > 0:
            ranked.append((score, rel))

    if not ranked:
        return _fill_budget(candidates)

    ranked.sort(key=lambda item: (-item[0], item[1]))
    return _fill_budget([(rel, content_by_rel.get(rel, "")) for _, rel in ranked])


def collect_metadata(root: Path) -> dict:
    """Basic repo stats."""
    lang_ext_counts: dict[str, int] = {}
    total_files = 0
    total_lines = 0

    EXT_LANG = {
        ".py": "Python", ".js": "JavaScript", ".ts": "TypeScript",
        ".tsx": "TypeScript/React", ".jsx": "JavaScript/React",
        ".rs": "Rust", ".go": "Go", ".rb": "Ruby",
        ".java": "Java", ".kt": "Kotlin", ".scala": "Scala",
        ".cs": "C#", ".fs": "F#", ".vb": "VB.NET",
        ".cpp": "C++", ".c": "C", ".h": "C/C++ header",
        ".php": "PHP", ".swift": "Swift", ".m": "Objective-C",
        ".ex": "Elixir", ".exs": "Elixir", ".erl": "Erlang",
        ".hs": "Haskell", ".ml": "OCaml", ".clj": "Clojure",
        ".lua": "Lua", ".r": "R", ".jl": "Julia",
        ".dart": "Dart", ".vue": "Vue", ".svelte": "Svelte",
        ".tf": "Terraform", ".sh": "Shell", ".bash": "Bash",
        ".ps1": "PowerShell", ".sql": "SQL",
    }

    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if not should_skip_dir(d)]
        for fname in filenames:
            ext = Path(fname).suffix.lower()
            if ext in EXT_LANG:
                lang = EXT_LANG[ext]
                lang_ext_counts[lang] = lang_ext_counts.get(lang, 0) + 1
                total_files += 1
                fpath = Path(dirpath) / fname
                try:
                    if fpath.stat().st_size < MAX_FILE_SIZE:
                        with open(fpath, "r", encoding="utf-8", errors="replace") as f:
                            total_lines += sum(1 for _ in f)
                except Exception:
                    pass

    top_langs = sorted(lang_ext_counts.items(), key=lambda x: -x[1])[:8]
    return {
        "source_files": total_files,
        "approx_lines": total_lines,
        "languages": top_langs,
    }


# ---------------------------------------------------------------------------
# Formatters
# ---------------------------------------------------------------------------

def format_xml(
    root: Path,
    tree: list[str],
    manifests: list[tuple[str, str]],
    meta: dict,
    include_tree: bool = True,
) -> str:
    parts = ['<repo_fingerprint>']

    parts.append(f'  <root>{root.name}</root>')
    parts.append(f'  <source_files>{meta["source_files"]}</source_files>')
    parts.append(f'  <approx_lines>{meta["approx_lines"]}</approx_lines>')

    lang_str = ", ".join(f'{l} ({c})' for l, c in meta["languages"])
    parts.append(f'  <languages>{lang_str}</languages>')

    if include_tree:
        parts.append('  <directory_tree>')
        parts.append("\n".join("    " + l for l in tree))
        parts.append('  </directory_tree>')

    parts.append('  <manifest_files>')
    for rel, content in manifests:
        parts.append(f'    <file path="{rel}">')
        # indent content
        indented = "\n".join("      " + line for line in content.splitlines())
        parts.append(indented)
        parts.append('    </file>')
    parts.append('  </manifest_files>')

    parts.append('</repo_fingerprint>')
    return "\n".join(parts)


def format_markdown(
    root: Path,
    tree: list[str],
    manifests: list[tuple[str, str]],
    meta: dict,
    include_tree: bool = True,
) -> str:
    parts = [f"# Repo fingerprint: `{root.name}`\n"]

    parts.append("## Stats")
    parts.append(f"- Source files: {meta['source_files']}")
    parts.append(f"- Approx lines: {meta['approx_lines']}")
    lang_str = ", ".join(f"**{l}** ({c})" for l, c in meta["languages"])
    parts.append(f"- Languages: {lang_str}\n")

    if include_tree:
        parts.append("## Directory structure")
        parts.append("```")
        parts.extend(tree)
        parts.append("```\n")

    parts.append("## Manifest & config files")
    for rel, content in manifests:
        parts.append(f"### `{rel}`")
        ext = Path(rel).suffix.lstrip(".")
        parts.append(f"```{ext}")
        parts.append(content)
        parts.append("```\n")

    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def fingerprint(
    format: str,
    repo_path: str,
    structure_only: bool = False,
    selected_files: Optional[list[str]] = None,
    include_tree: bool = True,
    context: Optional[str] = None,
) -> str:
    root = Path(repo_path).resolve()
    if not root.is_dir():
        raise ValueError(f"Not a directory: {root}")

    ctx = f" [{context}]" if context else ""
    log_info(f"Scanning {root} ...{ctx}")
    log_trace(f"fingerprint{ctx}(format={format}, structure_only={structure_only}, include_tree={include_tree})")
    tree = build_tree(root) if include_tree else []
    manifests: list[tuple[str, str]]

    if structure_only:
        log_info(f"Structure-only mode: skipping file content collection{ctx}")
        manifests = []
    elif selected_files is not None:
        log_info(f"Collecting selected files ({len(selected_files)}) ...{ctx}")
        manifests = collect_selected_files(root, selected_files)
    else:
        log_info(f"Collecting manifest files ...{ctx}")
        manifests = collect_manifest_files(root)

    log_info(f"Collecting metadata ...{ctx}")
    meta = collect_metadata(root)

    log_info(
        f"Found {len(manifests)} manifest files, "
        f"{meta['source_files']} source files, "
        f"~{meta['approx_lines']:,} lines{ctx}"
    )

    if format == "md":
        return format_markdown(root, tree, manifests, meta, include_tree=include_tree)
    return format_xml(root, tree, manifests, meta, include_tree=include_tree)
