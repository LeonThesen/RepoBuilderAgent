from typing import Callable

import yaml
from langchain_core.tools import tool


def build_search_structure_paths_tool(structure_summary: str) -> Callable[[str, int], str]:
    @tool
    def search_structure_paths(keyword: str, limit: int = 8) -> str:
        """Search structure summary lines by keyword and return matching file paths."""
        key = (keyword or "").strip().lower()
        if not key:
            return "[]"
        matches: list[str] = []
        for line in structure_summary.splitlines():
            candidate = line.strip().lstrip("- ").strip()
            if not candidate or "/" not in candidate:
                continue
            if key in candidate.lower():
                matches.append(candidate)
            if len(matches) >= max(1, int(limit)):
                break
        return yaml.dump({"matches": matches}, sort_keys=False, allow_unicode=True)

    return search_structure_paths


def build_select_default_files_tool(default_selected_files: list[str]) -> Callable[[], str]:
    @tool
    def select_default_files() -> str:
        """Return fallback default selected files used by the classifier."""
        return yaml.dump({"default_selected_files": default_selected_files}, sort_keys=False, allow_unicode=True)

    return select_default_files


def build_fetch_file_context_tool(file_context_by_path: dict[str, str]) -> Callable[[str, int], str]:
    @tool
    def fetch_file_context(path: str, max_chars: int = 1200) -> str:
        """Fetch truncated file context for a repository-relative path."""
        normalized = (path or "").strip().lstrip("./")
        if not normalized:
            return ""
        content = file_context_by_path.get(normalized, "")
        if len(content) > max(300, int(max_chars)):
            content = content[: max(300, int(max_chars))] + "\n... [truncated]"
        return content

    return fetch_file_context


def build_list_selected_files_tool(selected_files: list[str]) -> Callable[[], str]:
    @tool
    def list_selected_files() -> str:
        """Return all currently selected repository-relative files for this loop."""
        return yaml.dump({"selected_files": selected_files}, sort_keys=False, allow_unicode=True)

    return list_selected_files


def build_search_selected_files_tool(selected_files: list[str]) -> Callable[[str, int], str]:
    @tool
    def search_selected_files(keyword: str, limit: int = 8) -> str:
        """Search selected file paths by keyword and return matching paths."""
        key = (keyword or "").strip().lower()
        if not key:
            return "[]"
        matches = [path for path in selected_files if key in path.lower()][: max(1, int(limit))]
        return yaml.dump({"matches": matches}, sort_keys=False, allow_unicode=True)

    return search_selected_files


def build_think_tool() -> Callable[[str], str]:
    @tool
    def think(note: str) -> str:
        """Record a brief reasoning note before deciding next tool/action."""
        text = (note or "").strip()
        if not text:
            return yaml.dump({"accepted": False, "message": "empty_think_note"}, sort_keys=False, allow_unicode=True)
        return yaml.dump(
            {
                "accepted": True,
                "note": text[:1200],
            },
            sort_keys=False,
            allow_unicode=True,
        )

    return think
