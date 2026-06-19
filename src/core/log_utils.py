import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

try:
    from tqdm import tqdm
except Exception:
    tqdm = None


_TRACE_ENABLED = False
_ACTIVE_TQDM_BAR: Any = None
_DUMP_PROMPTS_DIR: Path | None = None
_DUMP_CALL_COUNTS: defaultdict = defaultdict(int)

# ANSI color codes
_GREEN = "\033[92m"
_RED = "\033[91m"
_YELLOW = "\033[93m"
_CYAN = "\033[96m"
_MAGENTA = "\033[95m"
_BOLD = "\033[1m"
_DIM = "\033[2m"
_RESET = "\033[0m"


def set_tqdm_bar(bar: Any) -> None:
    global _ACTIVE_TQDM_BAR
    _ACTIVE_TQDM_BAR = bar


def _emit(msg: str) -> None:
    if _ACTIVE_TQDM_BAR is not None:
        _ACTIVE_TQDM_BAR.write(msg)
    elif tqdm is not None:
        tqdm.write(msg)
    else:
        print(msg, file=sys.stderr)


def set_trace_enabled(enabled: bool) -> None:
    global _TRACE_ENABLED
    _TRACE_ENABLED = enabled


def set_dump_prompts_dir(path: str | Path) -> None:
    global _DUMP_PROMPTS_DIR, _DUMP_CALL_COUNTS
    _DUMP_PROMPTS_DIR = Path(path)
    _DUMP_CALL_COUNTS = defaultdict(int)


def dump_prompt(repo_url: str, phase: str, messages: list[dict]) -> None:
    if _DUMP_PROMPTS_DIR is None:
        return
    repo_name = repo_url.rstrip("/").split("/")[-1].replace(".git", "") if repo_url else "unknown"
    key = (repo_name, phase)
    _DUMP_CALL_COUNTS[key] += 1
    n = _DUMP_CALL_COUNTS[key]
    out_dir = _DUMP_PROMPTS_DIR / repo_name
    out_dir.mkdir(parents=True, exist_ok=True)
    lines: list[str] = [f"# repo: {repo_name}", f"# phase: {phase}", f"# call: {n}", ""]
    for msg in messages:
        role = msg.get("role", "unknown")
        content = msg.get("content", "")
        lines.append(f"[{role}]")
        lines.append(content)
        lines.append("")
    (out_dir / f"{phase}.{n}.txt").write_text("\n".join(lines), encoding="utf-8")


def prompt_dump_enabled() -> bool:
    """True when a prompt-dump directory has been configured for this run."""
    return _DUMP_PROMPTS_DIR is not None


def dump_metadata(repo_url: str, phase: str) -> dict:
    """LangChain run metadata that carries repo + phase to the prompt-dump
    callback. Threaded through each ReAct invocation's config so every LLM turn
    (not just the seed prompt) is attributed and dumped. Keys are namespaced to
    avoid colliding with other metadata."""
    return {"dump_repo": repo_url, "dump_phase": phase}


def log_info(msg: str) -> None:
    _emit(f"{_CYAN}[*]{_RESET} {msg}")


def log_warn(msg: str) -> None:
    _emit(f"{_YELLOW}{_BOLD}[!]{_RESET} {msg}")


def log_error(msg: str) -> None:
    _emit(f"{_RED}{_BOLD}[x]{_RESET} {msg}")


def log_trace(msg: str) -> None:
    if _TRACE_ENABLED:
        _emit(f"{_MAGENTA}[.]{_RESET} {_DIM}{msg}{_RESET}")


def log_file_delta(repo_name: str, baseline_files: list[str], selected_files: list[str]) -> None:
    """Log file set differences between baseline and LLM-selected files (trace only)."""
    if not _TRACE_ENABLED:
        return
    
    baseline_set = set(baseline_files)
    selected_set = set(selected_files)
    
    common = baseline_set & selected_set
    only_baseline = baseline_set - selected_set
    only_selected = selected_set - baseline_set
    
    _emit(f"[.] File set delta for {repo_name}:")
    _emit(f"[.]   {_GREEN}Common ({len(common)}){_RESET}: {', '.join(sorted(common)[:5])}{'...' if len(common) > 5 else ''}")
    if only_baseline:
        _emit(f"[.]   {_RED}Only in baseline ({len(only_baseline)}){_RESET}: {', '.join(sorted(only_baseline)[:5])}{'...' if len(only_baseline) > 5 else ''}")
    if only_selected:
        _emit(f"[.]   {_YELLOW}Only in LLM-selected ({len(only_selected)}){_RESET}: {', '.join(sorted(only_selected)[:5])}{'...' if len(only_selected) > 5 else ''}")