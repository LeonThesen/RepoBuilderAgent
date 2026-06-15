import subprocess
import tempfile
from pathlib import Path
from typing import Any, Callable

import yaml
from langchain_core.tools import tool

# Reserves headroom under the model's ~60k input cap for the system prompt + the
# model's own response. create_react_agent re-sends the FULL accumulated message
# history on every turn, so we trim the per-turn input down to this budget.
HISTORY_BUDGET = 48000


def tool_call_budget(recursion_limit: int) -> int:
    """Max tool calls an agent can make under a given LangGraph recursion_limit.

    Each tool call costs ~2 graph nodes (the model turn + the tool turn); we leave
    one node for the finalize call. Floored at 1 so prompts never advertise 0 calls.
    """
    return max(1, int(recursion_limit) // 2 - 1)


def _message_token_counter(model_name: str) -> "Callable[[list[Any]], int]":
    """Return a token_counter(list[messages]) -> int using tiktoken.

    Falls back to cl100k_base when the model name is unknown. Adds ~4 tokens of
    per-message overhead, which is the standard chat-format framing fudge factor.
    """
    import tiktoken

    try:
        encoding = tiktoken.encoding_for_model(model_name)
    except Exception:
        encoding = tiktoken.get_encoding("cl100k_base")

    def _count(messages: "list[Any]") -> int:
        total = 0
        for message in messages:
            content = getattr(message, "content", message)
            text = content if isinstance(content, str) else str(content)
            total += len(encoding.encode(text)) + 4
        return total

    return _count


def make_history_trim_hook(model_name: str, max_tokens: int) -> "Callable[[dict], dict]":
    """Build a create_react_agent pre_model_hook that caps accumulated history.

    create_react_agent re-sends the entire message history (every tool observation)
    on every model turn with no built-in trimming, so token usage grows unbounded
    over a ReAct loop. This hook keeps the system message plus the most-recent
    messages that fit under max_tokens and returns them via "llm_input_messages"
    (which feeds the model WITHOUT mutating the persisted state["messages"]).
    """
    from langchain_core.messages import trim_messages

    token_counter = _message_token_counter(model_name)

    def hook(state: dict) -> dict:
        messages = (state or {}).get("messages") or []
        if not messages:
            return {"llm_input_messages": messages}
        try:
            trimmed = trim_messages(
                messages,
                strategy="last",
                token_counter=token_counter,
                max_tokens=max_tokens,
                include_system=True,
                start_on="human",
                allow_partial=False,
            )
            if trimmed:
                return {"llm_input_messages": trimmed}
        except Exception:
            pass
        # Manual fallback: keep the leading system message (if any) + as many of the
        # most-recent messages as fit under the budget.
        from langchain_core.messages import SystemMessage

        system_msgs = [m for m in messages if isinstance(m, SystemMessage)]
        head = system_msgs[:1]
        base_cost = token_counter(head) if head else 0
        kept_tail: list[Any] = []
        running = base_cost
        for message in reversed([m for m in messages if m not in head]):
            cost = token_counter([message])
            if running + cost > max_tokens and kept_tail:
                break
            kept_tail.append(message)
            running += cost
        kept_tail.reverse()
        result = head + kept_tail
        if not result:
            result = messages[-1:]
        return {"llm_input_messages": result}

    return hook

_SKIP_DIRS = {
    ".git", "node_modules", "__pycache__", ".venv", "venv", "env",
    ".tox", "dist", "build", "target", ".next", ".nuxt", "out",
    ".gradle", ".idea", ".vscode", "coverage", ".nyc_output",
    "vendor", "third_party", ".terraform", ".serverless", ".eggs",
}
_INCLUDE_HIDDEN = {".github", ".circleci", ".devcontainer"}


def build_read_file_tool(repo_path: Path) -> Callable:
    _root = repo_path.resolve()

    @tool
    def read_file(path: str, max_chars: int = 3000) -> str:
        """Read a repository file by its repo-relative path and return its content.

        path must be a repo-relative path (e.g. 'src/setup.py', 'Dockerfile').
        Content is truncated at max_chars (default 3000).
        """
        rel = (path or "").strip().lstrip("./")
        if not rel:
            return yaml.dump({"error": "empty_path"}, sort_keys=False)
        resolved = (_root / rel).resolve()
        try:
            resolved.relative_to(_root)
        except ValueError:
            return yaml.dump({"error": "path_outside_repo"}, sort_keys=False)
        if not resolved.exists():
            return yaml.dump({"error": f"not_found: {rel}"}, sort_keys=False)
        if not resolved.is_file():
            return yaml.dump({"error": f"not_a_file: {rel}"}, sort_keys=False)
        try:
            content = resolved.read_text(encoding="utf-8", errors="replace")
        except Exception as exc:
            return yaml.dump({"error": str(exc)}, sort_keys=False)
        cap = max(300, int(max_chars))
        if len(content) > cap:
            content = content[:cap] + "\n... [truncated]"
        return yaml.dump({"path": rel, "content": content}, sort_keys=False, allow_unicode=True)

    return read_file


def build_list_tree_tool(repo_path: Path) -> Callable:
    _root = repo_path.resolve()

    @tool
    def list_tree(path: str = "", depth: int = 2) -> str:
        """List directory contents (files and subdirectories) in the repository.

        path is a repo-relative directory path; empty string or '.' means the root.
        depth controls how many levels to recurse (1 = immediate children only, max 3).
        Each line is prefixed with 'f:' for files or 'd:' for directories.
        """
        rel = (path or "").strip().lstrip("./")
        target = (_root / rel).resolve() if rel else _root
        try:
            target.relative_to(_root)
        except ValueError:
            return yaml.dump({"error": "path_outside_repo"}, sort_keys=False)
        if not target.exists():
            return yaml.dump({"error": f"not_found: {rel or '.'}"}, sort_keys=False)
        if not target.is_dir():
            return yaml.dump({"error": f"not_a_directory: {rel or '.'}"}, sort_keys=False)

        max_depth = min(max(1, int(depth)), 3)
        lines: list[str] = []

        def _walk(current: Path, cur_depth: int) -> None:
            if cur_depth > max_depth:
                return
            try:
                entries = sorted(current.iterdir(), key=lambda e: (e.is_file(), e.name.lower()))
            except PermissionError:
                return
            for entry in entries:
                if entry.is_dir():
                    if entry.name not in _INCLUDE_HIDDEN and (entry.name in _SKIP_DIRS or entry.name.startswith(".")):
                        continue
                entry_rel = str(entry.relative_to(_root))
                lines.append(("f: " if entry.is_file() else "d: ") + entry_rel)
                if entry.is_dir() and cur_depth < max_depth:
                    _walk(entry, cur_depth + 1)
                if len(lines) >= 300:
                    return

        _walk(target, 1)
        if not lines:
            return "(empty directory)"
        if len(lines) == 300:
            lines.append("... [truncated at 300 entries]")
        return "\n".join(lines)

    return list_tree


def build_search_pattern_tool(repo_path: Path) -> Callable:
    _root = repo_path.resolve()

    @tool
    def search_pattern(pattern: str, limit: int = 50) -> str:
        """Find files matching a glob pattern relative to the repo root.

        Examples: '**/*.toml', 'src/**/*.py', 'Dockerfile*', '.github/workflows/*.yml'
        Returns repo-relative paths of matching files, capped at limit (max 50).
        """
        pat = (pattern or "").strip()
        if not pat:
            return yaml.dump({"error": "empty_pattern"}, sort_keys=False)
        cap = min(max(1, int(limit)), 50)
        matches: list[str] = []
        try:
            for match in _root.glob(pat):
                if match.is_file():
                    matches.append(str(match.relative_to(_root)))
                if len(matches) >= cap:
                    break
        except Exception as exc:
            return yaml.dump({"error": str(exc)}, sort_keys=False)
        return yaml.dump(
            {"matches": sorted(matches), "count": len(matches)},
            sort_keys=False,
            allow_unicode=True,
        )

    return search_pattern


def build_read_gitlog_tool(repo_path: Path) -> Callable:
    _root = repo_path.resolve()

    @tool
    def read_gitlog(n: int = 20) -> str:
        """Read the most recent n git commit messages (hash + subject line).

        Useful for discovering when build files were added or changed.
        n is capped at 50.
        """
        cap = min(max(1, int(n)), 50)
        try:
            result = subprocess.run(
                ["git", "log", f"--max-count={cap}", "--oneline"],
                cwd=str(_root),
                capture_output=True,
                text=True,
                timeout=10,
                check=False,
            )
            if result.returncode != 0:
                return yaml.dump({"error": result.stderr.strip() or "git_log_failed"}, sort_keys=False)
            lines = result.stdout.strip().splitlines()
            return yaml.dump({"commits": lines, "count": len(lines)}, sort_keys=False, allow_unicode=True)
        except FileNotFoundError:
            return yaml.dump({"error": "git_not_found"}, sort_keys=False)
        except subprocess.TimeoutExpired:
            return yaml.dump({"error": "timeout"}, sort_keys=False)
        except Exception as exc:
            return yaml.dump({"error": str(exc)}, sort_keys=False)

    return read_gitlog


def build_search_commits_tool(repo_path: Path) -> Callable:
    _root = repo_path.resolve()

    @tool
    def search_commits(keyword: str, n: int = 20) -> str:
        """Search git commit messages for a keyword and return matching commits.

        Examples: 'dockerfile', 'requirements', 'install', 'build', 'python'
        n is capped at 50. Search is case-insensitive.
        """
        kw = (keyword or "").strip()
        if not kw:
            return yaml.dump({"error": "empty_keyword"}, sort_keys=False)
        cap = min(max(1, int(n)), 50)
        try:
            result = subprocess.run(
                ["git", "log", f"--grep={kw}", f"--max-count={cap}", "--oneline", "--regexp-ignore-case"],
                cwd=str(_root),
                capture_output=True,
                text=True,
                timeout=10,
                check=False,
            )
            if result.returncode != 0:
                return yaml.dump({"error": result.stderr.strip() or "git_log_failed"}, sort_keys=False)
            lines = result.stdout.strip().splitlines()
            return yaml.dump({"keyword": kw, "matches": lines, "count": len(lines)}, sort_keys=False, allow_unicode=True)
        except FileNotFoundError:
            return yaml.dump({"error": "git_not_found"}, sort_keys=False)
        except subprocess.TimeoutExpired:
            return yaml.dump({"error": "timeout"}, sort_keys=False)
        except Exception as exc:
            return yaml.dump({"error": str(exc)}, sort_keys=False)

    return search_commits


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


def build_finalize_tool() -> Callable:
    @tool
    def finalize(answer: str) -> str:
        """Submit your FINAL answer as YAML and end the task. Call this exactly once,
        as your LAST action, when you have enough evidence OR when you are near your
        tool-call budget. Do not call any other tool after finalize."""
        return "finalized"
    return finalize


def extract_finalize_answer(messages) -> "str | None":
    """Scan messages in reverse for an AIMessage whose tool_calls contains a call
    named 'finalize', and return its args['answer'] string.

    Handles tool_call being a dict (LangGraph standard) or an object with .name/.args,
    and args being a dict or an object exposing .answer / ['answer'].
    Returns None when no finalize call with a usable answer is found.
    """
    for message in reversed(list(messages or [])):
        tool_calls = getattr(message, "tool_calls", None)
        if not tool_calls:
            continue
        for tc in tool_calls:
            if isinstance(tc, dict):
                name = tc.get("name")
                tc_args = tc.get("args")
            else:
                name = getattr(tc, "name", None)
                tc_args = getattr(tc, "args", None)
            if name != "finalize":
                continue
            answer = None
            if isinstance(tc_args, dict):
                answer = tc_args.get("answer")
            elif tc_args is not None:
                answer = getattr(tc_args, "answer", None)
            if isinstance(answer, str) and answer.strip():
                return answer
    return None


# LangGraph's recursion-limit placeholder. Depending on version/config it either
# RETURNS this as the last message OR raises GraphRecursionError. We normalize both
# to the same returned result (see ainvoke_with_recursion_guard) so every call site
# detects the limit uniformly via hit_step_limit() instead of crashing the stage.
RECURSION_LIMIT_PLACEHOLDER = "Sorry, need more steps to process this request."

try:  # pragma: no cover - import shape varies across langgraph versions
    from langgraph.errors import GraphRecursionError
except Exception:  # pragma: no cover
    GraphRecursionError = RecursionError


def hit_step_limit(result) -> bool:
    """True if the ReAct loop ended on LangGraph's recursion-limit placeholder."""
    messages = (result or {}).get("messages") or [] if isinstance(result, dict) else []
    if not messages:
        return False
    content = getattr(messages[-1], "content", "")
    text = content if isinstance(content, str) else str(content)
    return text.strip() == RECURSION_LIMIT_PLACEHOLDER


async def ainvoke_with_recursion_guard(agent, payload, config):
    """Invoke a create_react_agent, converting a *raised* GraphRecursionError into
    the same placeholder result LangGraph emits when it instead *returns* at the
    limit. This makes recursion-limit handling uniform across every ReAct loop:
    callers detect it with hit_step_limit() / get an empty payload and fall back,
    rather than letting the exception escape and crash the whole stage.

    Previously only L1 wrapped its agent in try/except; the generator, reviewer,
    validation and L3 loops relied on a non-raising placeholder this LangGraph
    version does not produce, so they crashed on repos that exhausted the budget.
    """
    from langchain_core.messages import AIMessage

    try:
        return await agent.ainvoke(payload, config=config)
    except GraphRecursionError:
        return {"messages": [AIMessage(content=RECURSION_LIMIT_PLACEHOLDER)]}


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


def build_get_dockerfile_snippet_tool() -> Callable:
    try:
        from RepoBuilderAgent.src.agent_tools.dockerfile_snippets import get_snippet, list_actions as _list_actions
    except ImportError:
        from agent_tools.dockerfile_snippets import get_snippet, list_actions as _list_actions

    @tool
    def get_dockerfile_snippet(action: str, version: str = "") -> str:
        """Return a validated Dockerfile RUN-block snippet for a common build toolchain.

        Call with action='list_actions' to see all available actions and descriptions.

        Common actions (version is optional):
          install_jdk(version)       — OpenJDK JDK from apt (default: 17)
          install_jre(version)       — OpenJDK JRE from apt (default: 17)
          install_node(version)      — Node.js via NodeSource (default: 20)
          install_cargo              — Rust + Cargo via rustup
          install_go(version)        — Go from go.dev tarball (default: 1.22)
          install_ruby               — Ruby from apt
          install_cmake              — CMake + build-essential
          install_maven              — Apache Maven from apt
          install_gradle(version)    — Gradle distribution (default: 8.5)
          install_build_essential    — GCC, Make, pkg-config
          install_elixir             — Elixir + Erlang/OTP
          install_dotnet(version)    — .NET SDK (default: 8)
          install_php                — PHP-CLI + Composer
          install_pip_requirements   — pip install -r requirements.txt
          install_npm_ci             — npm ci
          install_yarn_frozen        — yarn install --frozen-lockfile
          install_poetry             — Poetry install
          install_sbt                — sbt Scala build tool
        """
        return get_snippet((action or "").strip(), (version or "").strip())

    return get_dockerfile_snippet


def build_hadolint_snippet_tool() -> Callable[[str], str]:
    @tool
    def run_hadolint_on_snippet(dockerfile_text: str) -> str:
        """Validate a Dockerfile snippet with hadolint and return pass/fail details."""
        text = (dockerfile_text or "").strip()
        if not text:
            return yaml.dump(
                {"available": True, "valid": False, "error": "empty_dockerfile_text"},
                sort_keys=False,
                allow_unicode=True,
            )

        hadolint_path = Path("/usr/local/bin/hadolint")
        if not hadolint_path.exists():
            return yaml.dump(
                {
                    "available": False,
                    "valid": True,
                    "error": "hadolint_not_installed",
                },
                sort_keys=False,
                allow_unicode=True,
            )

        with tempfile.NamedTemporaryFile(mode="w", suffix=".Dockerfile", delete=True) as temp_file:
            temp_file.write(text)
            temp_file.flush()

            command = [str(hadolint_path)]
            repo_root = Path(__file__).resolve().parents[3]
            hadolint_config = repo_root / ".hadolint.yaml"
            if hadolint_config.exists():
                command.extend(["--config", str(hadolint_config)])
            command.append(temp_file.name)

            try:
                result = subprocess.run(
                    command,
                    capture_output=True,
                    text=True,
                    timeout=10,
                    check=False,
                )
            except subprocess.TimeoutExpired:
                return yaml.dump(
                    {
                        "available": True,
                        "valid": False,
                        "error": "hadolint_timeout",
                    },
                    sort_keys=False,
                    allow_unicode=True,
                )

        output = (result.stdout or "") + (result.stderr or "")
        return yaml.dump(
            {
                "available": True,
                "valid": result.returncode == 0,
                "error_excerpt": output[:1200],
            },
            sort_keys=False,
            allow_unicode=True,
        )

    return run_hadolint_on_snippet
