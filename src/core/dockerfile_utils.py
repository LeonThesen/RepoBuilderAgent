from __future__ import annotations

import re
from pathlib import Path
from typing import Callable


_DOCKERFILE_DIRECTIVES = {
    "FROM", "RUN", "CMD", "LABEL", "MAINTAINER", "EXPOSE", "ENV", "ADD", "COPY",
    "ENTRYPOINT", "VOLUME", "USER", "WORKDIR", "ARG", "ONBUILD", "STOPSIGNAL",
    "HEALTHCHECK", "SHELL",
}


def _strip_trailing_noise(content: str) -> str:
    """Drop trailing lines that are not part of the Dockerfile.

    A ReAct repair agent sometimes appends its finalize payload (e.g.
    ``done: true`` / ``stop_reason: "..."``) after the Dockerfile body when it
    answers without a code fence. Those lines are not Dockerfile syntax and break
    the build with a parse error, so truncate everything after the last real
    instruction (or its ``\\``-continuation). Comments and blanks between
    instructions are preserved; only the trailing non-Dockerfile block is removed.
    """
    lines = content.split("\n")
    last_valid = -1
    in_continuation = False
    for idx, line in enumerate(lines):
        stripped = line.strip()
        if in_continuation:
            last_valid = idx
            in_continuation = stripped.endswith("\\")
            continue
        if stripped == "" or stripped.startswith("#"):
            continue
        directive = stripped.split(None, 1)[0].upper()
        if directive in _DOCKERFILE_DIRECTIVES:
            last_valid = idx
            in_continuation = stripped.endswith("\\")
        else:
            in_continuation = False
    if last_valid < 0:
        return content
    return "\n".join(lines[: last_valid + 1])


def extract_base_image(dockerfile_content: str) -> str:
    """Return the image ref from the first ``FROM`` instruction (dropping any
    ``AS <stage>`` alias and ``--platform=`` flag), or ``""`` if none is found.
    Used to point the apt-search repair tool at the exact base the build uses."""
    for line in dockerfile_content.splitlines():
        stripped = line.strip()
        if stripped.upper().startswith("FROM "):
            tokens = [t for t in stripped.split()[1:] if not t.startswith("--")]
            return tokens[0] if tokens else ""
    return ""


def extract_dockerfile(raw: str) -> str:
    match = re.search(r"```(?:dockerfile)?\n(.*?)```", raw, re.DOTALL | re.IGNORECASE)
    content = match.group(1) if match else raw
    return _strip_trailing_noise(content).strip() + "\n"


def get_base_template(
    classification: dict,
    templates_dir: Path,
    *,
    log_warn: Callable[[str], None],
    log_error: Callable[[str], None],
) -> str:
    
    template_name = "Dockerfile.base"

    template_path = templates_dir / template_name
    if template_path.exists():
        with open(template_path, "r", encoding="utf-8") as template_file:
            return template_file.read()

    log_error(f"Base template not found at {template_path}")
    return ""
