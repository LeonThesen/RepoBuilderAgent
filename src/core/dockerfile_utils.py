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
    # Accept any fence language tag (```dockerfile, ```bash, bare ```), not just
    # "dockerfile" — a mis-tagged fence otherwise leaks its opening ``` line into the
    # written Dockerfile and buildkit rejects it ("can't find = in #").
    match = re.search(r"```[a-zA-Z]*\n(.*?)```", raw, re.DOTALL | re.IGNORECASE)
    content = match.group(1) if match else raw
    return _strip_trailing_noise(content).strip() + "\n"


def has_from_instruction(dockerfile_content: str) -> bool:
    """True if the Dockerfile has at least one ``FROM`` instruction (a real build
    stage). A Dockerfile without one fails to build with "no build stage in current
    context" — the symptom when a model emits only the AGENT_BUILD_STEPS body and
    drops the base template verbatim it was told to keep."""
    return any(
        line.strip().upper().startswith("FROM ")
        for line in dockerfile_content.splitlines()
    )


def ensure_base_template(
    dockerfile_content: str,
    base_template: str,
    *,
    log_warn: Callable[[str], None] | None = None,
) -> str:
    """Self-heal a generated Dockerfile that lost its base stage.

    The generation contract is "reproduce the base template verbatim, fill only the
    AGENT_BUILD_STEPS region". When a model instead returns just the region (no
    ``FROM``), the build dies with "no build stage in current context". If the
    output still carries the two markers we can reconstruct a valid file: splice the
    model's build-steps region into the base template, preserving everything else.

    Returns the content unchanged when it already has a ``FROM`` (the normal path),
    and returns it unchanged (un-healable) when no markers are present to splice.
    """
    try:
        from RepoBuilderAgent.src.core.build_spec import BUILD_START_MARKER, BUILD_END_MARKER
    except ImportError:
        from core.build_spec import BUILD_START_MARKER, BUILD_END_MARKER

    if has_from_instruction(dockerfile_content):
        return dockerfile_content

    gen_start = dockerfile_content.find(BUILD_START_MARKER)
    gen_end = dockerfile_content.find(BUILD_END_MARKER)
    base_start = base_template.find(BUILD_START_MARKER)
    base_end = base_template.find(BUILD_END_MARKER)
    # The base template must have both markers to define the splice slot; the model
    # output only needs a START marker — if it truncated before the END marker, treat
    # everything from START to end-of-output as the region and re-append END.
    if gen_start < 0 or base_start < 0 or base_end < 0 or not base_template.strip():
        if log_warn:
            log_warn(
                "Generated Dockerfile has no FROM and cannot be healed (markers or "
                "base template missing); leaving as-is — the build will surface the error."
            )
        return dockerfile_content

    # The model's region, markers inclusive, dropped into the base template's slot.
    if gen_end < 0:
        region = dockerfile_content[gen_start:].rstrip() + "\n" + BUILD_END_MARKER
    else:
        region = dockerfile_content[gen_start : gen_end + len(BUILD_END_MARKER)]
    healed = base_template[:base_start] + region + base_template[base_end + len(BUILD_END_MARKER) :]
    if log_warn:
        log_warn(
            "Generated Dockerfile dropped the base template (no FROM); re-spliced the "
            "AGENT_BUILD_STEPS region into the base template to restore a valid build stage."
        )
    return healed.strip() + "\n"


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
