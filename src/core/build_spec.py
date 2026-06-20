"""Deterministic extraction of a parsable build/verify spec from a generated
Dockerfile (TODO 4).

The agent emits a free-form Dockerfile, but the base template brackets the
repo-specific build commands with explicit sentinel markers the generation and
repair prompts are instructed to keep verbatim:

    # AGENT_BUILD_STEPS_BEGIN
    RUN <build command>
    # AGENT_BUILD_STEPS_END

This module slices out the marked build section and strips the ``RUN`` prefix so
the agent's build_steps and verification command land in the same flat, parsable
shape as the ground-truth dataset YAML (``categories.build_steps.value`` and
``categories.verification.value``). That makes generated-vs-ground-truth
comparison (TODO 28) a straight list diff.

Extraction is purely deterministic: given the markers, the same Dockerfile always
yields the same spec. When the markers are absent (e.g. a repair rewrite dropped
them) ``build_steps`` is empty and ``markers_found`` is False so callers can flag it.
"""

from __future__ import annotations

# Explicit sentinels emitted by the base template + enforced by the prompts.
BUILD_START_MARKER = "# AGENT_BUILD_STEPS_BEGIN"
BUILD_END_MARKER = "# AGENT_BUILD_STEPS_END"

# Legacy fallback for Dockerfiles generated before the sentinels existed, or when a
# repair rewrite dropped them: the human section comments still bracket the build.
LEGACY_START_MARKER = "# Build instructions"
LEGACY_END_MARKERS = ("# Final CMD or ENTRYPOINT",)
_INSTRUCTION_TERMINATORS = ("CMD ", "ENTRYPOINT ", "CMD[", "ENTRYPOINT[")


def _join_run_command(first_line: str, lines: list[str], index: int) -> tuple[str, int]:
    """Join a RUN instruction that may span multiple ``\\``-continued lines.

    Returns the single-line command (without the ``RUN`` prefix) and the index of
    the last consumed line.
    """
    body = first_line[len("RUN "):]
    parts: list[str] = []
    current = body
    i = index
    while current.rstrip().endswith("\\"):
        parts.append(current.rstrip()[:-1].strip())
        i += 1
        if i >= len(lines):
            break
        current = lines[i]
    parts.append(current.strip())
    command = " ".join(part for part in parts if part)
    return " ".join(command.split()), i


def _section_bounds(lines: list[str]) -> tuple[int, tuple[str, ...]] | None:
    """Locate the build section. Prefer the explicit sentinels; fall back to the
    legacy section comments. Returns (start_index, end_markers) or None."""
    for idx, line in enumerate(lines):
        if line.strip() == BUILD_START_MARKER:
            return idx + 1, (BUILD_END_MARKER,)
    for idx, line in enumerate(lines):
        if line.strip() == LEGACY_START_MARKER:
            return idx + 1, LEGACY_END_MARKERS
    return None


def extract_build_steps(dockerfile_text: str) -> tuple[list[str], bool]:
    """Return (build_steps, markers_found) parsed from the marked build section."""
    lines = dockerfile_text.splitlines()
    bounds = _section_bounds(lines)
    if bounds is None:
        return [], False
    start, end_markers = bounds

    steps: list[str] = []
    i = start
    while i < len(lines):
        stripped = lines[i].strip()
        if stripped in end_markers:
            break
        if stripped.startswith(_INSTRUCTION_TERMINATORS):
            break
        if stripped.startswith("RUN "):
            command, i = _join_run_command(stripped, lines, i)
            if command:
                steps.append(command)
        i += 1
    return steps, True


def extract_build_spec(dockerfile_text: str, verify_command: str | None) -> dict:
    """Build the parsable spec: build_steps from the Dockerfile, verification from
    the generated verify command. Mirrors the ground-truth field shape."""
    steps, markers_found = extract_build_steps(dockerfile_text)
    verify = (verify_command or "").strip()
    return {
        "build_steps": steps,
        "verification": [verify] if verify else [],
        "markers_found": markers_found,
    }
