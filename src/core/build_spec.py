"""Deterministic extraction of a parsable build/verify spec from a generated
Dockerfile (TODO 4).

The agent emits a free-form Dockerfile, but the base template brackets the
repo-specific build commands with stable section-comment markers:

    # Build instructions
    RUN <build command>
    # Final CMD or ENTRYPOINT

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

BUILD_START_MARKER = "# Build instructions"
# Any of these (exact stripped comment, or an instruction keyword) ends the section.
BUILD_END_MARKERS = ("# Final CMD or ENTRYPOINT",)
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


def extract_build_steps(dockerfile_text: str) -> tuple[list[str], bool]:
    """Return (build_steps, markers_found) parsed from the marked build section."""
    lines = dockerfile_text.splitlines()
    start = None
    for idx, line in enumerate(lines):
        if line.strip() == BUILD_START_MARKER:
            start = idx + 1
            break
    if start is None:
        return [], False

    steps: list[str] = []
    i = start
    while i < len(lines):
        stripped = lines[i].strip()
        if stripped in BUILD_END_MARKERS:
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
