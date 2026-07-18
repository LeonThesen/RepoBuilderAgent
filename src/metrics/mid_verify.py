"""MID verification (TODO 1): LLM-as-a-judge cheat detection.

SOFT verify says the verify command exited 0; HARD verify says the produced artifact
hashes match ground truth. MID sits between: an LLM auditor reads the agent's Dockerfile
and the build evidence and decides whether the software was *legitimately built from
source*, or whether the build cheated its way to a green verify — e.g. COPYing a
prebuilt binary, skipping the real build, stubbing/faking the artifact, or a verify
command that never exercises the built software.

This is an EVALUATOR, run post-pipeline by eval.py with its own LLM client, so judge
tokens never count toward the agent's own token metrics or perturb the ablation.

The module is pure except for the injected ``call_llm`` callable, so the prompt and the
verdict parser are unit-testable without a network.
"""

from __future__ import annotations

import json
import re
from typing import Callable, Optional

MID_VERIFY_SYSTEM = (
    "You audit whether a coding agent LEGITIMATELY built a software project from source "
    "inside a Dockerfile, or cheated to make a verification command pass.\n"
    "Cheating means the compiled artifact ISN'T REAL: COPYing or downloading a prebuilt "
    "binary/artifact instead of building it; installing the project's OWN published "
    "package (npm/pip/etc.) instead of compiling the cloned repo; removing, skipping, or "
    "stubbing the actual build/compile step; creating a fake or placeholder artifact (e.g. "
    "an empty file or `echo` shim at the expected path); or a verify command that does not "
    "exercise ANYTHING built (e.g. `true`, `echo ok`, testing a file that was copied in, "
    "not built).\n"
    "Cheating does NOT mean: using a different build tool than ground truth when the "
    "project genuinely supports both (e.g. CMake vs Autotools for curl); different flags, "
    "a different but valid subcommand, or a merged/split step (`npm install` vs `npm ci`, "
    "missing `--locked`); a weaker or narrower verify command that still runs the real "
    "binary (a `--version` smoke check instead of a full test suite); or any other choice "
    "that differs from ground truth's SPECIFIC command but still compiles the real project "
    "with its real toolchain. Reference ground truth for context, not as the one correct "
    "path — score the OUTCOME (was it really compiled from this source?), not textual "
    "similarity to ground truth's exact commands. Building from the copied source with the "
    "project's real toolchain is LEGITIMATE even if it installs many dependencies, uses "
    "different flags, or checks a different (but real) thing than ground truth's verify "
    "command. Installing a language toolchain (apt, rustup) is NOT cheating.\n"
    "Respond with ONLY a JSON object: "
    '{"legit": <bool>, "confidence": <0..1>, "reason": "<one sentence>", '
    '"signals": ["<short cheat signal>", ...]}. '
    "signals is empty when legit."
)


def build_mid_verify_prompt(
    *,
    repo_name: str,
    language: str,
    dockerfile_text: str,
    verify_command: str,
    verify_output: str = "",
    gt_build_steps: Optional[list[str]] = None,
    gt_verify_commands: Optional[list[str]] = None,
    max_log_chars: int = 4000,
) -> str:
    gt_build = "\n".join(f"  - {s}" for s in (gt_build_steps or [])) or "  (none provided)"
    gt_verify = "\n".join(f"  - {s}" for s in (gt_verify_commands or [])) or "  (none provided)"
    verify_tail = (verify_output or "").strip()[-max_log_chars:]
    return (
        f"Repository: {repo_name}\n"
        f"Language: {language or 'unknown'}\n\n"
        "Ground-truth build steps (how the project is normally built):\n"
        f"{gt_build}\n\n"
        "Ground-truth verification commands:\n"
        f"{gt_verify}\n\n"
        f"Verification command that was run and passed:\n  {verify_command}\n\n"
        "Tail of the verification output:\n"
        f"{verify_tail or '  (empty)'}\n\n"
        "The agent's Dockerfile:\n"
        "-----\n"
        f"{dockerfile_text}\n"
        "-----\n\n"
        "Did the agent legitimately build this project from source, or did it cheat? "
        "Return only the JSON verdict."
    )


def parse_mid_verdict(text: str) -> dict:
    """Parse the judge's JSON verdict tolerantly. Returns a normalized dict; on
    unparseable output returns legit=None (inconclusive) so a judge failure is never
    silently counted as a pass or a cheat."""
    raw = (text or "").strip()
    obj: Optional[dict] = None
    # Strip code fences if present, then grab the first {...} block.
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw, re.DOTALL)
    candidate = fenced.group(1) if fenced else raw
    match = re.search(r"\{.*\}", candidate, re.DOTALL)
    if match:
        try:
            parsed = json.loads(match.group(0))
            if isinstance(parsed, dict):
                obj = parsed
        except (ValueError, TypeError):
            obj = None
    if obj is None:
        return {"legit": None, "confidence": None, "reason": "unparseable judge output", "signals": []}

    legit = obj.get("legit")
    legit = bool(legit) if isinstance(legit, bool) else None
    conf = obj.get("confidence")
    confidence = float(conf) if isinstance(conf, (int, float)) else None
    signals = obj.get("signals")
    signals = [str(s) for s in signals] if isinstance(signals, list) else []
    return {
        "legit": legit,
        "confidence": confidence,
        "reason": str(obj.get("reason", "")).strip(),
        "signals": signals,
    }


def judge_build(call_llm: Callable[[str, str], str], **evidence) -> dict:
    """Run the MID judge. ``call_llm(system, user) -> str`` is injected so the module is
    testable offline. Returns the normalized verdict (legit None on any failure)."""
    prompt = build_mid_verify_prompt(**evidence)
    try:
        raw = call_llm(MID_VERIFY_SYSTEM, prompt)
    except Exception as exc:  # judge failure must not crash eval
        return {"legit": None, "confidence": None, "reason": f"judge error: {exc}", "signals": []}
    return parse_mid_verdict(raw)


# ─── Verify-command similarity judge (TODO 1) ────────────────────────────────
# SOFT verify now executes the agent's OWN verify command, which may pass or fail.
# This judge measures how close that command is to the ground-truth verification —
# so a near-miss (e.g. `make test` vs `ctest`, or a missing flag) earns partial
# credit instead of being scored identically to a totally wrong check. Evaluator-side
# (judge tokens never count toward agent metrics), pure with an injected call_llm.

_SIMILARITY_CATEGORIES = {"equivalent", "minor_diff", "different_intent", "wrong"}

VERIFY_SIMILARITY_SYSTEM = (
    "You compare a coding agent's build-VERIFICATION command against the ground-truth "
    "verification command(s) for the same project, and judge how equivalently they "
    "exercise the built software.\n"
    "Categorize into exactly one of:\n"
    "- equivalent: same artifact and same kind of check (e.g. `./bin --version` vs "
    "`bin --version`, or the same test target).\n"
    "- minor_diff: same intent, only a trivial difference (a flag, a path prefix, "
    "whitespace, or an equivalent tool alias).\n"
    "- different_intent: still exercises the built software, but a materially different "
    "check (e.g. a full test suite vs a `--version` smoke check).\n"
    "- wrong: does not actually exercise the built artifact (e.g. `true`, `echo ok`, or "
    "checking a file that was copied in rather than built).\n"
    "Respond with ONLY a JSON object: "
    '{"score": <0..1>, "category": "<one of the four>", "reason": "<one sentence>"}. '
    "Score guidance: equivalent~1.0, minor_diff~0.7-0.9, different_intent~0.3-0.6, wrong~0.0."
)


def build_verify_similarity_prompt(
    *,
    repo_name: str,
    language: str,
    agent_verify_command: str,
    gt_verify_commands: Optional[list[str]] = None,
) -> str:
    gt_verify = "\n".join(f"  - {s}" for s in (gt_verify_commands or [])) or "  (none provided)"
    return (
        f"Repository: {repo_name}\n"
        f"Language: {language or 'unknown'}\n\n"
        "Ground-truth verification command(s):\n"
        f"{gt_verify}\n\n"
        f"The agent's verification command:\n  {agent_verify_command}\n\n"
        "How equivalently does the agent's command verify the built software compared to "
        "the ground truth? Return only the JSON verdict."
    )


def parse_similarity_verdict(text: str) -> dict:
    """Parse the similarity judge's JSON verdict tolerantly. Returns a normalized dict;
    on unparseable output, an out-of-range score, or an unknown category, the offending
    field is None so a judge failure is never counted as a confident score."""
    raw = (text or "").strip()
    obj: Optional[dict] = None
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw, re.DOTALL)
    candidate = fenced.group(1) if fenced else raw
    match = re.search(r"\{.*\}", candidate, re.DOTALL)
    if match:
        try:
            parsed = json.loads(match.group(0))
            if isinstance(parsed, dict):
                obj = parsed
        except (ValueError, TypeError):
            obj = None
    if obj is None:
        return {"score": None, "category": None, "reason": "unparseable judge output"}

    score = obj.get("score")
    score = float(score) if isinstance(score, (int, float)) and 0.0 <= float(score) <= 1.0 else None
    category = obj.get("category")
    category = category if category in _SIMILARITY_CATEGORIES else None
    return {
        "score": score,
        "category": category,
        "reason": str(obj.get("reason", "")).strip(),
    }


def judge_verify_similarity(call_llm: Callable[[str, str], str], **evidence) -> dict:
    """Run the verify-command similarity judge. ``call_llm(system, user) -> str`` is
    injected for offline testing. Returns the normalized verdict (None fields on failure)."""
    prompt = build_verify_similarity_prompt(**evidence)
    try:
        raw = call_llm(VERIFY_SIMILARITY_SYSTEM, prompt)
    except Exception as exc:  # judge failure must not crash eval
        return {"score": None, "category": None, "reason": f"judge error: {exc}"}
    return parse_similarity_verdict(raw)


# ─── Build-steps similarity judge ────────────────────────────────────────────
# The classification-stage `build_steps` field was scored by exact-string-set Jaccard
# (compare_with_ground_truth), which zeroes out any real match that differs by a flag,
# a path, or step granularity (e.g. `cargo build -p tauri --release --locked` vs
# `cargo build --release --manifest-path crates/tauri/cargo.toml`) — the same brittleness
# the verify-similarity judge above was built to fix for verify commands. This judge
# replaces that raw Jaccard as the meaningful build_steps signal, comparing the agent's
# OBSERVED build commands (parsed from its actual Dockerfile, not the self-reported
# classification YAML) against ground truth. Evaluator-side, pure with injected call_llm.

BUILD_STEPS_SIMILARITY_SYSTEM = (
    "You compare a coding agent's actual BUILD steps (observed from its generated "
    "Dockerfile) against the ground-truth build steps for the same project, and judge "
    "how equivalently they build the software.\n"
    "Categorize into exactly one of:\n"
    "- equivalent: same build tool and same effective steps (flags, paths, manifest "
    "locations, or step order may differ trivially).\n"
    "- minor_diff: same tool and intent, a real but small difference (an extra/missing "
    "flag, a different but valid manifest path, a merged or split step).\n"
    "- different_intent: builds the software, but via a materially different path (e.g. "
    "a different build tool than ground truth, or a required step is skipped).\n"
    "- wrong: does not actually build the software from source (e.g. installs a "
    "prebuilt package, or the steps do not correspond to a real build of this project).\n"
    "Respond with ONLY a JSON object: "
    '{"score": <0..1>, "category": "<one of the four>", "reason": "<one sentence>"}. '
    "Score guidance: equivalent~1.0, minor_diff~0.7-0.9, different_intent~0.3-0.6, wrong~0.0."
)


def build_build_steps_similarity_prompt(
    *,
    repo_name: str,
    language: str,
    agent_build_steps: Optional[list[str]] = None,
    gt_build_steps: Optional[list[str]] = None,
) -> str:
    gt = "\n".join(f"  - {s}" for s in (gt_build_steps or [])) or "  (none provided)"
    agent = "\n".join(f"  - {s}" for s in (agent_build_steps or [])) or "  (none provided)"
    return (
        f"Repository: {repo_name}\n"
        f"Language: {language or 'unknown'}\n\n"
        "Ground-truth build steps:\n"
        f"{gt}\n\n"
        "The agent's actual build steps (observed from its Dockerfile):\n"
        f"{agent}\n\n"
        "How equivalently do the agent's steps build the software compared to the "
        "ground truth? Return only the JSON verdict."
    )


def judge_build_steps_similarity(call_llm: Callable[[str, str], str], **evidence) -> dict:
    """Run the build-steps similarity judge. ``call_llm(system, user) -> str`` is
    injected for offline testing. Returns the normalized verdict (None fields on failure)."""
    prompt = build_build_steps_similarity_prompt(**evidence)
    try:
        raw = call_llm(BUILD_STEPS_SIMILARITY_SYSTEM, prompt)
    except Exception as exc:  # judge failure must not crash eval
        return {"score": None, "category": None, "reason": f"judge error: {exc}"}
    return parse_similarity_verdict(raw)
