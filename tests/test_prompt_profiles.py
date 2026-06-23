"""Tests for apply_prompt_profile and related helpers in core/prompt_profiles.py.

Run from the repo root:
    pytest RepoBuilderAgent/tests/test_prompt_profiles.py -v

Uses only the live Phase-1 study profiles (no legacy profiles):
    phase1_zero_baseline  — anchor: 0-shot, detailed, strict, neutral
    phase1_role_only      — expert role framing
    phase1_few_shot_only  — 3 few-shot hints
    baseline_structured   — P* default: structured output
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from core.prompt_profiles import apply_prompt_profile, resolve_prompt_profile

# ─── Helpers ──────────────────────────────────────────────────────────────────

def _resolve(name: str) -> dict:
    return resolve_prompt_profile(name)


TEMPLATE_WITH_MARKERS = (
    "{{PROMPT_PROFILE_DIRECTIVES}}\n\nAnalyse the repo.\n\n{{PROMPT_PROFILE_FEWSHOT}}"
)
TEMPLATE_NO_MARKERS = "Analyse the repo."


# ─── Directives block ─────────────────────────────────────────────────────────

class TestDirectivesBlock:
    def test_markers_are_replaced(self):
        profile = _resolve("phase1_zero_baseline")
        result = apply_prompt_profile(TEMPLATE_WITH_MARKERS, profile, "classify-step2")
        assert "{{PROMPT_PROFILE_DIRECTIVES}}" not in result
        assert "<prompt_profile_directives>" in result

    def test_directives_prepended_when_no_marker(self):
        profile = _resolve("phase1_zero_baseline")
        result = apply_prompt_profile(TEMPLATE_NO_MARKERS, profile, "classify-step2")
        assert result.startswith("<prompt_profile_directives>")

    def test_profile_name_appears_in_directives(self):
        profile = _resolve("phase1_zero_baseline")
        result = apply_prompt_profile(TEMPLATE_WITH_MARKERS, profile, "classify-step2")
        assert "phase1_zero_baseline" in result

    def test_strict_output_mode_directive(self):
        # phase1_zero_baseline has output_format_mode: strict
        profile = _resolve("phase1_zero_baseline")
        result = apply_prompt_profile(TEMPLATE_WITH_MARKERS, profile, "dockerfile")
        assert "Strictly follow the requested output format" in result

    def test_structured_output_mode_directive(self):
        # baseline_structured (P*) has output_format_mode: structured
        profile = _resolve("baseline_structured")
        result = apply_prompt_profile(TEMPLATE_WITH_MARKERS, profile, "dockerfile")
        assert "well-structured output" in result

    def test_role_framing_directive_differs(self):
        neutral = apply_prompt_profile(TEMPLATE_WITH_MARKERS, _resolve("phase1_zero_baseline"), "dockerfile")
        expert = apply_prompt_profile(TEMPLATE_WITH_MARKERS, _resolve("phase1_role_only"), "dockerfile")
        assert "neutral, evidence-only tone" in neutral
        assert "expert implementer tone" in expert


# ─── Few-shot injection ───────────────────────────────────────────────────────

class TestFewShot:
    def test_zero_examples_produces_no_block(self):
        profile = _resolve("phase1_zero_baseline")
        result = apply_prompt_profile(TEMPLATE_WITH_MARKERS, profile, "classify-step2")
        assert "<few_shot_hints>" not in result
        assert "{{PROMPT_PROFILE_FEWSHOT}}" not in result

    def test_two_examples_injected(self):
        # few_shot_count=2 -> exactly 2 <example> demos for any file-backed phase.
        profile = _resolve("phase1_few_shot_only")
        for phase in ("classify-step1-selection", "classify-step2", "dockerfile", "repair", "install-guide"):
            result = apply_prompt_profile(TEMPLATE_WITH_MARKERS, profile, phase)
            assert "<few_shot_examples>" in result, phase
            assert result.count("<example index=") == 2, phase

    def test_worked_demos_only_via_profile(self):
        # Demos flow ONLY through the few-shot channel (not baked into base prompts):
        # a 0-shot profile injects nothing, an N-shot profile injects worked <example>s.
        zero = apply_prompt_profile(TEMPLATE_WITH_MARKERS, _resolve("phase1_zero_baseline"), "classify-step2")
        few = apply_prompt_profile(TEMPLATE_WITH_MARKERS, _resolve("phase1_few_shot_only"), "classify-step2")
        assert "<example" not in zero
        assert "<few_shot_examples>" in few
        assert "<example index=" in few

    def test_examples_appended_when_no_marker(self):
        profile = _resolve("phase1_few_shot_only")
        result = apply_prompt_profile(TEMPLATE_NO_MARKERS, profile, "dockerfile")
        assert result.endswith("</few_shot_examples>")

    def test_examples_are_phase_specific(self):
        # dockerfile phase has different example text than classify-step2
        profile = _resolve("phase1_few_shot_only")
        result_classify = apply_prompt_profile(TEMPLATE_NO_MARKERS, profile, "classify-step2")
        result_dockerfile = apply_prompt_profile(TEMPLATE_NO_MARKERS, profile, "dockerfile")
        assert result_classify != result_dockerfile

    def test_unknown_phase_produces_no_examples(self):
        profile = _resolve("phase1_few_shot_only")
        result = apply_prompt_profile(TEMPLATE_WITH_MARKERS, profile, "nonexistent-phase")
        assert "<few_shot_hints>" not in result


# ─── Profile differences visible in output ────────────────────────────────────

class TestProfileDifferences:
    def test_zero_vs_few_shot_differ(self):
        p0 = _resolve("phase1_zero_baseline")
        p3 = _resolve("phase1_few_shot_only")
        r0 = apply_prompt_profile(TEMPLATE_WITH_MARKERS, p0, "dockerfile")
        r3 = apply_prompt_profile(TEMPLATE_WITH_MARKERS, p3, "dockerfile")
        assert r0 != r3
        assert "<few_shot_examples>" not in r0
        assert "<few_shot_examples>" in r3

    def test_temperature_not_in_prompt_text(self):
        # temperature only affects the API call, not the prompt string
        profile = _resolve("phase1_zero_baseline")
        result = apply_prompt_profile(TEMPLATE_WITH_MARKERS, profile, "classify-step2")
        assert "temperature" not in result.lower()
