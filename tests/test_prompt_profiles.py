"""Tests for apply_prompt_profile and related helpers in core/prompt_profiles.py.

Run from the repo root:
    pytest RepoBuilderAgent/tests/test_prompt_profiles.py -v
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
        profile = _resolve("phase1_fewshot_0")
        result = apply_prompt_profile(TEMPLATE_WITH_MARKERS, profile, "classify-step2")
        assert "{{PROMPT_PROFILE_DIRECTIVES}}" not in result
        assert "<prompt_profile_directives>" in result

    def test_directives_prepended_when_no_marker(self):
        profile = _resolve("phase1_fewshot_0")
        result = apply_prompt_profile(TEMPLATE_NO_MARKERS, profile, "classify-step2")
        assert result.startswith("<prompt_profile_directives>")

    def test_profile_name_appears_in_directives(self):
        profile = _resolve("phase1_fewshot_0")
        result = apply_prompt_profile(TEMPLATE_WITH_MARKERS, profile, "classify-step2")
        assert "phase1_fewshot_0" in result

    def test_structured_output_mode_directive(self):
        # phase1_structured_output has output_format_mode: structured
        profile = _resolve("phase1_structured_output")
        result = apply_prompt_profile(TEMPLATE_WITH_MARKERS, profile, "dockerfile")
        assert "Strictly follow the requested output format." in result

    def test_strict_output_mode_directive(self):
        # phase1_fewshot_1 has output_format_mode: strict — same directive text
        profile = _resolve("phase1_fewshot_1")
        result = apply_prompt_profile(TEMPLATE_WITH_MARKERS, profile, "dockerfile")
        assert "Strictly follow the requested output format." in result


# ─── Few-shot injection ───────────────────────────────────────────────────────

class TestFewShot:
    def test_zero_examples_produces_no_block(self):
        profile = _resolve("phase1_fewshot_0")
        result = apply_prompt_profile(TEMPLATE_WITH_MARKERS, profile, "classify-step2")
        assert "<few_shot_hints>" not in result
        assert "{{PROMPT_PROFILE_FEWSHOT}}" not in result

    def test_one_example_injected(self):
        profile = _resolve("phase1_fewshot_1")
        result = apply_prompt_profile(TEMPLATE_WITH_MARKERS, profile, "classify-step2")
        assert "- Example 1:" in result
        assert "- Example 2:" not in result

    def test_three_examples_injected(self):
        profile = _resolve("phase1_fewshot_3")
        result = apply_prompt_profile(TEMPLATE_WITH_MARKERS, profile, "classify-step2")
        assert "- Example 1:" in result
        assert "- Example 2:" in result
        assert "- Example 3:" in result

    def test_examples_appended_when_no_marker(self):
        profile = _resolve("phase1_fewshot_1")
        result = apply_prompt_profile(TEMPLATE_NO_MARKERS, profile, "classify-step2")
        assert result.endswith("</few_shot_hints>")

    def test_examples_are_phase_specific(self):
        # dockerfile phase has different example text than classify-step2
        profile = _resolve("phase1_fewshot_1")
        result_classify = apply_prompt_profile(TEMPLATE_NO_MARKERS, profile, "classify-step2")
        result_dockerfile = apply_prompt_profile(TEMPLATE_NO_MARKERS, profile, "dockerfile")
        assert result_classify != result_dockerfile

    def test_unknown_phase_produces_no_examples(self):
        profile = _resolve("phase1_fewshot_3")
        result = apply_prompt_profile(TEMPLATE_WITH_MARKERS, profile, "nonexistent-phase")
        assert "<few_shot_hints>" not in result


# ─── Profile differences visible in output ────────────────────────────────────

class TestProfileDifferences:
    def test_fewshot_0_vs_fewshot_3_differ(self):
        p0 = _resolve("phase1_fewshot_0")
        p3 = _resolve("phase1_fewshot_3")
        r0 = apply_prompt_profile(TEMPLATE_WITH_MARKERS, p0, "dockerfile")
        r3 = apply_prompt_profile(TEMPLATE_WITH_MARKERS, p3, "dockerfile")
        assert r0 != r3
        assert "<few_shot_hints>" not in r0
        assert "<few_shot_hints>" in r3

    def test_temperature_not_in_prompt_text(self):
        # temperature only affects the API call, not the prompt string
        profile = _resolve("phase1_fewshot_0")
        result = apply_prompt_profile(TEMPLATE_WITH_MARKERS, profile, "classify-step2")
        assert "temperature" not in result.lower()
