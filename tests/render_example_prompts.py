"""Write rendered prompt examples for the Phase-1 prompt-text arms to tests/rendered_prompts/.

Uses the actual prompt templates from RepoBuilderAgent/prompts/ so the output
matches what the pipeline sends to the LLM (before per-repo variable substitution).
AB-03 (verbosity) loads the caveman-compressed PROMPT_*.concise.md variants via the
same prompt_path() + set_prompt_length_mode() mechanism the pipeline uses at runtime.

AB-01/AB-02 are omitted: they vary only temperature, which does not change prompt text,
so they render identically to AB-00.

Usage:
    python RepoBuilderAgent/tests/render_example_prompts.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from core.common import prompt_path, set_prompt_length_mode
from core.prompt_profiles import apply_prompt_profile, resolve_prompt_profile

OUT_DIR = Path(__file__).parent / "rendered_prompts"
OUT_DIR.mkdir(exist_ok=True)

# Map prompt-profile phase key -> prompt file name (resolved through prompt_path so
# concise mode picks the .concise.md sibling automatically).
PHASE_PROMPTS = {
    "classify-step1-selection": "PROMPT_SELECT_FILES.md",
    "classify-step2":           "PROMPT.md",
    "dockerfile":               "PROMPT_DOCKERFILE.md",
    "repair":                   "PROMPT_DOCKERFILE_REPAIR.md",
    "install-guide":            "PROMPT_INSTALL_GUIDE.md",
}

# Only the arms that change prompt TEXT (AB-00 anchor + the three OFAT prompt arms).
PROFILES = [
    ("AB-00", "phase1_zero_baseline"),
    ("AB-03", "phase1_length_only"),
    ("AB-04", "phase1_role_only"),
    ("AB-05", "phase1_few_shot_only"),
]

written = 0
for run_label, profile_name in PROFILES:
    profile = resolve_prompt_profile(profile_name)
    set_prompt_length_mode(profile["factors"]["prompt_length_mode"])
    for phase, prompt_name in PHASE_PROMPTS.items():
        template = prompt_path(prompt_name).read_text(encoding="utf-8")
        rendered = apply_prompt_profile(template, profile, phase)
        out_path = OUT_DIR / f"{run_label}_{profile_name}_{phase}.txt"
        out_path.write_text(rendered, encoding="utf-8")
        print(f"wrote {out_path.relative_to(Path(__file__).parent.parent.parent)}")
        written += 1
set_prompt_length_mode("detailed")

print(f"\n{written} files written to {OUT_DIR}")
