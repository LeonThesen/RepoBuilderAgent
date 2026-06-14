"""Write rendered prompt examples for all Phase 1 profiles to tests/rendered_prompts/.

Uses the actual prompt templates from RepoBuilderAgent/prompts/ so the output
matches what the pipeline sends to the LLM (before per-repo variable substitution).

Usage:
    python RepoBuilderAgent/tests/render_example_prompts.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from core.prompt_profiles import apply_prompt_profile, resolve_prompt_profile

PROMPTS_DIR = Path(__file__).parent.parent / "prompts"
OUT_DIR = Path(__file__).parent / "rendered_prompts"
OUT_DIR.mkdir(exist_ok=True)

# Map prompt-profile phase key -> actual template file
PHASE_TEMPLATES = {
    "classify-step1-selection": PROMPTS_DIR / "PROMPT_SELECT_FILES.md",
    "classify-step2":           PROMPTS_DIR / "PROMPT.md",
    "dockerfile":               PROMPTS_DIR / "PROMPT_DOCKERFILE.md",
    "repair":                   PROMPTS_DIR / "PROMPT_DOCKERFILE_REPAIR.md",
    "install-guide":            PROMPTS_DIR / "PROMPT_INSTALL_GUIDE.md",
}

PROFILES = [
    ("AB-00", "phase1_fewshot_0"),
    ("AB-01", "phase1_fewshot_1"),
    ("AB-02", "phase1_fewshot_3"),
    ("AB-03", "phase1_structured_output"),
]

for run_label, profile_name in PROFILES:
    profile = resolve_prompt_profile(profile_name)
    for phase, template_path in PHASE_TEMPLATES.items():
        template = template_path.read_text(encoding="utf-8")
        rendered = apply_prompt_profile(template, profile, phase)
        out_path = OUT_DIR / f"{run_label}_{profile_name}_{phase}.txt"
        out_path.write_text(rendered, encoding="utf-8")
        print(f"wrote {out_path.relative_to(Path(__file__).parent.parent.parent)}")

total = len(PROFILES) * len(PHASE_TEMPLATES)
print(f"\n{total} files written to {OUT_DIR}")
