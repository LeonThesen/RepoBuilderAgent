from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

DEFAULT_FACTORS = {
    "few_shot_count": 0,
    "prompt_length_mode": "detailed",
    "reasoning_instruction": True,
    "output_format_mode": "structured",
    "temperature": 0.0,
    "role_framing": "expert",
}


def _config_path() -> Path:
    return Path(__file__).resolve().parent.parent.parent / "config" / "prompt_profiles.yaml"


def _to_bool(value: Any, fallback: bool) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "on"}:
            return True
        if normalized in {"0", "false", "no", "off"}:
            return False
    return fallback


def _to_int(value: Any, fallback: int) -> int:
    try:
        return int(value)
    except Exception:
        return fallback


def _to_float(value: Any, fallback: float) -> float:
    try:
        return float(value)
    except Exception:
        return fallback


def _load_profiles_config() -> dict:
    path = _config_path()
    if not path.exists():
        return {}
    try:
        loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return loaded if isinstance(loaded, dict) else {}


def resolve_prompt_profile(profile_name: str | None) -> dict:
    config = _load_profiles_config()
    profiles = config.get("profiles", {}) if isinstance(config.get("profiles"), dict) else {}
    default_profile = str(config.get("default_profile") or "baseline_structured")
    pstar_profile = str(config.get("pstar_profile") or default_profile)

    requested = profile_name or default_profile
    resolved_name = pstar_profile if requested == "P*" else requested
    if resolved_name not in profiles:
        resolved_name = default_profile

    raw_profile = profiles.get(resolved_name, {}) if isinstance(profiles.get(resolved_name), dict) else {}
    factors = dict(DEFAULT_FACTORS)
    factors["few_shot_count"] = max(0, _to_int(raw_profile.get("few_shot_count"), int(DEFAULT_FACTORS["few_shot_count"])))
    factors["prompt_length_mode"] = str(raw_profile.get("prompt_length_mode") or DEFAULT_FACTORS["prompt_length_mode"])
    factors["reasoning_instruction"] = _to_bool(raw_profile.get("reasoning_instruction"), bool(DEFAULT_FACTORS["reasoning_instruction"]))
    factors["output_format_mode"] = str(raw_profile.get("output_format_mode") or DEFAULT_FACTORS["output_format_mode"])
    factors["temperature"] = _to_float(raw_profile.get("temperature"), float(DEFAULT_FACTORS["temperature"]))
    factors["role_framing"] = str(raw_profile.get("role_framing") or DEFAULT_FACTORS["role_framing"])

    return {
        "requested_profile": requested,
        "resolved_profile": resolved_name,
        "pstar_profile": pstar_profile,
        "factors": factors,
    }


def resolve_prompt_temperature(cli_temperature: float | None, profile: dict) -> float:
    if cli_temperature is not None:
        return float(cli_temperature)
    factors = profile.get("factors", {}) if isinstance(profile, dict) else {}
    return _to_float(factors.get("temperature"), 0.0)


def prompt_profile_metadata(profile: dict, effective_temperature: float) -> dict:
    factors = profile.get("factors", {}) if isinstance(profile, dict) else {}
    metadata = {
        "requested_profile": profile.get("requested_profile"),
        "resolved_profile": profile.get("resolved_profile"),
        "pstar_profile": profile.get("pstar_profile"),
        "factors": dict(factors),
        "effective_temperature": float(effective_temperature),
    }
    metadata["factors"]["temperature"] = float(effective_temperature)
    return metadata


def _profile_directives(profile: dict) -> str:
    factors = profile.get("factors", {}) if isinstance(profile, dict) else {}

    role_framing = str(factors.get("role_framing", "expert"))
    role_line = "Adopt an expert implementer tone." if role_framing == "expert" else "Use a neutral, evidence-only tone."

    # NOTE: prompt verbosity (prompt_length_mode) is NOT a directive line. The
    # concise/detailed contrast is realized by loading a different prompt file
    # (PROMPT_X.concise.md vs PROMPT_X.md) via prompt_path() — see set_prompt_length_mode
    # in core.common. Injecting a "be concise" sentence here would be a near-null run.

    reasoning_line = (
        "Before final output, reason step by step internally and verify evidence alignment."
        if bool(factors.get("reasoning_instruction", True))
        else "Do not add extra reasoning guidance beyond task instructions."
    )

    output_mode = str(factors.get("output_format_mode", "structured"))
    if output_mode == "strict":
        output_line = "Strictly follow the requested output format with no deviation."
    elif output_mode == "structured":
        output_line = "Return well-structured output that follows the requested schema and field order."
    else:
        output_line = "Use readable free-form output while preserving required fields."

    return (
        "<prompt_profile_directives>\n"
        f"- Profile: {profile.get('resolved_profile', 'unknown')}\n"
        f"- {role_line}\n"
        f"- {reasoning_line}\n"
        f"- {output_line}\n"
        "</prompt_profile_directives>"
    )


# Few-shot content is full worked <user>/<assistant> demonstrations, one file per phase.
# Each file holds <example>...</example> blocks grounded in the real artifact shapes the
# pipeline emits (classification YAML, selected_files lists, RUN-step Dockerfile bodies,
# repair patches, INSTALL.md). They teach the target OUTPUT FORMAT, not repo-specific
# answers — examples span build systems so the model generalizes (no-dataset-overfit).
#
# The demos flow ONLY through this channel (injected at {{PROMPT_PROFILE_FEWSHOT}} when
# few_shot_count > 0); none are baked into the base prompts, so a 0-shot profile is a
# true 0-shot run and the few-shot ablation (AB-05) is a clean 0 → N contrast.
_FEWSHOT_FILES = {
    "classify-step1-selection": "PROMPT_SELECT_FILES_FEWSHOT.md",
    "classify-step2":           "PROMPT_CLASSIFY_FEWSHOT.md",
    "dockerfile":               "PROMPT_DOCKERFILE_FEWSHOT.md",
    "repair":                   "PROMPT_DOCKERFILE_REPAIR_FEWSHOT.md",
    "install-guide":            "PROMPT_INSTALL_GUIDE_FEWSHOT.md",
}


def _prompts_dir() -> Path:
    # core/prompt_profiles.py -> core -> src -> RepoBuilderAgent/prompts
    return Path(__file__).resolve().parents[2] / "prompts"


def _load_file_examples(filename: str) -> list[str]:
    import re

    path = _prompts_dir() / filename
    if not path.exists():
        return []
    text = path.read_text(encoding="utf-8")
    return re.findall(r"<example\b.*?</example>", text, flags=re.DOTALL)


def _few_shot_block(profile: dict, phase: str) -> str:
    factors = profile.get("factors", {}) if isinstance(profile, dict) else {}
    count = max(0, _to_int(factors.get("few_shot_count"), 0))
    if count <= 0:
        return ""

    filename = _FEWSHOT_FILES.get(phase)
    if not filename:
        return ""

    # Take the first `count` <example> blocks and inject them verbatim. The available
    # pool may be smaller than count (then all of it is used).
    selected = _load_file_examples(filename)[:count]
    if not selected:
        return ""
    return "<few_shot_examples>\n" + "\n\n".join(selected) + "\n</few_shot_examples>"


def apply_prompt_profile(template: str, profile: dict, phase: str) -> str:
    rendered = template
    directives = _profile_directives(profile)
    few_shot = _few_shot_block(profile, phase)

    if "{{PROMPT_PROFILE_DIRECTIVES}}" in rendered:
        rendered = rendered.replace("{{PROMPT_PROFILE_DIRECTIVES}}", directives)
    else:
        rendered = directives + "\n\n" + rendered

    if "{{PROMPT_PROFILE_FEWSHOT}}" in rendered:
        rendered = rendered.replace("{{PROMPT_PROFILE_FEWSHOT}}", few_shot)
    elif few_shot:
        rendered = rendered + "\n\n" + few_shot

    return rendered
