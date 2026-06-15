"""Shared parsing of LLM YAML replies.

Models are asked to answer in YAML, often wrapped in a ```yaml fenced block.
Both shapes live here so the fence-stripping rule is defined once instead of
copied into each stage (previously three near-identical copies).
"""
from __future__ import annotations

import re
from typing import Any

import yaml

_FENCE = re.compile(r"```(?:yaml)?\n(.*?)```", re.DOTALL | re.IGNORECASE)


def parse_llm_yaml(raw: str) -> Any:
    """Parse a model reply as YAML, preferring a fenced block when present.

    Returns whatever ``yaml.safe_load`` yields (dict, list, scalar, or None) and
    may raise on malformed YAML. Callers that need a guaranteed dict should use
    :func:`parse_llm_yaml_dict`.
    """
    match = _FENCE.search(raw or "")
    content = match.group(1) if match else (raw or "")
    return yaml.safe_load(content)


def parse_llm_yaml_dict(raw: str) -> dict:
    """Lenient variant that always returns a dict — ``{}`` for an empty, malformed,
    or non-mapping reply — so callers can fall back to deterministic defaults
    instead of crashing on a bad model response.
    """
    if not isinstance(raw, str) or not raw.strip():
        return {}
    try:
        parsed = parse_llm_yaml(raw)
    except Exception:
        return {}
    return parsed if isinstance(parsed, dict) else {}
