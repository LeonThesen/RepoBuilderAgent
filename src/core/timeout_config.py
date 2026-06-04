from __future__ import annotations

from pathlib import Path
from typing import Any

try:
    import yaml
except Exception:  # pragma: no cover - optional dependency safety
    yaml = None


def _to_number(value: Any, fallback: int | float) -> int | float:
    if isinstance(fallback, int) and not isinstance(fallback, bool):
        try:
            return int(value)
        except Exception:
            return fallback
    if isinstance(fallback, float):
        try:
            return float(value)
        except Exception:
            return fallback
    return fallback


def load_timeout_defaults(section: str, defaults: dict[str, int | float]) -> dict[str, int | float]:
    """Load timeout defaults for a script from RepoBuilderAgent/config/timeouts.yaml.

    Values from `common` are merged first, then overridden by the script-specific
    section. Unknown keys are ignored and missing/invalid values fall back to the
    provided defaults.
    """
    merged: dict[str, int | float] = dict(defaults)

    config_path = Path(__file__).resolve().parent.parent / "config" / "timeouts.yaml"
    if yaml is None or not config_path.exists():
        return merged

    try:
        loaded = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except Exception:
        return merged

    if not isinstance(loaded, dict):
        return merged

    for bucket_name in ("common", section):
        bucket = loaded.get(bucket_name)
        if not isinstance(bucket, dict):
            continue
        for key, fallback in defaults.items():
            if key in bucket:
                merged[key] = _to_number(bucket[key], fallback)

    return merged
