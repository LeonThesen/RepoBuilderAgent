from __future__ import annotations

import re
from pathlib import Path
from typing import Callable


def extract_dockerfile(raw: str) -> str:
    match = re.search(r"```(?:dockerfile)?\n(.*?)```", raw, re.DOTALL | re.IGNORECASE)
    content = match.group(1) if match else raw
    return content.strip() + "\n"


def get_base_template(
    classification: dict,
    templates_dir: Path,
    *,
    log_warn: Callable[[str], None],
    log_error: Callable[[str], None],
) -> str:
    """Select and load the base Dockerfile template for the detected language."""
    languages = classification.get("categories", {}).get("programming_language", {}).get("value", [])
    if not languages:
        log_warn("No programming language detected in classification; defaulting to C template")
        template_name = "Dockerfile.base-c"
    else:
        lang = languages[0].lower()
        if "python" in lang:
            template_name = "Dockerfile.base-python"
        elif "c++" in lang or "cpp" in lang:
            template_name = "Dockerfile.base-cpp"
        elif "c" in lang and "c++" not in lang:
            template_name = "Dockerfile.base-c"
        elif "typescript" in lang or "javascript" in lang:
            template_name = "Dockerfile.base-typescript"
        elif "rust" in lang:
            template_name = "Dockerfile.base-rust"
        elif "java" in lang or "kotlin" in lang:
            template_name = "Dockerfile.base-java"
        else:
            log_warn(f"Unknown language {lang}; defaulting to C template")
            template_name = "Dockerfile.base-c"

    template_path = templates_dir / template_name
    if template_path.exists():
        with open(template_path, "r", encoding="utf-8") as template_file:
            return template_file.read()

    log_error(f"Base template not found at {template_path}")
    return ""
