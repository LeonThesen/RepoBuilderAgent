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
    
    template_name = "Dockerfile.base"

    template_path = templates_dir / template_name
    if template_path.exists():
        with open(template_path, "r", encoding="utf-8") as template_file:
            return template_file.read()

    log_error(f"Base template not found at {template_path}")
    return ""
