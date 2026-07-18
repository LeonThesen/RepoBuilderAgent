"""
eval_metrics_lib.py – Shared metrics computation for eval.py and eval_metrics.py.

Import this module instead of duplicating logic between the two scripts.
"""

from __future__ import annotations

import fnmatch
import json
import re
import shutil
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Optional

try:
    import yaml
    _YAML_AVAILABLE = True
except ImportError:
    _YAML_AVAILABLE = False


# ---------------------------------------------------------------------------
# Ground truth comparison
# ---------------------------------------------------------------------------

# Fields to compare between the agent's prediction and the ground truth.
# Each entry: (field_key, comparison_mode)
#   "set"    – normalise both to lowercase sets, compute Jaccard similarity
#   "exact"  – case-insensitive exact string equality (score 0 or 1)
# `system_dependencies` and `build_steps` are deliberately absent:
#   - system_dependencies: the dataset's GT schema has no such field (it stores package
#     names under `packages`) — comparing against it always scored 0.0, and the real
#     package comparison already exists as `packages` ("apt_packages" mode, below).
#   - build_steps: exact-string-set Jaccard zeroed out near-identical builds that only
#     differed by a flag or path (e.g. two `cargo build` invocations with different
#     manifest paths). Replaced by the `build_steps_similarity` LLM judge (mid_verify.py),
#     which scores build-command equivalence the same way `verify_cmd_similarity` already
#     does for verify commands.
_COMPARABLE_FIELDS: list[tuple[str, str]] = [
    ("programming_language", "set"),
    ("build_tool",           "set"),
    ("packages",             "apt_packages"),
    ("verification",         "set"),
    ("installation_strategy","exact"),
    ("runtime_environment",  "set"),
    ("os_compatibility",     "set"),
]


def _load_yaml(path: Path) -> Optional[dict]:
    if not _YAML_AVAILABLE or not path.exists():
        return None
    try:
        with open(path, encoding="utf-8") as fh:
            return yaml.safe_load(fh)
    except Exception:
        return None


def _get_field_value(doc: dict, field: str):
    """Extract a field value from a schema v1.0 or v1.1 classification YAML."""
    cats = doc.get("categories", {}) if isinstance(doc, dict) else {}
    # A schema-drifted classification can emit `categories` as a list (e.g. a bare
    # list of domain tags) instead of the field->value mapping. Treat any non-dict
    # shape as "field not predicted" rather than crashing the whole eval run.
    if not isinstance(cats, dict):
        return None
    entry = cats.get(field, {})
    if entry is None:
        return None
    if isinstance(entry, dict):
        val = entry.get("value")
        if val is None:
            # v1.0 schema: deps might be nested differently
            val = entry
        return val
    return entry


def _pred_steps(steps):
    """Agent build_steps/verification are lists of {step, command} dicts. Pull the
    command text (the comparable part) — falling back to the step label, then raw
    strings. GT stores these as flat command-string lists, so this lines the two up."""
    if not isinstance(steps, list):
        return steps
    out: list[str] = []
    for s in steps:
        if isinstance(s, dict):
            out.append(str(s.get("command") or s.get("step") or "").strip())
        elif isinstance(s, str):
            out.append(s.strip())
    return [s for s in out if s]


def _pred_system_packages(doc: dict) -> list[str]:
    """Agent-predicted system packages from the flat classification schema. The agent
    lists these under `dependencies.{build,runtime,system}` (and occasionally a flat
    `system_dependencies`), NOT under GT's `categories.system_dependencies.value`."""
    names: list[str] = []
    deps = doc.get("dependencies")
    if isinstance(deps, dict):
        for key in ("build", "runtime", "system"):
            for item in deps.get(key) or []:
                if isinstance(item, dict):
                    names.append(str(item.get("name", "")).strip().lower())
                elif isinstance(item, str):
                    names.append(item.strip().lower())
    sd = doc.get("system_dependencies")
    if isinstance(sd, list):
        for item in sd:
            if isinstance(item, dict):
                names.append(str(item.get("name", "")).strip().lower())
            elif isinstance(item, str):
                names.append(item.strip().lower())
    return [n for n in names if n]


def _get_pred_field_value(doc: dict, field: str):
    """Extract a GT field from the AGENT'S flat classification schema, which differs
    from the GT's `categories.<field>.value` shape and drifts across repos. Maps each
    GT field name to the agent's actual key(s), tolerating absence. If a prediction
    happens to carry a GT-style `categories` dict, defer to _get_field_value."""
    if not isinstance(doc, dict):
        return None
    cats = doc.get("categories")
    if isinstance(cats, dict) and field in cats:
        return _get_field_value(doc, field)
    if field == "build_tool":
        return doc.get("build_systems") or doc.get("build_tool")
    if field == "programming_language":
        return doc.get("programming_language") or doc.get("languages") or doc.get("language")
    if field in ("system_dependencies", "packages"):
        return _pred_system_packages(doc)
    if field in ("build_steps", "verification"):
        return _pred_steps(doc.get(field))
    # installation_strategy / runtime_environment / os_compatibility: direct key or absent.
    return doc.get(field)


def _normalise_list(value) -> set[str]:
    """Turn any value into a normalised, lowercase set of strings."""
    if value is None:
        return set()
    if isinstance(value, str):
        return {value.lower().strip()} if value.strip() else set()
    if isinstance(value, list):
        result = set()
        for item in value:
            if isinstance(item, str):
                result.add(item.lower().strip())
            elif isinstance(item, dict):
                # e.g. build_steps are sometimes dicts with sub-keys; stringify
                result.add(str(list(item.values())[0]).lower().strip() if item else "")
        return result - {""}
    if isinstance(value, dict):
        # e.g. language_version: {C: unknown} → {"c"}
        return {k.lower() for k in value}
    return set()


def _normalise_apt_packages(value) -> set[str]:
    """Extract apt package names from a ground-truth ``packages`` list.

    Dataset entries are ``name:manager`` strings (e.g. ``cmake:apt``,
    ``python3-pip:apt``). Keep only apt-managed packages (and bare names with no
    manager suffix, treated as system packages) so the set lines up with what the
    Dockerfile actually installs via apt-get/apk. Returns lowercase names.
    """
    if value is None:
        return set()
    items = value if isinstance(value, list) else [value]
    names: set[str] = set()
    for item in items:
        if isinstance(item, dict):
            item = item.get("name") or (next(iter(item.values()), "") if item else "")
        text = str(item).strip().lower()
        if not text:
            continue
        if ":" in text:
            name, manager = text.rsplit(":", 1)
            if manager not in ("apt", ""):
                continue
            text = name.strip()
        if text:
            names.add(text)
    return names


# Generic Docker base-image → provided-toolchain map. Keyed on language/toolchain
# images only (NOT repo names) — a `rust:` image ships rustc+cargo, a `node:` image
# ships nodejs+npm, etc. Used to subtract base-provided packages so the apt metric
# scores only the EXTRA system packages a build must install on top of its base.
# Each entry: (base-image-name regex, [provided package-name regexes]).
_BASE_IMAGE_PROVIDES: list[tuple[str, list[str]]] = [
    (r"(^|/)rust(:|$)",                       [r"^rustc$", r"^cargo$"]),
    (r"(^|/)node(:|$)",                       [r"^nodejs$", r"^node$", r"^npm$", r"^yarn$"]),
    (r"(^|/)python(:|$)",                     [r"^python3?$", r"^python3-pip$", r"^pip3?$",
                                              r"^python3-venv$", r"^python3-dev$"]),
    (r"(^|/)golang(:|$)|(^|/)go(:|$)",        [r"^golang.*$", r"^go$"]),
    (r"(^|/)gcc(:|$)|buildpack-deps",         [r"^gcc$", r"^g\+\+$", r"^build-essential$"]),
    # JDK/JRE-bearing bases (incl. maven/gradle images, which are JDK-based)
    (r"eclipse-temurin|openjdk|amazoncorretto|(^|/)maven(:|$)|(^|/)gradle(:|$)",
     [r"^openjdk-\d+-(jdk|jre)(-headless)?$", r"^default-(jdk|jre)(-headless)?$", r"^java$"]),
    (r"(^|/)maven(:|$)",                      [r"^maven$"]),
    (r"(^|/)gradle(:|$)",                     [r"^gradle$"]),
]


def base_provided_matchers(base_images: list[str]) -> list:
    """Compiled package-name regexes for toolchains the given base images provide.

    A GT/installed apt package is "base-provided" (and so excluded from the extras
    comparison) if its name matches any returned matcher. Unions across every FROM
    stage so a toolchain provided by any stage is credited.
    """
    matchers = []
    for image in base_images:
        img = (image or "").lower()
        for image_re, pkg_res in _BASE_IMAGE_PROVIDES:
            if re.search(image_re, img):
                matchers.extend(re.compile(p) for p in pkg_res)
    return matchers


def _drop_base_provided(packages: set[str], matchers: list) -> set[str]:
    """Remove packages a base image already provides, leaving only extras."""
    if not matchers:
        return set(packages)
    return {p for p in packages if not any(m.match(p) for m in matchers)}


def _jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 1.0
    union = a | b
    if not union:
        return 1.0
    return round(len(a & b) / len(union), 4)


def _normalise_str(value) -> str:
    if value is None:
        return ""
    return str(value).lower().strip()


def build_ground_truth_index(dataset_dir: Path) -> dict[str, Path]:
    """
    Scan all *.yaml files under dataset_dir and return a mapping of
    repo URL -> yaml file path, using the 'repo:' field inside each file.
    """
    index: dict[str, Path] = {}
    if not _YAML_AVAILABLE or not dataset_dir.exists():
        return index
    for yaml_path in sorted(dataset_dir.rglob("*.yaml")):
        if yaml_path.name == "schema.yml":
            continue
        doc = _load_yaml(yaml_path)
        if doc and isinstance(doc.get("repo"), str):
            url = doc["repo"].rstrip("/")
            index[url] = yaml_path
    return index


def load_gt_for_repo(dataset_dir: Path, repo_url: str) -> Optional[dict]:
    """Load the ground truth YAML for a single repo URL. Returns None if not found."""
    index = build_ground_truth_index(dataset_dir)
    gt_path = index.get(repo_url.rstrip("/"))
    return _load_yaml(gt_path) if gt_path else None


def get_gt_verify_commands(gt_doc: dict) -> list[str]:
    """Extract verification commands from a GT YAML document as a list of strings."""
    if not gt_doc:
        return []
    v = gt_doc.get("categories", {}).get("verification", {})
    if not v:
        return []
    cmds = v.get("value", [])
    if isinstance(cmds, str):
        return [cmds] if cmds.strip() else []
    return [c for c in cmds if isinstance(c, str) and c.strip()]


def get_gt_build_steps(gt_doc: dict) -> list[str]:
    """Extract build steps from a GT YAML document as a list of strings."""
    if not gt_doc:
        return []
    v = gt_doc.get("categories", {}).get("build_steps", {})
    if not v:
        return []
    steps = v.get("value", [])
    if isinstance(steps, str):
        return [steps] if steps.strip() else []
    return [s for s in steps if isinstance(s, str) and s.strip()]


def get_gt_key_artifact(gt_doc: dict, verify_commands: list[str]) -> Optional[dict]:
    """
    Find the key final artifact for binary size/hash comparison.

    Strategy: extract relative file paths referenced in verify commands
    (e.g. 'target/release/bat --version' → 'target/release/bat'),
    then match against the GT artifacts list.

    Returns {'path': str, 'size_bytes': int|None, 'digest': str|None} or None.
    """
    if not gt_doc or not verify_commands:
        return None
    artifacts = gt_doc.get("categories", {}).get("artifacts", {}).get("value", [])
    if not artifacts:
        return None

    # Collect candidate relative paths from verify commands.
    # A token is a candidate if it contains '/' and doesn't look like a flag or env var.
    candidate_paths: set[str] = set()
    for cmd in verify_commands:
        parts = cmd.strip().split()
        for part in parts:
            if "/" in part and not part.startswith("-") and not part.startswith("$") and not part.startswith("http"):
                # Strip leading './' so paths match artifact locations
                candidate_paths.add(part.lstrip("./"))

    if not candidate_paths:
        return None

    artifact_by_loc: dict[str, dict] = {}
    for art in artifacts:
        loc = art.get("location", "").lstrip("./")
        if loc:
            artifact_by_loc[loc] = art

    for cand in candidate_paths:
        art = artifact_by_loc.get(cand)
        if art:
            return {
                "path": art.get("location", cand),
                "size_bytes": art.get("size_bytes"),
                "digest": art.get("digest"),
            }

    return None


def observe_dockerfile(workspace_root: Path, repo_url: str, dockerfiles_dir: str = "dockerfiles") -> dict:
    """
    Parse the generated Dockerfile to extract observed build facts:
    - base_image and inferred language/version
    - system packages installed via apt-get / apk
    - observed build commands (RUN lines, stripped of shell boilerplate)
    """
    repo_name = repo_url.rstrip("/").split("/")[-1].replace(".git", "")
    df_path = workspace_root / dockerfiles_dir / f"{repo_name}.Dockerfile"
    if not df_path.exists():
        return {}
    try:
        content = df_path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return {}

    lines = content.splitlines()

    # FROM lines (multi-stage: keep every base image; first is the primary)
    from_lines = [l.strip() for l in lines if l.strip().upper().startswith("FROM ")]
    base_images = [p.split()[1] for p in from_lines if len(p.split()) >= 2]
    base_image = base_images[0] if base_images else ""

    # System packages from apt-get install / apk add lines. Collapse shell
    # line-continuations (``\`` + newline) into one logical line first, then capture
    # each install's argument run up to the next command boundary (``&&``/``;``/
    # newline). This catches single-line, ``;``-terminated, and multi-line installs
    # alike; arguments beginning with ``-`` (flags) are dropped below.
    apt_packages: list[str] = []
    joined = re.sub(r"\\\s*\n", " ", content)
    _apt_re = re.compile(r"apt(?:-get)?\s+install\s+([^\n;]*?)(?:&&|;|\n|$)", re.I)
    _apk_re = re.compile(r"apk\s+add\s+([^\n;]*?)(?:&&|;|\n|$)", re.I)
    for m in _apt_re.finditer(joined):
        apt_packages.extend(t for t in m.group(1).split() if not t.startswith("-"))
    for m in _apk_re.finditer(joined):
        apt_packages.extend(t for t in m.group(1).split() if not t.startswith("-"))
    apt_packages = sorted({p.lower().rstrip("\\") for p in apt_packages if p.strip()})

    # RUN commands that look like build steps (not apt/apk installs, not env setup)
    build_run_commands: list[str] = []
    _skip_prefixes = ("apt", "apk", "yum", "dnf", "pip", "npm", "yarn", "pnpm",
                      "cargo", "mvn", "gradle", "echo", "mkdir", "chmod", "chown",
                      "ln ", "update-ca", "curl", "wget")
    in_run = False
    run_buf: list[str] = []
    for line in lines:
        stripped = line.strip()
        if stripped.upper().startswith("RUN "):
            in_run = True
            run_buf = [stripped[4:].strip()]
        elif in_run:
            if stripped.endswith("\\"):
                run_buf.append(stripped.rstrip("\\").strip())
            else:
                run_buf.append(stripped)
                in_run = False
                cmd = " ".join(run_buf).strip()
                if cmd and not any(cmd.lower().startswith(p) for p in _skip_prefixes):
                    build_run_commands.append(cmd)
                run_buf = []

    return {
        "base_image": base_image,
        "base_images": base_images,
        "system_packages_installed": apt_packages,
        "build_run_commands": build_run_commands,
    }


def compare_with_ground_truth(
    workspace_root: Path,
    repo_url: str,
    dataset_dir: Path,
    ground_truth_index: dict[str, Path],
    results_dir: str = "classification_results",
    dockerfiles_dir: str = "dockerfiles",
) -> Optional[dict]:
    """
    Compare the agent's classification prediction and observed Dockerfile
    against the dataset ground truth YAML for this repo.

    """
    url_key = repo_url.rstrip("/")
    gt_path = ground_truth_index.get(url_key)
    if gt_path is None:
        return None

    ground_truth = _load_yaml(gt_path)
    if not ground_truth:
        return None

    repo_name = repo_url.rstrip("/").split("/")[-1].replace(".git", "")
    pred_path = workspace_root / results_dir / f"{repo_name}.yaml"
    prediction = _load_yaml(pred_path)
    if not prediction:
        return None

    observed = observe_dockerfile(workspace_root, repo_url, dockerfiles_dir)

    field_scores: dict[str, dict] = {}
    total_score = 0.0
    scored_fields = 0

    for field, mode in _COMPARABLE_FIELDS:
        gt_val = _get_field_value(ground_truth, field)
        pred_val = _get_pred_field_value(prediction, field)

        if mode == "set":
            gt_set = _normalise_list(gt_val)
            pred_set = _normalise_list(pred_val)

            score = _jaccard(gt_set, pred_set)
            field_scores[field] = {
                "mode": "set",
                "ground_truth": sorted(gt_set),
                "predicted": sorted(pred_set),
                "intersection": sorted(gt_set & pred_set),
                "only_in_gt": sorted(gt_set - pred_set),
                "only_in_pred": sorted(pred_set - gt_set),
                "jaccard": score,
            }
        elif mode == "apt_packages":
            # EXTRAS-ONLY apt comparison. The GT package list mixes base-provided
            # toolchains (rustc, nodejs, python3 — shipped by the chosen base image)
            # with extra system packages a build must apt-install on top. Subtract
            # base-provided from BOTH sides and score only the extras: did the agent
            # apt-install the extra system packages the build actually needs?
            matchers = base_provided_matchers(observed.get("base_images", []))
            gt_all = _normalise_apt_packages(gt_val)
            installed_all = {p.lower() for p in observed.get("system_packages_installed", [])}
            gt_extra = _drop_base_provided(gt_all, matchers)
            installed_extra = _drop_base_provided(installed_all, matchers)
            inter = gt_extra & installed_extra
            # recall/precision/jaccard only defined when there ARE extras to score;
            # a build fully covered by its base image yields recall=None (n/a).
            recall = round(len(inter) / len(gt_extra), 4) if gt_extra else None
            precision = round(len(inter) / len(installed_extra), 4) if (gt_extra and installed_extra) else None
            score = _jaccard(gt_extra, installed_extra) if gt_extra else None
            field_scores[field] = {
                "mode": "apt_packages",
                "ground_truth": sorted(gt_extra),       # GT extras (base-provided removed)
                "installed": sorted(installed_extra),   # installed extras
                "base_provided_excluded": sorted((gt_all - gt_extra) | (installed_all - installed_extra)),
                "intersection": sorted(inter),
                "missing": sorted(gt_extra - installed_extra),    # needed extras NOT installed
                "extra": sorted(installed_extra - gt_extra),      # installed but not in GT extras
                "jaccard": score,
                "recall": recall,
                "precision": precision,
            }
        else:  # exact
            gt_str = _normalise_str(gt_val)
            pred_str = _normalise_str(pred_val)
            score = 1.0 if gt_str == pred_str and gt_str != "" else 0.0
            field_scores[field] = {
                "mode": "exact",
                "ground_truth": gt_str,
                "predicted": pred_str,
                "match": score == 1.0,
                "score": score,
            }

        # Only include field in overall average if ground truth has a non-empty value
        gt_nonempty = bool(field_scores[field].get("ground_truth") or field_scores[field].get("ground_truth") == 0)
        if gt_nonempty:
            total_score += score
            scored_fields += 1

    overall = round(total_score / scored_fields, 4) if scored_fields > 0 else None

    return {
        "ground_truth_file": str(gt_path),
        "prediction_file": str(pred_path),
        "observed_dockerfile": observed,
        "overall_score": overall,
        "scored_fields": scored_fields,
        "field_scores": field_scores,
    }


# Ordered most-specific → most-generic; classify_failure returns the first hit. Build-layer
# modes (apt, toolchain, compiler, linker, deps) precede the broad docker-layer ones so a
# real cause like "Unable to locate package" isn't masked by an incidental "Permission denied"
# elsewhere in the log. Each label names an ACTIONABLE failure class for repair triage.
FAILURE_PATTERNS = [
    # Docker / image layer
    ("wrong_base_image",     [re.compile(r"FROM .* not found|manifest for .* not found|pull access denied", re.I)]),
    ("dockerfile_syntax",    [re.compile(r"dockerfile parse error|unknown instruction|syntax error", re.I)]),
    # apt / package availability (a common failure: a version Ubuntu 24.04 does not carry)
    ("apt_package_missing",  [re.compile(r"Unable to locate package|has no installation candidate|Couldn't find any package", re.I)]),
    ("apt_repo_error",       [re.compile(r"Failed to fetch|apt-get update.*exit code|Could not get lock|GPG error|NO_PUBKEY", re.I)]),
    # network / TLS (CA bootstrap, proxy, DNS)
    ("network_tls",          [re.compile(r"SSL certificate problem|server certificate verification failed|unable to get local issuer|self.signed certificate|Could not resolve host|Temporary failure in name resolution|Connection timed out|failed: timeout", re.I)]),
    # source completeness (submodules not checked out)
    ("missing_submodule",    [re.compile(r"Missing the .* submodule|git submodule update --init|submodule .* (?:not initialized|is not registered)", re.I)]),
    # toolchain misconfiguration
    ("java_home_invalid",    [re.compile(r"JAVA_HOME is set to an invalid directory|JAVA_HOME is not defined", re.I)]),
    ("command_not_found",    [re.compile(r"\bcommand not found\b|: \d+: [^:]+: not found", re.I)]),
    ("pkg_config_missing",   [re.compile(r"No package '[^']+' found|required by crate .* was not found|pkg-config.*not found", re.I)]),
    # dependency resolution (maven/gradle/npm/cargo registry)
    ("dependency_resolution",[re.compile(r"Could not resolve dependencies|Could not find artifact|Could not determine the dependencies|npm ERR!|ERESOLVE|version conflict|could not download", re.I)]),
    # build-system configuration (meson/cmake/configure/autotools)
    ("build_config_error",   [re.compile(r"meson\.build:\d+:\d+: ERROR:|CMake Error|configure: error:|Module \"[^\"]+\" does not exist|No CMAKE_.*could be found", re.I)]),
    # compile / link / build-tool failures
    ("compiler_error",       [re.compile(r"fatal error: .*: No such file or directory|all warnings being treated as errors|undefined reference to|cc1plus: |error: could not compile|error\[E\d+\]", re.I)]),
    ("make_error",           [re.compile(r"make(\[\d+\])?: \*\*\*|recipe for target .* failed|ninja: build stopped|Error [12]\b", re.I)]),
    # environment
    ("disk_space",           [re.compile(r"No space left on device", re.I)]),
    ("permission_error",     [re.compile(r"Permission denied|EACCES|cannot open.*permission denied", re.I)]),
    ("verification_failed",  [re.compile(r"verify.*failed|verification.*exit.*code [^0]", re.I)]),
]


def classify_failure(log_text: str) -> str:
    """Return the first matching failure mode label, or 'unknown'."""
    for label, patterns in FAILURE_PATTERNS:
        for pat in patterns:
            if pat.search(log_text):
                return label
    return "unknown"


# ---------------------------------------------------------------------------
# Token parsing
# ---------------------------------------------------------------------------

def parse_tokens_from_log(log_path: Path) -> dict:
    """Parse [TOKENS] JSON lines emitted by agents and aggregate by phase."""
    totals: dict[str, dict] = {}
    if not log_path.exists():
        return totals
    try:
        text = log_path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return totals
    for line in text.splitlines():
        idx = line.find("[TOKENS]")
        if idx == -1:
            continue
        json_part = line[idx + len("[TOKENS]"):].strip()
        try:
            rec = json.loads(json_part)
        except json.JSONDecodeError:
            continue
        phase = rec.get("phase", "unknown")
        entry = totals.setdefault(phase, {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0, "calls": 0})
        entry["prompt_tokens"] += rec.get("prompt_tokens", 0)
        entry["completion_tokens"] += rec.get("completion_tokens", 0)
        entry["total_tokens"] += rec.get("total_tokens", 0)
        entry["calls"] += 1
    return totals


def aggregate_tokens(phase_tokens: dict) -> dict:
    agg = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0, "calls": 0}
    for phase_data in phase_tokens.values():
        agg["prompt_tokens"] += phase_data["prompt_tokens"]
        agg["completion_tokens"] += phase_data["completion_tokens"]
        agg["total_tokens"] += phase_data["total_tokens"]
        agg["calls"] += phase_data["calls"]
    return agg


# ---------------------------------------------------------------------------
# Repair report helpers
# ---------------------------------------------------------------------------

def read_repair_report(workspace_root: Path, repo_url: str, reports_dir: str = "repair-reports") -> Optional[dict]:
    if not _YAML_AVAILABLE:
        return None
    repo_name = repo_url.rstrip("/").split("/")[-1].replace(".git", "")
    report_path = workspace_root / reports_dir / repo_name / "report.yaml"
    if not report_path.exists():
        return None
    try:
        with open(report_path, encoding="utf-8") as fh:
            return yaml.safe_load(fh)
    except Exception:
        return None


def detect_error_recurrence(attempts: list[dict]) -> bool:
    """True if the same non-unknown failure mode appears in ≥2 consecutive attempts."""
    modes: list[str] = []
    for attempt in attempts:
        log_path = Path(attempt.get("build_log", ""))
        if log_path.exists():
            try:
                text = log_path.read_text(encoding="utf-8", errors="replace")
                modes.append(classify_failure(text))
            except OSError:
                modes.append("unknown")
    for i in range(len(modes) - 1):
        if modes[i] == modes[i + 1] and modes[i] != "unknown":
            return True
    return False


# ---------------------------------------------------------------------------
# Dockerfile analysis
# ---------------------------------------------------------------------------

def _hadolint_warning_count(df_path: Path) -> Optional[int]:
    """Count hadolint findings on the final Dockerfile. None if hadolint is unavailable."""
    if shutil.which("hadolint") is None:
        return None
    try:
        proc = subprocess.run(
            ["hadolint", "--no-fail", "--format", "json", str(df_path)],
            capture_output=True, text=True, timeout=60,
        )
        findings = json.loads(proc.stdout or "[]")
        return len(findings) if isinstance(findings, list) else None
    except (OSError, ValueError, subprocess.SubprocessError):
        return None


def analyze_dockerfile(workspace_root: Path, repo_url: str, dockerfiles_dir: str = "dockerfiles") -> dict:
    repo_name = repo_url.rstrip("/").split("/")[-1].replace(".git", "")
    df_path = workspace_root / dockerfiles_dir / f"{repo_name}.Dockerfile"
    if not df_path.exists():
        return {}
    try:
        content = df_path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return {}
    lines = content.splitlines()
    run_layers = sum(1 for l in lines if l.strip().upper().startswith("RUN "))
    from_line = next((l.strip() for l in lines if l.strip().upper().startswith("FROM ")), "")
    base_image = ""
    if from_line:
        parts = from_line.split()
        # FROM <image> [AS <alias>]  → take index 1
        base_image = parts[1] if len(parts) >= 2 else ""
    return {
        "line_count": len(lines),
        "byte_count": len(content.encode("utf-8")),
        "run_layers": run_layers,
        "base_image": base_image,
        "hadolint_warnings": _hadolint_warning_count(df_path),
    }


# ---------------------------------------------------------------------------
# Per-repo metrics
# ---------------------------------------------------------------------------

def compute_repo_metrics(
    workspace_root: Path,
    repo_result: dict,
    reports_dir: str = "repair-reports",
    dockerfiles_dir: str = "dockerfiles",
    results_dir: str = "classification_results",
    summaries_dir: str = "summaries",
    dataset_dir: Optional[Path] = None,
    ground_truth_index: Optional[dict] = None,
) -> dict:
    repo_url = repo_result["url"]
    # A repo that failed before the pipeline ran (e.g. a checkout error) has
    # "log": None, so .get(..., "") returns None, not the default. Guard against it.
    log_path = Path(repo_result.get("log") or "")

    phase_tokens = parse_tokens_from_log(log_path)
    token_summary = aggregate_tokens(phase_tokens)

    report = read_repair_report(workspace_root, repo_url, reports_dir)
    repair_metrics: dict = {}
    if report:
        attempts = report.get("attempts", [])
        successful_attempt = report.get("successful_attempt")
        build_success = report.get("success", False)

        verify_passed = False
        for attempt in reversed(attempts):
            # Check if verification (first, retry, or deterministic fallback) passed in this attempt
            bv_retry = attempt.get("build_verification_retry")
            if bv_retry and bv_retry.get("exit_code") == 0:
                verify_passed = True
                break
            bv_fallback = attempt.get("build_verification_fallback")
            if bv_fallback and bv_fallback.get("exit_code") == 0:
                verify_passed = True
                break
            bv = attempt.get("build_verification")
            if bv and bv.get("exit_code") == 0:
                verify_passed = True
                break

        first_failure_mode = "n/a"
        for attempt in attempts:
            if attempt.get("exit_code", 1) != 0:
                log_p = Path(attempt.get("build_log", ""))
                if log_p.exists():
                    try:
                        first_failure_mode = classify_failure(log_p.read_text(encoding="utf-8", errors="replace"))
                    except OSError:
                        pass
                break

        binary_metrics_raw = report.get("binary_metrics") or {}
        # Final image size recorded on the successful-build attempt (None if pre-dating capture).
        image_size_bytes = None
        if successful_attempt is not None:
            succ = next((a for a in attempts if a.get("attempt") == successful_attempt), None)
            if succ:
                image_size_bytes = succ.get("image_size_bytes")

        # Tiered verification (TODO 1): split the formerly-conflated build_success into
        # distinct, separately-reportable tiers.
        #   build_ok    — the image built (docker build exit 0), regardless of verify.
        #   soft_verify — the agent's own verify command exited 0.
        #   hard_verify — build_ok AND the produced artifact's hash matched the GT digest,
        #                 independent of soft_verify. None = inconclusive (no GT artifact/hash).
        #                 NOTE: this is the scaffold fallback; eval.py's combined-verify
        #                 post-processing overrides hard_verify when artifact_listing + GT exist.
        build_ok = any(a.get("exit_code") == 0 for a in attempts)
        soft_verify = verify_passed
        hash_match = binary_metrics_raw.get("binary_hash_match")  # True / False / None
        hard_verify: Optional[bool] = (
            None if hash_match is None else (build_ok and hash_match is True)
        )
        # Per-artifact match rate. Single key artifact today; becomes matched/total once
        # multi-artifact hashing lands. None when no artifact was hashable.
        artifact_match_rate = (
            None if hash_match is None else (1.0 if hash_match else 0.0)
        )

        repair_metrics = {
            "build_success": build_success,
            "build_ok": build_ok,
            "soft_verify": soft_verify,
            "hard_verify": hard_verify,
            "artifact_match_rate": artifact_match_rate,
            "first_attempt_success": bool(build_success and successful_attempt == 1),
            "repair_salvaged": bool(build_success and successful_attempt is not None and successful_attempt > 1),
            "verification_passed": verify_passed,
            "total_attempts": len(attempts),
            "successful_attempt": successful_attempt,
            "first_failure_mode": first_failure_mode,
            "error_recurrence": detect_error_recurrence(attempts),
            "binary_size_plausible": binary_metrics_raw.get("binary_size_plausible"),
            "binary_hash_match": binary_metrics_raw.get("binary_hash_match"),
            "image_size_bytes": image_size_bytes,
        }

    dockerfile_metrics = analyze_dockerfile(workspace_root, repo_url, dockerfiles_dir)

    # Ground-truth / retrieval / package scoring all parse the agent's free-form
    # classification YAML, whose schema can drift (e.g. `categories` emitted as a list
    # instead of a dict). A parse crash here must NOT take down build/repair metrics —
    # those are computed from report.yaml and are the run's source of truth. Each scoring
    # block degrades to None on failure and logs, so a drifted repo loses only its
    # research scores, not its build_success.

    # Ground truth comparison (only if dataset_dir provided)
    gt_comparison: Optional[dict] = None
    if dataset_dir is not None:
        if ground_truth_index is None:
            ground_truth_index = build_ground_truth_index(dataset_dir)
        try:
            gt_comparison = compare_with_ground_truth(
                workspace_root=workspace_root,
                repo_url=repo_url,
                dataset_dir=dataset_dir,
                ground_truth_index=ground_truth_index,
                results_dir=results_dir,
                dockerfiles_dir=dockerfiles_dir,
            )
        except Exception as exc:
            print(f"[metrics] WARN ground-truth scoring failed for {repo_url}: "
                  f"{type(exc).__name__}: {exc}", file=sys.stderr)

    # Retrieval quality: L1 selected files vs curated gold set (only when both exist).
    retrieval_quality = None
    package_quality = None
    if dataset_dir is not None:
        try:
            gt_doc = load_gt_for_repo(dataset_dir, repo_url)
            gold = get_gt_install_relevant_files(gt_doc)
            predicted = read_selected_files(workspace_root, repo_url, summaries_dir)
            strategy = read_retrieval_strategy(workspace_root, repo_url, summaries_dir)
            # one_shot_fingerprint deliberately selects no files (it feeds the full repo
            # fingerprint as context), so per-file precision/recall is N/A — scoring its empty
            # selection as 0 would unfairly sink it against strategies that do select.
            if strategy == "one_shot_fingerprint":
                retrieval_quality = None
            elif gold and predicted is not None:
                retrieval_quality = compute_retrieval_quality(predicted, gold)
            # Package classification: predicted system packages vs GT packages (any manager).
            gold_pkgs = get_gt_packages(gt_doc)
            pred_pkgs = read_predicted_packages(workspace_root, repo_url, results_dir)
            if gold_pkgs and pred_pkgs is not None:
                package_quality = compute_package_quality(pred_pkgs, gold_pkgs)
        except Exception as exc:
            print(f"[metrics] WARN retrieval/package scoring failed for {repo_url}: "
                  f"{type(exc).__name__}: {exc}", file=sys.stderr)

    return {
        "tokens": {
            "by_phase": phase_tokens,
            "total": token_summary,
        },
        "repair": repair_metrics,
        "dockerfile": dockerfile_metrics,
        "ground_truth_comparison": gt_comparison,
        "retrieval_quality": retrieval_quality,
        "package_quality": package_quality,
    }


# ---------------------------------------------------------------------------
# Aggregate metrics
# ---------------------------------------------------------------------------

def load_model_pricing(workspace_root: Optional[Path] = None) -> dict:
    """Load config/model_pricing.json. Returns {} if missing/invalid (cost stays unpriced)."""
    root = workspace_root or Path.cwd()
    path = Path(root) / "config" / "model_pricing.json"
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data.get("models", {}) if isinstance(data, dict) else {}
    except (json.JSONDecodeError, OSError):
        return {}


def compute_cost_usd(total_tokens: dict, model: Optional[str], pricing: Optional[dict]) -> dict:
    """Estimate dollar cost from recorded tokens. Unknown/unpriced models -> priced=false."""
    entry = (pricing or {}).get(model or "")
    in_price = entry.get("input_per_mtok") if entry else None
    out_price = entry.get("output_per_mtok") if entry else None
    if in_price is None or out_price is None:
        return {"priced": False, "model": model}
    prompt = total_tokens.get("prompt_tokens", 0)
    completion = total_tokens.get("completion_tokens", 0)
    input_usd = prompt / 1_000_000 * in_price
    output_usd = completion / 1_000_000 * out_price
    return {
        "priced": True,
        "model": model,
        "input_usd": round(input_usd, 4),
        "output_usd": round(output_usd, 4),
        "total_usd": round(input_usd + output_usd, 4),
    }


# ---------------------------------------------------------------------------
# Retrieval quality (Stage 1 L1 file selection vs curated gold set)
# ---------------------------------------------------------------------------

def _norm_path(p: str) -> str:
    return p.strip().lstrip("./").replace("\\", "/")


def get_gt_install_relevant_files(gt_doc: Optional[dict]) -> list[str]:
    """Curated gold set: categories.install_relevant_files.value (empty if absent)."""
    if not gt_doc:
        return []
    v = (gt_doc.get("categories") or {}).get("install_relevant_files") or {}
    val = v.get("value") if isinstance(v, dict) else None
    return val if isinstance(val, list) else []


def read_selected_files(workspace_root: Path, repo_url: str, summaries_dir: str = "summaries") -> Optional[list[str]]:
    """Read the L1 prediction artifact: {repo}.selected-files.yaml -> selected_files."""
    repo_name = repo_url.rstrip("/").split("/")[-1].replace(".git", "")
    doc = _load_yaml(workspace_root / summaries_dir / f"{repo_name}.selected-files.yaml")
    if not doc:
        return None
    sf = doc.get("selected_files")
    return sf if isinstance(sf, list) else None


def read_retrieval_strategy(workspace_root: Path, repo_url: str, summaries_dir: str = "summaries") -> Optional[str]:
    """Read which retrieval strategy produced the selected-files artifact."""
    repo_name = repo_url.rstrip("/").split("/")[-1].replace(".git", "")
    doc = _load_yaml(workspace_root / summaries_dir / f"{repo_name}.selected-files.yaml")
    if not doc:
        return None
    s = doc.get("retrieval_strategy")
    return str(s) if s else None


def compute_retrieval_quality(predicted: list[str], gold: list[str]) -> Optional[dict]:
    """Glob-aware precision/recall/F1 of L1-selected files vs the curated gold set.

    `predicted` entries may be concrete paths or globs (e.g. '.github/workflows/*.yml');
    `gold` is the curated concrete file list. A gold file is covered (recall) if some
    predicted entry equals or glob-matches it; a predicted entry is a hit (precision) if
    it matches at least one gold file. Returns None when no gold set exists for the repo.
    """
    gold_set = {_norm_path(g) for g in gold if isinstance(g, str) and g.strip()}
    if not gold_set:
        return None
    pred_unique = list(dict.fromkeys(_norm_path(p) for p in predicted if isinstance(p, str) and p.strip()))

    def matches(p: str, g: str) -> bool:
        return p == g or fnmatch.fnmatch(g, p)

    covered = {g for g in gold_set if any(matches(p, g) for p in pred_unique)}
    hits = [p for p in pred_unique if any(matches(p, g) for g in gold_set)]

    precision = round(len(hits) / len(pred_unique), 4) if pred_unique else 0.0
    recall = round(len(covered) / len(gold_set), 4)
    f1 = round(2 * precision * recall / (precision + recall), 4) if (precision + recall) else 0.0
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "predicted_count": len(pred_unique),
        "gold_count": len(gold_set),
        "matched_gold": len(covered),   # gold files a prediction covers  (TP, recall side)
        "matched_pred": len(hits),       # predictions that hit some gold   (TP, precision side)
        "missed_gold": sorted(gold_set - covered),
    }


def get_gt_packages(gt_doc: Optional[dict]) -> list[str]:
    """Ground-truth package names from `categories.packages.value`.

    GT entries are ``name:manager`` (e.g. ``libssl-dev:apt``, ``pnpm:npm``, ``rustup:sh``)
    spanning every package manager. We score on the package NAME (manager stripped),
    lower-cased, since that is the classification target."""
    if not isinstance(gt_doc, dict):
        return []
    cats = gt_doc.get("categories")
    val = (cats.get("packages") or {}).get("value") if isinstance(cats, dict) else None
    names: list[str] = []
    if isinstance(val, list):
        for entry in val:
            if isinstance(entry, str) and entry.strip():
                name = entry.split(":", 1)[0].strip().lower()
                if name:
                    names.append(name)
    return names


def read_predicted_packages(workspace_root: Path, repo_url: str, results_dir: str) -> Optional[list[str]]:
    """Agent-predicted install packages = names under classification `system_dependencies`.

    These are the system packages the agent says to install (apt/pip/etc.), the analogue of
    GT `packages`. Project library dependencies (`dependencies_packages`, e.g. cargo crates)
    are deliberately excluded — they are resolved by the build tool, not installed, and are
    not part of GT packages. Returns None when no classification result exists for the repo."""
    repo_name = repo_url.rstrip("/").split("/")[-1].replace(".git", "")
    path = workspace_root / results_dir / f"{repo_name}.yaml"
    if not path.exists():
        results_root = workspace_root / results_dir
        target = f"{repo_name.lower()}.yaml"
        match = next((c for c in results_root.iterdir() if c.name.lower() == target), None) if results_root.exists() else None
        if match is None:
            return None
        path = match
    try:
        doc = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError):
        return None
    if not isinstance(doc, dict):
        return None
    # Prefer a GT-style categories.system_dependencies.value if the agent emitted one;
    # otherwise read the flat schema (dependencies.{build,runtime,system}).
    cats = doc.get("categories")
    if isinstance(cats, dict):
        sysdeps = (cats.get("system_dependencies") or {}).get("value")
        names: list[str] = []
        if isinstance(sysdeps, list):
            for item in sysdeps:
                if isinstance(item, dict):
                    name = str(item.get("name", "")).strip().lower()
                elif isinstance(item, str):
                    name = item.strip().lower()
                else:
                    name = ""
                if name:
                    names.append(name)
        if names:
            return names
    return _pred_system_packages(doc)


def compute_package_quality(predicted: list[str], gold: list[str]) -> Optional[dict]:
    """Precision/recall/F1 of predicted vs ground-truth package names (set comparison).
    Returns None when no gold package set exists for the repo."""
    gold_set = {g for g in gold if isinstance(g, str) and g.strip()}
    if not gold_set:
        return None
    pred_set = {p for p in predicted if isinstance(p, str) and p.strip()}
    matched = pred_set & gold_set
    precision = round(len(matched) / len(pred_set), 4) if pred_set else 0.0
    recall = round(len(matched) / len(gold_set), 4)
    f1 = round(2 * precision * recall / (precision + recall), 4) if (precision + recall) else 0.0
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "predicted_count": len(pred_set),
        "gold_count": len(gold_set),
        "matched_gold": len(matched),
        "missed_gold": sorted(gold_set - pred_set),
    }


def _percentile(sorted_values: list[float], pct: float) -> float:
    """Linear-interpolation percentile over an already-sorted, non-empty list."""
    if not sorted_values:
        return 0.0
    if len(sorted_values) == 1:
        return float(sorted_values[0])
    rank = (pct / 100.0) * (len(sorted_values) - 1)
    lo = int(rank)
    hi = min(lo + 1, len(sorted_values) - 1)
    frac = rank - lo
    return sorted_values[lo] * (1 - frac) + sorted_values[hi] * frac


def compute_aggregate_metrics(
    results: list[dict],
    model: Optional[str] = None,
    pricing: Optional[dict] = None,
) -> dict:
    total = len(results)
    if total == 0:
        return {}

    completed_pipelines = [r for r in results if r.get("status") == "success"]
    build_success = [r for r in results if r.get("metrics", {}).get("repair", {}).get("build_success")]
    first_attempt = [r for r in results if r.get("metrics", {}).get("repair", {}).get("first_attempt_success")]
    repair_salvaged = [r for r in results if r.get("metrics", {}).get("repair", {}).get("repair_salvaged")]
    verify_passed = [r for r in results if r.get("metrics", {}).get("repair", {}).get("verification_passed")]
    # Tiered verification (TODO 1): build_ok and soft_verify span all repos; hard_verify is
    # rated only over repos where it was conclusive (a GT artifact hash was available).
    build_ok = [r for r in results if r.get("metrics", {}).get("repair", {}).get("build_ok")]
    soft_verify = [r for r in results if r.get("metrics", {}).get("repair", {}).get("soft_verify")]
    hard_applicable = [r for r in results if r.get("metrics", {}).get("repair", {}).get("hard_verify") is not None]
    hard_pass = [r for r in hard_applicable if r["metrics"]["repair"]["hard_verify"] is True]
    # MID verify (LLM judge): rated only over repos with a conclusive verdict (legit not None).
    mid_applicable = [r for r in results if isinstance(r.get("metrics", {}).get("repair", {}).get("mid_verify"), dict) and r["metrics"]["repair"]["mid_verify"].get("legit") is not None]
    mid_legit = [r for r in mid_applicable if r["metrics"]["repair"]["mid_verify"]["legit"] is True]
    # Verify-command similarity (TODO 1): rated over repos with a conclusive score (not None).
    sim_applicable = [
        r for r in results
        if isinstance(r.get("metrics", {}).get("repair", {}).get("verify_cmd_similarity"), dict)
        and r["metrics"]["repair"]["verify_cmd_similarity"].get("score") is not None
    ]
    sim_scores = [r["metrics"]["repair"]["verify_cmd_similarity"]["score"] for r in sim_applicable]
    sim_categories: dict[str, int] = defaultdict(int)
    for r in sim_applicable:
        cat = r["metrics"]["repair"]["verify_cmd_similarity"].get("category")
        if cat:
            sim_categories[cat] += 1
    # Build-steps similarity (replaces the exact-match build_steps Jaccard): rated over
    # repos with a conclusive score (not None), same shape as verify_cmd_similarity.
    build_sim_applicable = [
        r for r in results
        if isinstance(r.get("metrics", {}).get("repair", {}).get("build_steps_similarity"), dict)
        and r["metrics"]["repair"]["build_steps_similarity"].get("score") is not None
    ]
    build_sim_scores = [r["metrics"]["repair"]["build_steps_similarity"]["score"] for r in build_sim_applicable]
    build_sim_categories: dict[str, int] = defaultdict(int)
    for r in build_sim_applicable:
        cat = r["metrics"]["repair"]["build_steps_similarity"].get("category")
        if cat:
            build_sim_categories[cat] += 1
    # binary_size_plausible: only count repos where GT binary info was available (not None)
    binary_plausible_applicable = [r for r in results if r.get("metrics", {}).get("repair", {}).get("binary_size_plausible") is not None]
    binary_plausible_pass = [r for r in binary_plausible_applicable if r["metrics"]["repair"]["binary_size_plausible"] is True]

    total_tokens: dict[str, int] = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0, "calls": 0}
    for r in results:
        t = r.get("metrics", {}).get("tokens", {}).get("total", {})
        for k in total_tokens:
            total_tokens[k] += t.get(k, 0)

    attempt_histogram: dict[int, int] = defaultdict(int)
    for r in results:
        sa = r.get("metrics", {}).get("repair", {}).get("successful_attempt")
        if sa is not None:
            attempt_histogram[sa] += 1

    failure_modes: dict[str, int] = defaultdict(int)
    for r in results:
        mode = r.get("metrics", {}).get("repair", {}).get("first_failure_mode", "")
        if mode and mode not in ("n/a", ""):
            failure_modes[mode] += 1

    by_language: dict[str, dict] = {}
    for r in results:
        lang = r.get("language", "unknown")
        entry = by_language.setdefault(lang, {"total": 0, "build_success": 0, "first_attempt_success": 0})
        entry["total"] += 1
        if r.get("metrics", {}).get("repair", {}).get("build_success"):
            entry["build_success"] += 1
        if r.get("metrics", {}).get("repair", {}).get("first_attempt_success"):
            entry["first_attempt_success"] += 1
    for entry in by_language.values():
        entry["success_rate"] = round(entry["build_success"] / entry["total"], 4) if entry["total"] else 0.0

    by_complexity: dict[int, dict] = {}
    for r in results:
        cplx = r.get("complexity") or 0
        entry = by_complexity.setdefault(cplx, {"total": 0, "build_success": 0, "first_attempt_success": 0})
        entry["total"] += 1
        if r.get("metrics", {}).get("repair", {}).get("build_success"):
            entry["build_success"] += 1
        if r.get("metrics", {}).get("repair", {}).get("first_attempt_success"):
            entry["first_attempt_success"] += 1
    for entry in by_complexity.values():
        entry["success_rate"] = round(entry["build_success"] / entry["total"], 4) if entry["total"] else 0.0

    df_lines = [r.get("metrics", {}).get("dockerfile", {}).get("line_count") for r in results]
    df_lines_valid = [v for v in df_lines if v is not None]
    df_runs = [r.get("metrics", {}).get("dockerfile", {}).get("run_layers") for r in results]
    df_runs_valid = [v for v in df_runs if v is not None]
    hado = [r.get("metrics", {}).get("dockerfile", {}).get("hadolint_warnings") for r in results]
    hado_valid = [v for v in hado if v is not None]

    # Final image size (bytes), recorded on successful builds.
    img_sizes = sorted(
        v for v in (r.get("metrics", {}).get("repair", {}).get("image_size_bytes") for r in results)
        if isinstance(v, int) and v > 0
    )
    image_size = {
        "builds_measured": len(img_sizes),
        "mean_mb": round(sum(img_sizes) / len(img_sizes) / 1_048_576, 1),
        "median_mb": round(_percentile(img_sizes, 50) / 1_048_576, 1),
        "p90_mb": round(_percentile(img_sizes, 90) / 1_048_576, 1),
        "max_mb": round(img_sizes[-1] / 1_048_576, 1),
    } if img_sizes else {}

    # Wall-clock: per-repo durations recorded by eval.py. Skipped repos have 0.0,
    # so a positive-value filter keeps only repos that actually executed.
    durations = sorted(
        d for d in (r.get("duration_seconds") for r in results)
        if isinstance(d, (int, float)) and d > 0
    )
    wall_clock = {
        "repos_timed": len(durations),
        "total_seconds": round(sum(durations), 1),
        "mean_seconds": round(sum(durations) / len(durations), 1),
        "median_seconds": round(_percentile(durations, 50), 1),
        "p90_seconds": round(_percentile(durations, 90), 1),
        "min_seconds": round(durations[0], 1),
        "max_seconds": round(durations[-1], 1),
    } if durations else {}

    # Cost efficiency: total spend per *successful* build — the Pareto axis that pairs
    # quality (build_success_rate) with cost (tokens + wall-clock).
    n_build = len(build_success)
    n_verify = len(verify_passed)
    total_tok = total_tokens["total_tokens"]
    wc_total = sum(durations) if durations else None
    cost_efficiency = {
        "tokens_per_successful_build": round(total_tok / n_build) if n_build else None,
        "tokens_per_verified_build": round(total_tok / n_verify) if n_verify else None,
        "wall_clock_seconds_per_successful_build": (
            round(wc_total / n_build, 1) if (n_build and wc_total is not None) else None
        ),
    }

    # Dollar cost (estimate; unpriced models are reported as priced=false).
    cost_usd = compute_cost_usd(total_tokens, model, pricing)
    if cost_usd.get("priced") and n_build:
        cost_efficiency["usd_per_successful_build"] = round(cost_usd["total_usd"] / n_build, 4)

    return {
        "total_repos": total,
        "pipeline_completed_rate": round(len(completed_pipelines) / total, 4),
        "build_success_rate": round(len(build_success) / total, 4),
        "build_ok_rate": round(len(build_ok) / total, 4),
        "soft_verify_rate": round(len(soft_verify) / total, 4),
        "hard_verify_rate": round(len(hard_pass) / len(hard_applicable), 4) if hard_applicable else None,
        "hard_verify_applicable": len(hard_applicable),
        "mid_verify_legit_rate": round(len(mid_legit) / len(mid_applicable), 4) if mid_applicable else None,
        "mid_verify_applicable": len(mid_applicable),
        "verify_cmd_similarity_applicable": len(sim_applicable),
        "verify_cmd_similarity_mean_score": round(sum(sim_scores) / len(sim_scores), 4) if sim_scores else None,
        "verify_cmd_similarity_categories": dict(sim_categories),
        "build_steps_similarity_applicable": len(build_sim_applicable),
        "build_steps_similarity_mean_score": round(sum(build_sim_scores) / len(build_sim_scores), 4) if build_sim_scores else None,
        "build_steps_similarity_categories": dict(build_sim_categories),
        "first_attempt_success_rate": round(len(first_attempt) / total, 4),
        "repair_salvage_rate": round(len(repair_salvaged) / total, 4),
        "verification_pass_rate": round(len(verify_passed) / total, 4),
        "binary_size_plausible_rate": round(len(binary_plausible_pass) / len(binary_plausible_applicable), 4) if binary_plausible_applicable else None,
        "total_tokens": total_tokens,
        "avg_tokens_per_repo": {k: round(total_tokens[k] / total) for k in total_tokens} if total else {},
        "wall_clock_seconds": wall_clock,
        "cost_efficiency": cost_efficiency,
        "cost_usd": cost_usd,
        "attempts_to_success_histogram": dict(sorted(attempt_histogram.items())),
        "failure_mode_distribution": dict(sorted(failure_modes.items(), key=lambda x: -x[1])),
        "by_language": dict(sorted(by_language.items())),
        "by_complexity": {str(k): v for k, v in sorted(by_complexity.items())},
        "dockerfile_stats": {
            "line_count": {"mean": round(sum(df_lines_valid) / len(df_lines_valid), 1), "min": min(df_lines_valid), "max": max(df_lines_valid)} if df_lines_valid else {},
            "run_layers": {"mean": round(sum(df_runs_valid) / len(df_runs_valid), 1), "min": min(df_runs_valid), "max": max(df_runs_valid)} if df_runs_valid else {},
            "hadolint_warnings": {"dockerfiles_linted": len(hado_valid), "mean": round(sum(hado_valid) / len(hado_valid), 1), "min": min(hado_valid), "max": max(hado_valid)} if hado_valid else {},
        },
        "image_size": image_size,
        "ground_truth_scores": _aggregate_gt_scores(results),
        "retrieval_quality": _aggregate_retrieval_quality(results),
        "package_quality": _aggregate_package_quality(results),
    }


def _aggregate_retrieval_quality(results: list[dict]) -> dict:
    """Mean precision/recall/F1 of L1 file selection over repos with a curated gold set."""
    rqs = [
        r["metrics"]["retrieval_quality"]
        for r in results
        if r.get("metrics", {}).get("retrieval_quality")
    ]
    if not rqs:
        return {"available": 0}
    n = len(rqs)
    return {
        "available": n,
        "mean_precision": round(sum(q["precision"] for q in rqs) / n, 4),
        "mean_recall": round(sum(q["recall"] for q in rqs) / n, 4),
        "mean_f1": round(sum(q["f1"] for q in rqs) / n, 4),
    }


def _aggregate_package_quality(results: list[dict]) -> dict:
    """Mean precision/recall/F1 of package classification over repos with a GT package set."""
    pqs = [
        r["metrics"]["package_quality"]
        for r in results
        if r.get("metrics", {}).get("package_quality")
    ]
    if not pqs:
        return {"available": 0}
    n = len(pqs)
    return {
        "available": n,
        "mean_precision": round(sum(q["precision"] for q in pqs) / n, 4),
        "mean_recall": round(sum(q["recall"] for q in pqs) / n, 4),
        "mean_f1": round(sum(q["f1"] for q in pqs) / n, 4),
    }


def _aggregate_gt_scores(results: list[dict]) -> dict:
    """Aggregate ground truth comparison scores across all repos."""
    overall_scores = [
        r["metrics"]["ground_truth_comparison"]["overall_score"]
        for r in results
        if r.get("metrics", {}).get("ground_truth_comparison") is not None
        and r["metrics"]["ground_truth_comparison"].get("overall_score") is not None
    ]
    if not overall_scores:
        return {
            "available": 0,
            "mean_overall_score": None,
            "per_field": {},
            "apt_packages": {"available": 0, "mean_recall": None, "mean_precision": None},
        }

    # Per-field means. Mirrors the "only non-empty GT counts" rule compare_with_ground_truth
    # applies to overall_score (§2.4) — without it, a field the dataset never populates for
    # ANY repo (a GT-schema mismatch, not a real miss) silently reports a misleading 0.0
    # mean instead of being absent, and a legitimately-empty-for-this-repo GT (e.g. no extra
    # apt packages needed) would count as a miss instead of being skipped.
    field_accum: dict[str, list[float]] = defaultdict(list)
    for r in results:
        gt = r.get("metrics", {}).get("ground_truth_comparison")
        if not gt:
            continue
        for field, info in gt.get("field_scores", {}).items():
            if not info.get("ground_truth"):
                continue
            # set + apt_packages modes carry "jaccard"; exact carries "score".
            score = info.get("jaccard", info.get("score"))
            if score is not None:
                field_accum[field].append(score)

    per_field = {
        field: round(sum(scores) / len(scores), 4)
        for field, scores in sorted(field_accum.items())
    }

    # Headline apt-package install fidelity: did the agent install the apt packages
    # the ground truth requires (recall) without spraying extras (precision)?
    apt_recall = [
        info["recall"]
        for r in results
        if (gt := r.get("metrics", {}).get("ground_truth_comparison"))
        for info in [gt.get("field_scores", {}).get("packages", {})]
        if info.get("recall") is not None
    ]
    apt_precision = [
        info["precision"]
        for r in results
        if (gt := r.get("metrics", {}).get("ground_truth_comparison"))
        for info in [gt.get("field_scores", {}).get("packages", {})]
        if info.get("precision") is not None
    ]
    apt_packages = {
        "available": len(apt_recall),
        "mean_recall": round(sum(apt_recall) / len(apt_recall), 4) if apt_recall else None,
        "mean_precision": round(sum(apt_precision) / len(apt_precision), 4) if apt_precision else None,
    }

    return {
        "available": len(overall_scores),
        "mean_overall_score": round(sum(overall_scores) / len(overall_scores), 4),
        "per_field_mean": per_field,
        "apt_packages": apt_packages,
    }


# ---------------------------------------------------------------------------
# Regression delta
# ---------------------------------------------------------------------------

def compute_regression_delta(current_results: list[dict], prior_path: Path) -> Optional[dict]:
    if not prior_path.exists():
        return None
    try:
        with open(prior_path, encoding="utf-8") as fh:
            prior = json.load(fh)
    except Exception:
        return None

    prior_by_url = {r["url"]: r for r in prior.get("results", [])}

    regressions: list[str] = []
    improvements: list[str] = []
    unchanged = 0

    for curr in current_results:
        url = curr["url"]
        prev = prior_by_url.get(url)
        if prev is None:
            continue
        curr_ok = curr.get("metrics", {}).get("repair", {}).get("build_success", False)
        prev_ok = prev.get("metrics", {}).get("repair", {}).get("build_success", False)
        if curr_ok == prev_ok:
            unchanged += 1
        elif not curr_ok and prev_ok:
            regressions.append(url)
        else:
            improvements.append(url)

    prior_agg = prior.get("aggregate_metrics", {})
    prior_build_rate = prior_agg.get("build_success_rate")
    curr_build_ok = sum(1 for r in current_results if r.get("metrics", {}).get("repair", {}).get("build_success"))
    curr_build_rate = round(curr_build_ok / len(current_results), 4) if current_results else 0.0

    # Flip rate = fraction of matched repos whose build outcome changed in either
    # direction. Against an identical-config prior (e.g. the AB-26 confirmation rerun
    # vs the champion's first run) this is the non-determinism floor: it tells you how
    # large a build_success_rate delta must be before it can be read as signal rather
    # than run-to-run noise.
    matched = unchanged + len(regressions) + len(improvements)
    flipped = len(regressions) + len(improvements)

    return {
        "prior_eval": str(prior_path),
        "prior_run_id": prior.get("run_id"),
        "build_success_rate_delta": round(curr_build_rate - prior_build_rate, 4) if prior_build_rate is not None else None,
        "regressions": regressions,
        "improvements": improvements,
        "unchanged_count": unchanged,
        "matched_count": matched,
        "flipped_count": flipped,
        "flip_rate": round(flipped / matched, 4) if matched else None,
    }
