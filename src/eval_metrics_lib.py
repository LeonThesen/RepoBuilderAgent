"""
eval_metrics_lib.py – Shared metrics computation for eval.py and eval_metrics.py.

Import this module instead of duplicating logic between the two scripts.
"""

from __future__ import annotations

import json
import re
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
_COMPARABLE_FIELDS: list[tuple[str, str]] = [
    ("programming_language", "set"),
    ("build_tool",           "set"),
    ("system_dependencies",  "set"),
    ("build_steps",          "set"),
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
    cats = doc.get("categories", {})
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

    # FROM line
    from_line = next((l.strip() for l in lines if l.strip().upper().startswith("FROM ")), "")
    parts = from_line.split()
    base_image = parts[1] if len(parts) >= 2 else ""

    # System packages from apt-get install lines
    apt_packages: list[str] = []
    _apt_re = re.compile(r"apt(?:-get)?\s+install\s+(?:-[^\s]+\s+)*(.+?)(?:\\|$)", re.I)
    _apk_re = re.compile(r"apk\s+add\s+(?:--[^\s]+\s+)*(.+?)(?:\\|$)", re.I)
    full_text = content
    for m in _apt_re.finditer(full_text):
        apt_packages.extend(t for t in m.group(1).split() if not t.startswith("-"))
    for m in _apk_re.finditer(full_text):
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
        pred_val = _get_field_value(prediction, field)

        if mode == "set":
            gt_set = _normalise_list(gt_val)
            pred_set = _normalise_list(pred_val)

            # Augment prediction with observed Dockerfile facts for relevant fields
            if field == "system_dependencies" and observed.get("system_packages_installed"):
                pred_set = pred_set | set(observed["system_packages_installed"])
            if field == "build_steps" and observed.get("build_run_commands"):
                # Normalise observed commands into the pred set
                pred_set = pred_set | _normalise_list(observed["build_run_commands"])

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


FAILURE_PATTERNS = [
    ("wrong_base_image",    [re.compile(r"FROM .* not found|manifest for .* not found|pull access denied", re.I)]),
    ("permission_error",    [re.compile(r"Permission denied|EACCES|cannot open.*permission denied", re.I)]),
    ("dockerfile_syntax",   [re.compile(r"dockerfile parse error|unknown instruction|syntax error", re.I)]),
    ("verification_failed", [re.compile(r"verify.*failed|verification.*exit.*code [^0]", re.I)]),
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
    dataset_dir: Optional[Path] = None,
    ground_truth_index: Optional[dict] = None,
) -> dict:
    repo_url = repo_result["url"]
    log_path = Path(repo_result.get("log", ""))

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
            bv = attempt.get("build_verification")
            if bv:
                verify_passed = bv.get("exit_code") == 0
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

        repair_metrics = {
            "build_success": build_success,
            "first_attempt_success": bool(build_success and successful_attempt == 1),
            "repair_salvaged": bool(build_success and successful_attempt is not None and successful_attempt > 1),
            "verification_passed": verify_passed,
            "total_attempts": len(attempts),
            "successful_attempt": successful_attempt,
            "first_failure_mode": first_failure_mode,
            "error_recurrence": detect_error_recurrence(attempts),
        }

    dockerfile_metrics = analyze_dockerfile(workspace_root, repo_url, dockerfiles_dir)

    # Ground truth comparison (only if dataset_dir provided)
    gt_comparison: Optional[dict] = None
    if dataset_dir is not None:
        if ground_truth_index is None:
            ground_truth_index = build_ground_truth_index(dataset_dir)
        gt_comparison = compare_with_ground_truth(
            workspace_root=workspace_root,
            repo_url=repo_url,
            dataset_dir=dataset_dir,
            ground_truth_index=ground_truth_index,
            results_dir=results_dir,
            dockerfiles_dir=dockerfiles_dir,
        )

    return {
        "tokens": {
            "by_phase": phase_tokens,
            "total": token_summary,
        },
        "repair": repair_metrics,
        "dockerfile": dockerfile_metrics,
        "ground_truth_comparison": gt_comparison,
    }


# ---------------------------------------------------------------------------
# Aggregate metrics
# ---------------------------------------------------------------------------

def compute_aggregate_metrics(results: list[dict]) -> dict:
    total = len(results)
    if total == 0:
        return {}

    completed_pipelines = [r for r in results if r.get("status") == "success"]
    build_success = [r for r in results if r.get("metrics", {}).get("repair", {}).get("build_success")]
    first_attempt = [r for r in results if r.get("metrics", {}).get("repair", {}).get("first_attempt_success")]
    repair_salvaged = [r for r in results if r.get("metrics", {}).get("repair", {}).get("repair_salvaged")]
    verify_passed = [r for r in results if r.get("metrics", {}).get("repair", {}).get("verification_passed")]

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

    return {
        "total_repos": total,
        "pipeline_completed_rate": round(len(completed_pipelines) / total, 4),
        "build_success_rate": round(len(build_success) / total, 4),
        "first_attempt_success_rate": round(len(first_attempt) / total, 4),
        "repair_salvage_rate": round(len(repair_salvaged) / total, 4),
        "verification_pass_rate": round(len(verify_passed) / total, 4),
        "total_tokens": total_tokens,
        "avg_tokens_per_repo": {k: round(total_tokens[k] / total) for k in total_tokens} if total else {},
        "attempts_to_success_histogram": dict(sorted(attempt_histogram.items())),
        "failure_mode_distribution": dict(sorted(failure_modes.items(), key=lambda x: -x[1])),
        "by_language": dict(sorted(by_language.items())),
        "by_complexity": {str(k): v for k, v in sorted(by_complexity.items())},
        "dockerfile_stats": {
            "line_count": {"mean": round(sum(df_lines_valid) / len(df_lines_valid), 1), "min": min(df_lines_valid), "max": max(df_lines_valid)} if df_lines_valid else {},
            "run_layers": {"mean": round(sum(df_runs_valid) / len(df_runs_valid), 1), "min": min(df_runs_valid), "max": max(df_runs_valid)} if df_runs_valid else {},
        },
        "ground_truth_scores": _aggregate_gt_scores(results),
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
        return {"available": 0, "mean_overall_score": None, "per_field": {}}

    # Per-field means
    field_accum: dict[str, list[float]] = defaultdict(list)
    for r in results:
        gt = r.get("metrics", {}).get("ground_truth_comparison")
        if not gt:
            continue
        for field, info in gt.get("field_scores", {}).items():
            score = info.get("jaccard") if info.get("mode") == "set" else info.get("score")
            if score is not None:
                field_accum[field].append(score)

    per_field = {
        field: round(sum(scores) / len(scores), 4)
        for field, scores in sorted(field_accum.items())
    }

    return {
        "available": len(overall_scores),
        "mean_overall_score": round(sum(overall_scores) / len(overall_scores), 4),
        "per_field_mean": per_field,
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

    return {
        "prior_eval": str(prior_path),
        "prior_run_id": prior.get("run_id"),
        "build_success_rate_delta": round(curr_build_rate - prior_build_rate, 4) if prior_build_rate is not None else None,
        "regressions": regressions,
        "improvements": improvements,
        "unchanged_count": unchanged,
    }
