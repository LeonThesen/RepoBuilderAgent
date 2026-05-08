#!/usr/bin/env python3
"""Parse generated dataset outputs into aggregate analyses.

This script reads:
- classification_results/*.yaml (classification outputs)
- classification_results/*.token-metrics.yaml (token accounting)
- summaries/*.selected-files.yaml (selected file paths)

And writes three analysis files:
- analysis/results-analysis.yaml
- analysis/token-analysis.yaml
- analysis/selected-files-analysis.yaml
"""

import argparse
import json
from collections import Counter
from pathlib import Path
from statistics import median
from typing import Any

from jinja2 import Environment, select_autoescape
import yaml


def load_yaml(path: Path) -> tuple[Any, str | None]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f), None
    except Exception as exc:
        return None, str(exc)


def as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def nested_get(data: dict[str, Any], keys: list[str], default: Any = None) -> Any:
    cur: Any = data
    for key in keys:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def top_items(counter: Counter, limit: int) -> list[dict[str, Any]]:
    return [{"name": name, "count": count} for name, count in counter.most_common(limit)]


def analyze_classification_results(results_dir: Path, top_k: int) -> dict[str, Any]:
    files = sorted(results_dir.glob("*.yaml"))
    class_files = [p for p in files if not p.name.endswith(".token-metrics.yaml")]

    parse_errors = []
    schema_errors = 0
    processed = 0

    language_counter: Counter[str] = Counter()
    build_tool_counter: Counter[str] = Counter()
    runtime_counter: Counter[str] = Counter()
    os_counter: Counter[str] = Counter()
    confidence_sums: dict[str, float] = {}
    confidence_counts: dict[str, int] = {}

    for path in class_files:
        data, err = load_yaml(path)
        if err:
            parse_errors.append({"file": path.name, "error": err})
            continue
        if not isinstance(data, dict):
            schema_errors += 1
            continue

        if data.get("error"):
            # Keep track of model-output parse fallback docs.
            schema_errors += 1
            continue

        categories = data.get("categories")
        if not isinstance(categories, dict):
            schema_errors += 1
            continue

        processed += 1

        languages = as_list(nested_get(categories, ["programming_language", "value"], []))
        build_tools = as_list(nested_get(categories, ["build_tool", "value"], []))
        runtimes = as_list(nested_get(categories, ["runtime_environment", "value"], []))
        os_values = as_list(nested_get(categories, ["os_compatibility", "value"], []))

        language_counter.update(str(x) for x in languages if x)
        build_tool_counter.update(str(x) for x in build_tools if x)
        runtime_counter.update(str(x) for x in runtimes if x)
        os_counter.update(str(x) for x in os_values if x)

        conf_scores = nested_get(categories, ["confidence_scores", "value"], {})
        if isinstance(conf_scores, dict):
            for cat, score in conf_scores.items():
                try:
                    val = float(score)
                except (TypeError, ValueError):
                    continue
                confidence_sums[cat] = confidence_sums.get(cat, 0.0) + val
                confidence_counts[cat] = confidence_counts.get(cat, 0) + 1

    avg_confidence_by_category = {
        cat: round(confidence_sums[cat] / confidence_counts[cat], 4)
        for cat in confidence_sums
        if confidence_counts.get(cat, 0) > 0
    }

    return {
        "total_result_files": len(class_files),
        "processed_result_files": processed,
        "schema_or_error_result_files": schema_errors,
        "yaml_parse_errors": len(parse_errors),
        "top_programming_languages": top_items(language_counter, top_k),
        "top_build_tools": top_items(build_tool_counter, top_k),
        "top_runtime_environments": top_items(runtime_counter, top_k),
        "top_os_compatibility": top_items(os_counter, top_k),
        "avg_confidence_by_category": avg_confidence_by_category,
        "parse_error_files": parse_errors,
    }


def analyze_token_metrics(results_dir: Path) -> dict[str, Any]:
    metric_files = sorted(results_dir.glob("*.token-metrics.yaml"))

    parse_errors = []
    processed = 0

    baseline_vals: list[int] = []
    step1_vals: list[int] = []
    step2_vals: list[int] = []
    two_step_vals: list[int] = []

    per_repo_rows: list[dict[str, Any]] = []

    for path in metric_files:
        data, err = load_yaml(path)
        if err:
            parse_errors.append({"file": path.name, "error": err})
            continue
        if not isinstance(data, dict):
            parse_errors.append({"file": path.name, "error": "YAML root is not a mapping"})
            continue

        tokens = data.get("tokens")
        if not isinstance(tokens, dict):
            parse_errors.append({"file": path.name, "error": "Missing tokens mapping"})
            continue

        baseline = int(tokens.get("baseline_full_classification", 0) or 0)
        step1 = int(tokens.get("step1_selection_prompt", 0) or 0)
        step2 = int(tokens.get("step2_reduced_classification", 0) or 0)
        two_step_total = int(tokens.get("two_step_total", step1 + step2) or 0)

        processed += 1
        baseline_vals.append(baseline)
        step1_vals.append(step1)
        step2_vals.append(step2)
        two_step_vals.append(two_step_total)

        repo_name = path.name.replace(".token-metrics.yaml", "")
        step2_savings = baseline - step2
        step2_savings_percent = round((step2_savings / baseline) * 100.0, 3) if baseline > 0 else 0.0
        two_step_savings = baseline - two_step_total
        two_step_savings_percent = round((two_step_savings / baseline) * 100.0, 3) if baseline > 0 else 0.0
        per_repo_rows.append(
            {
                "repo": repo_name,
                "baseline_full_classification": baseline,
                "step1_selection_prompt": step1,
                "step2_reduced_classification": step2,
                "two_step_total": two_step_total,
                "step2_savings_vs_baseline": step2_savings,
                "step2_savings_percent_vs_baseline": step2_savings_percent,
                "two_step_savings_vs_baseline": two_step_savings,
                "two_step_savings_percent_vs_baseline": two_step_savings_percent,
            }
        )

    per_repo_rows.sort(key=lambda row: row["step2_reduced_classification"], reverse=True)
    repos_by_two_step_savings = sorted(
        per_repo_rows,
        key=lambda row: row["two_step_savings_vs_baseline"],
        reverse=True,
    )
    top_repos_by_two_step_savings = repos_by_two_step_savings

    def summarize(values: list[int]) -> dict[str, Any]:
        if not values:
            return {"sum": 0, "avg": 0.0, "median": 0.0, "min": 0, "max": 0}
        return {
            "sum": int(sum(values)),
            "avg": round(sum(values) / len(values), 3),
            "median": float(median(values)),
            "min": int(min(values)),
            "max": int(max(values)),
        }

    summary = {
        "total_metric_files": len(metric_files),
        "processed_metric_files": processed,
        "yaml_or_schema_errors": len(parse_errors),
        "baseline_full_classification": summarize(baseline_vals),
        "step1_selection_prompt": summarize(step1_vals),
        "step2_reduced_classification": summarize(step2_vals),
        "two_step_total": summarize(two_step_vals),
        "aggregate_savings": {
            "step2_vs_baseline": int(sum(baseline_vals) - sum(step2_vals)),
            "two_step_total_vs_baseline": int(sum(baseline_vals) - sum(two_step_vals)),
        },
        "parse_error_files": parse_errors,
        "top_repos_by_step2_tokens": per_repo_rows,
        "top_repos_by_two_step_savings": top_repos_by_two_step_savings,
    }

    return summary


def analyze_selected_files(summaries_dir: Path, top_k: int) -> dict[str, Any]:
    selected_files = sorted(summaries_dir.glob("*.selected-files.yaml"))

    parse_errors = []
    processed = 0

    name_counter: Counter[str] = Counter()
    ext_counter: Counter[str] = Counter()

    for path in selected_files:
        data, err = load_yaml(path)
        if err:
            parse_errors.append({"file": path.name, "error": err})
            continue
        if not isinstance(data, dict):
            parse_errors.append({"file": path.name, "error": "YAML root is not a mapping"})
            continue

        paths = data.get("selected_files")
        if not isinstance(paths, list):
            parse_errors.append({"file": path.name, "error": "Missing selected_files list"})
            continue

        processed += 1

        for raw in paths:
            file_path = str(raw)
            name = Path(file_path).name.strip()
            if not name:
                continue
            ext = Path(name).suffix.lower() if Path(name).suffix else "<no_ext>"
            name_counter[name] += 1
            ext_counter[ext] += 1

    return {
        "total_selected_files_docs": len(selected_files),
        "processed_selected_files_docs": processed,
        "yaml_or_schema_errors": len(parse_errors),
        "top_file_names": top_items(name_counter, top_k),
        "top_file_extensions": top_items(ext_counter, top_k),
        "parse_error_files": parse_errors,
    }


def write_yaml(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.dump(payload, f, sort_keys=False, allow_unicode=True)


def write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)


def _fmt_top_rows(rows: list[dict[str, Any]], limit: int = 10) -> str:
    trimmed = rows[:limit]
    if not trimmed:
        return "(none)"
    return ", ".join(f"{row['name']} ({row['count']})" for row in trimmed)


def print_console_summary(
    results_analysis: dict[str, Any],
    tokens_analysis: dict[str, Any],
    selected_files_analysis: dict[str, Any],
) -> None:
    print("\n=== Results Analysis ===")
    print(
        f"Processed {results_analysis['processed_result_files']}/"
        f"{results_analysis['total_result_files']} result files"
    )
    print(
        "Top languages: "
        + _fmt_top_rows(results_analysis.get("top_programming_languages", []), limit=8)
    )
    print(
        "Top build tools: "
        + _fmt_top_rows(results_analysis.get("top_build_tools", []), limit=8)
    )

    print("\n=== Token Analysis ===")
    token_step2 = tokens_analysis.get("step2_reduced_classification", {})
    token_baseline = tokens_analysis.get("baseline_full_classification", {})
    savings = tokens_analysis.get("aggregate_savings", {})
    print(
        f"Processed {tokens_analysis['processed_metric_files']}/"
        f"{tokens_analysis['total_metric_files']} token metric files"
    )
    print(
        "Workflow: baseline = one-shot full classification prompt; "
        "sequential two-prompt approach = step1 LLM file pre-filter + "
        "step2 final classification over selected files."
    )
    print(
        f"One-shot baseline full prompt: {token_baseline.get('sum', 0):,} | "
        f"Full two-prompt pipeline: {tokens_analysis.get('two_step_total', {}).get('sum', 0):,}"
    )
    print(
        "Aggregate savings (primary) - "
        f"full two-prompt pipeline vs baseline: {savings.get('two_step_total_vs_baseline', 0):,}"
    )
    print(
        "Aggregate savings (secondary diagnostic) - "
        f"step2 final prompt only vs baseline: {savings.get('step2_vs_baseline', 0):,}"
    )
    print(
        f"Step-2 final classification prompt total: {token_step2.get('sum', 0):,}"
    )

    print("\n=== Selected Files Analysis ===")
    print(
        f"Processed {selected_files_analysis['processed_selected_files_docs']}/"
        f"{selected_files_analysis['total_selected_files_docs']} selected-files docs"
    )
    print(
        "Top file names: "
        + _fmt_top_rows(selected_files_analysis.get("top_file_names", []), limit=10)
    )
    print(
        "Top extensions: "
        + _fmt_top_rows(selected_files_analysis.get("top_file_extensions", []), limit=10)
    )


def build_markdown_report(
    results_analysis: dict[str, Any],
    tokens_analysis: dict[str, Any],
    selected_files_analysis: dict[str, Any],
) -> str:
    lines: list[str] = []
    lines.append("# Dataset Analysis Report")
    lines.append("")

    lines.append("## Results Analysis")
    lines.append(
        f"- Processed result files: {results_analysis['processed_result_files']}/"
        f"{results_analysis['total_result_files']}"
    )
    lines.append(f"- Schema/error files: {results_analysis['schema_or_error_result_files']}")
    lines.append(f"- YAML parse errors: {results_analysis['yaml_parse_errors']}")
    lines.append("")
    lines.append("### Top Programming Languages")
    for row in results_analysis.get("top_programming_languages", [])[:20]:
        lines.append(f"- {row['name']}: {row['count']}")
    lines.append("")
    lines.append("### Top Build Tools")
    for row in results_analysis.get("top_build_tools", [])[:20]:
        lines.append(f"- {row['name']}: {row['count']}")

    lines.append("")
    lines.append("## Token Analysis")
    lines.append(
        f"- Processed metric files: {tokens_analysis['processed_metric_files']}/"
        f"{tokens_analysis['total_metric_files']}"
    )
    lines.append(f"- YAML/schema errors: {tokens_analysis['yaml_or_schema_errors']}")
    baseline = tokens_analysis.get("baseline_full_classification", {})
    step1 = tokens_analysis.get("step1_selection_prompt", {})
    step2 = tokens_analysis.get("step2_reduced_classification", {})
    two_step = tokens_analysis.get("two_step_total", {})
    savings = tokens_analysis.get("aggregate_savings", {})
    lines.append("")
    lines.append("### Repositories With The Highest Two-Step Token Savings")
    lines.append(
        "This table answers: for which repositories does the full two-step pipeline "
        "(step1 pre-filter + step2 final classification) save the most tokens vs one-shot baseline?"
    )
    lines.append(
        "Savings are computed as baseline - full two-step pipeline total; "
        "negative values mean the two-step pipeline is larger than baseline."
    )
    lines.append("")
    lines.append("| Repo | One-Shot Baseline | Step 1 | Step 2 | Full Two-Prompt Pipeline | Savings vs Baseline | Savings % |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for row in tokens_analysis.get("top_repos_by_two_step_savings", [])[:20]:
        lines.append(
            f"| {row['repo']} | {row['baseline_full_classification']:,} | "
            f"{row['step1_selection_prompt']:,} | {row['step2_reduced_classification']:,} | "
            f"{row['two_step_total']:,} | {row['two_step_savings_vs_baseline']:,} | "
            f"{row['two_step_savings_percent_vs_baseline']:.3f}% |"
        )
    lines.append("")
    lines.append("### Prompt Comparison")
    lines.append(f"- One-shot baseline full prompt tokens: {baseline.get('sum', 0):,}")
    lines.append(f"- Step 1 LLM pre-filter prompt tokens: {step1.get('sum', 0):,}")
    lines.append(f"- Final classification prompt tokens after LLM pre-filtering: {step2.get('sum', 0):,}")
    lines.append(f"- Full sequential two-prompt pipeline tokens: {two_step.get('sum', 0):,}")
    lines.append(
        "- Aggregate savings: "
        f"final_prompt_vs_baseline={savings.get('step2_vs_baseline', 0):,}, "
        f"two_prompt_pipeline_vs_baseline={savings.get('two_step_total_vs_baseline', 0):,}"
    )
    lines.append("- Token formula: two-step total = step1 pre-filter prompt + step2 final classification prompt.")
    lines.append("")
    lines.append("### Pipeline Flow")
    lines.append(
        "- Workflow definition: baseline = one-shot full classification prompt; "
        "step1 = LLM-based pre-filter that selects likely relevant files from the structure summary; "
        "step2 = final classification prompt using only those selected files; "
        "two-step total = step1 + step2."
    )
    lines.append("1. Build a structure summary for the repository.")
    lines.append("2. Step 1 prompt asks the LLM to pre-filter and select likely relevant files.")
    lines.append("3. Gather contents only for those selected files.")
    lines.append("4. Step 2 prompt performs final installation/classification analysis on that reduced context.")
    lines.append("5. Compare this sequential pipeline against the one-shot baseline full prompt.")

    lines.append("")
    lines.append("## Selected Files Analysis")
    lines.append(
        f"- Processed selected-files docs: {selected_files_analysis['processed_selected_files_docs']}/"
        f"{selected_files_analysis['total_selected_files_docs']}"
    )
    lines.append(f"- YAML/schema errors: {selected_files_analysis['yaml_or_schema_errors']}")
    lines.append("")
    lines.append("### Top File Names")
    for row in selected_files_analysis.get("top_file_names", [])[:25]:
        lines.append(f"- {row['name']}: {row['count']}")
    lines.append("")
    lines.append("### Top File Extensions")
    for row in selected_files_analysis.get("top_file_extensions", [])[:25]:
        lines.append(f"- {row['name']}: {row['count']}")

    lines.append("")
    return "\n".join(lines)


def _format_int(value: Any) -> str:
    try:
        return f"{int(value):,}"
    except Exception:
        return "0"


def build_html_dashboard(
        results_analysis: dict[str, Any],
        tokens_analysis: dict[str, Any],
        selected_files_analysis: dict[str, Any],
) -> str:
        top_languages = results_analysis.get("top_programming_languages", [])[:15]
        top_build_tools = results_analysis.get("top_build_tools", [])[:15]
        top_repos_tokens = tokens_analysis.get("top_repos_by_two_step_savings", [])
        top_file_names = selected_files_analysis.get("top_file_names", [])[:20]
        top_file_exts = selected_files_analysis.get("top_file_extensions", [])[:20]
        avg_confidence = results_analysis.get("avg_confidence_by_category", {})

        env = Environment(autoescape=select_autoescape(default_for_string=True))
        env.filters["fmt_int"] = _format_int

        template = env.from_string(
                """<!doctype html>
<html lang=\"en\">
<head>
    <meta charset=\"utf-8\">
    <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">
    <title>Dataset Analysis Dashboard</title>
    <script src=\"https://cdn.jsdelivr.net/npm/chart.js\"></script>
    <link rel=\"stylesheet\" href=\"https://cdn.jsdelivr.net/npm/simple-datatables@9.0.3/dist/style.css\">
    <script src=\"https://cdn.jsdelivr.net/npm/simple-datatables@9.0.3\"></script>
    <style>
        :root {
            --bg: #f5f4ef;
            --ink: #172321;
            --muted: #4b5b57;
            --card: #ffffff;
            --accent: #005f73;
            --accent-2: #ca6702;
            --line: #d6d7d1;
        }
        * { box-sizing: border-box; }
        body {
            margin: 0;
            font-family: "IBM Plex Sans", "Segoe UI", sans-serif;
            color: var(--ink);
            background: radial-gradient(circle at top right, #e9f2f3, var(--bg) 40%);
        }
        .wrap { max-width: 1200px; margin: 0 auto; padding: 28px 18px 42px; }
        .hero { display: flex; justify-content: space-between; align-items: end; gap: 12px; flex-wrap: wrap; }
        h1 { margin: 0; font-size: 1.9rem; letter-spacing: 0.02em; }
        .sub { margin: 6px 0 0; color: var(--muted); }
        .cards { display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 12px; margin-top: 18px; }
        .card { background: var(--card); border: 1px solid var(--line); border-radius: 12px; padding: 14px; }
        .k { font-size: 0.86rem; color: var(--muted); text-transform: uppercase; letter-spacing: 0.08em; }
        .v { font-size: 1.5rem; margin-top: 6px; font-weight: 700; color: var(--accent); }
        .grid { display: grid; grid-template-columns: 1fr 1fr; gap: 14px; margin-top: 16px; }
        .panel { background: var(--card); border: 1px solid var(--line); border-radius: 12px; padding: 14px; overflow: auto; }
        .panel h2 { margin: 0 0 10px; font-size: 1.05rem; }
        .chart-wrap { height: 280px; }
        table { width: 100%; border-collapse: collapse; font-size: 0.93rem; }
        th, td { border-bottom: 1px solid var(--line); padding: 8px 6px; text-align: left; }
        th { font-size: 0.82rem; color: var(--muted); text-transform: uppercase; letter-spacing: 0.06em; }
        tr:nth-child(even) td { background: #fafaf7; }
        .datatable-wrapper { margin-top: 8px; }
        .datatable-top, .datatable-bottom { gap: 8px; }
        .datatable-input {
            border: 1px solid var(--line);
            border-radius: 8px;
            padding: 6px 10px;
            background: #fff;
        }
        .datatable-selector {
            border: 1px solid var(--line);
            border-radius: 8px;
            padding: 4px 6px;
            background: #fff;
        }
        .flow {
            display: grid;
            grid-template-columns: repeat(5, minmax(120px, 1fr));
            gap: 10px;
            align-items: center;
            margin-top: 14px;
        }
        .flow-step {
            background: linear-gradient(180deg, #ffffff, #f7f7f2);
            border: 1px solid var(--line);
            border-radius: 12px;
            padding: 12px;
            min-height: 92px;
            display: flex;
            flex-direction: column;
            justify-content: center;
        }
        .flow-step strong {
            font-size: 0.8rem;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            color: var(--accent);
            margin-bottom: 6px;
        }
        .flow-arrow {
            text-align: center;
            color: var(--muted);
            font-size: 1.4rem;
            font-weight: 700;
        }
        .compare-grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 14px;
            margin-top: 14px;
        }
        .compare-box {
            background: linear-gradient(180deg, #ffffff, #f7f7f2);
            border: 1px solid var(--line);
            border-radius: 12px;
            padding: 14px;
        }
        .compare-box h3 {
            margin: 0 0 8px;
            font-size: 0.95rem;
        }
        .formula {
            margin-top: 12px;
            padding: 10px 12px;
            border-radius: 10px;
            background: #f1f6f6;
            border: 1px solid #d6e4e5;
            font-family: "IBM Plex Mono", "Consolas", monospace;
            font-size: 0.92rem;
        }
        .accent { color: var(--accent-2); }
        @media (max-width: 900px) {
            .grid { grid-template-columns: 1fr; }
            .flow { grid-template-columns: 1fr; }
            .compare-grid { grid-template-columns: 1fr; }
            .flow-arrow { display: none; }
        }
    </style>
</head>
<body>
    <div class=\"wrap\">
        <div class=\"hero\">
            <div>
                <h1>Dataset Analysis Dashboard</h1>
                <p class=\"sub\">Generated from results, token metrics, and selected-files summaries.</p>
            </div>
            <div class=\"sub\">Files processed: {{ processed_results|fmt_int }}/{{ total_results|fmt_int }}</div>
        </div>

        <section class=\"grid\" style=\"margin-top: 16px;\">
            <article class=\"panel\">
                <h2>Top Programming Languages (Chart.js)</h2>
                <div class=\"chart-wrap\"><canvas id=\"langChart\"></canvas></div>
            </article>
            <article class=\"panel\">
                <h2>Top Build Tools (Chart.js)</h2>
                <div class=\"chart-wrap\"><canvas id=\"buildChart\"></canvas></div>
            </article>
        </section>

        <section class=\"panel\" style=\"margin-top: 14px;\">
            <h2>Average Confidence Score by Category</h2>
            <p class=\"sub\" style=\"margin: 0 0 10px;\">Mean confidence score (0–1) reported by the model across all classified repositories.</p>
            <div class=\"chart-wrap\"><canvas id=\"confidenceChart\"></canvas></div>
        </section>

        <section class=\"grid\">
            <article class=\"panel\">
                <h2>Top File Names</h2>
                <table id=\"fileNamesTable\">
                    <thead><tr><th>File Name</th><th>Count</th></tr></thead>
                    <tbody>
                        {% for row in top_file_names %}
                        <tr><td>{{ row.name }}</td><td>{{ row.count|fmt_int }}</td></tr>
                        {% endfor %}
                    </tbody>
                </table>
            </article>
            <article class=\"panel\">
                <h2>Top File Extensions</h2>
                <table id=\"fileExtTable\">
                    <thead><tr><th>Extension</th><th>Count</th></tr></thead>
                    <tbody>
                        {% for row in top_file_exts %}
                        <tr><td>{{ row.name }}</td><td>{{ row.count|fmt_int }}</td></tr>
                        {% endfor %}
                    </tbody>
                </table>
            </article>
        </section>

        <section class=\"panel\" style=\"margin-top: 14px;\">
            <h2>Prompt Comparison And Workflow Context</h2>
            <div class=\"compare-grid\">
                <div class=\"compare-box\">
                    <h3>One-Shot Baseline</h3>
                    <p class=\"sub\" style=\"margin: 0;\">
                        A single classification prompt receives the full repository fingerprint at once.
                    </p>
                </div>
                <div class=\"compare-box\">
                    <h3>Sequential Two-Prompt Pipeline</h3>
                    <p class=\"sub\" style=\"margin: 0;\">
                        Step 1 uses an LLM to pre-filter likely relevant files from the structure summary.
                        Step 2 runs the final classification only on those selected files.
                    </p>
                </div>
            </div>
            <div class=\"formula\">two-step total = step1 pre-filter prompt + step2 final classification prompt</div>
            <div class=\"flow\">
                <div class=\"flow-step\">
                    <strong>Input</strong>
                    <span>Repository structure summary and metadata.</span>
                </div>
                <div class=\"flow-arrow\">→</div>
                <div class=\"flow-step\">
                    <strong>Step 1</strong>
                    <span>LLM pre-filters and selects likely relevant files.</span>
                </div>
                <div class=\"flow-arrow\">→</div>
                <div class=\"flow-step\">
                    <strong>Step 2</strong>
                    <span>Final classification prompt runs on the reduced file set.</span>
                </div>
            </div>
        </section>

        <section class=\"cards\">
            <article class=\"card\"><div class=\"k\">One-Shot Baseline Full Prompt</div><div class=\"v\">{{ baseline_sum|fmt_int }}</div></article>
            <article class=\"card\"><div class=\"k\">Step 1 LLM Pre-Filter Prompt</div><div class=\"v\">{{ step1_sum|fmt_int }}</div></article>
            <article class=\"card\"><div class=\"k\">Step 2 Final Prompt After Pre-Filtering</div><div class=\"v\">{{ step2_sum|fmt_int }}</div></article>
            <article class=\"card\"><div class=\"k\">Full Two-Prompt Pipeline</div><div class=\"v\">{{ two_step_sum|fmt_int }}</div></article>
            <article class=\"card\"><div class=\"k\">Full Two-Prompt Savings vs Baseline</div><div class=\"v accent\">{{ two_step_savings|fmt_int }}</div></article>
        </section>

        <section class=\"panel\" style=\"margin-top: 14px;\">
            <h2>Repositories With The Highest Two-Step Token Savings</h2>
            <p class=\"sub\" style=\"margin: 0 0 10px;\">
                Savings are baseline - full two-step pipeline total (step1 + step2). Negative values mean no savings for that repository.
            </p>
            <table id=\"repoTokensTable\">
                <thead>
                    <tr><th>Repo</th><th>One-Shot Baseline</th><th>Step 1</th><th>Step 2</th><th>Full Two-Prompt Pipeline</th><th>Savings vs Baseline</th><th>Savings %</th></tr>
                </thead>
                <tbody>
                    {% for row in top_repos_tokens %}
                    <tr>
                        <td>{{ row.repo }}</td>
                        <td>{{ row.baseline_full_classification|fmt_int }}</td>
                        <td>{{ row.step1_selection_prompt|fmt_int }}</td>
                        <td>{{ row.step2_reduced_classification|fmt_int }}</td>
                        <td>{{ row.two_step_total|fmt_int }}</td>
                        <td>{{ row.two_step_savings_vs_baseline|fmt_int }}</td>
                        <td>{{ "%.3f"|format(row.two_step_savings_percent_vs_baseline) }}%</td>
                    </tr>
                    {% endfor %}
                </tbody>
            </table>
        </section>
    </div>

    <script>
        const langLabels = {{ lang_labels_json | safe }};
        const langValues = {{ lang_values_json | safe }};
        const buildLabels = {{ build_labels_json | safe }};
        const buildValues = {{ build_values_json | safe }};
        const confLabels = {{ conf_labels_json | safe }};
        const confValues = {{ conf_values_json | safe }};

        const commonOptions = {
            responsive: true,
            maintainAspectRatio: false,
            plugins: { legend: { display: false } },
            scales: {
                x: { ticks: { color: '#4b5b57' }, grid: { color: '#e7e8e2' } },
                y: { ticks: { color: '#4b5b57' }, grid: { color: '#e7e8e2' } }
            }
        };

        new Chart(document.getElementById('langChart'), {
            type: 'bar',
            data: {
                labels: langLabels,
                datasets: [{ data: langValues, backgroundColor: '#005f73' }]
            },
            options: commonOptions
        });

        new Chart(document.getElementById('buildChart'), {
            type: 'bar',
            data: {
                labels: buildLabels,
                datasets: [{ data: buildValues, backgroundColor: '#ca6702' }]
            },
            options: commonOptions
        });

        new Chart(document.getElementById('confidenceChart'), {
            type: 'bar',
            data: {
                labels: confLabels,
                datasets: [{ data: confValues, backgroundColor: '#005f73' }]
            },
            options: {
                ...commonOptions,
                scales: {
                    ...commonOptions.scales,
                    y: { ...commonOptions.scales.y, min: 0, max: 1 }
                }
            }
        });

        const tableOptions = {
            perPage: 10,
            perPageSelect: [5, 10, 20, 50],
            searchable: true,
            sortable: true,
            labels: {
                placeholder: 'Search...',
                perPage: ' rows per table',
                noRows: 'No rows to display',
                info: 'Showing {start} to {end} of {rows} rows'
            }
        };

        new simpleDatatables.DataTable('#repoTokensTable', tableOptions);
        new simpleDatatables.DataTable('#fileNamesTable', tableOptions);
        new simpleDatatables.DataTable('#fileExtTable', tableOptions);
    </script>
</body>
</html>
"""
        )

        return template.render(
                processed_results=results_analysis.get("processed_result_files", 0),
                total_results=results_analysis.get("total_result_files", 0),
                baseline_sum=tokens_analysis.get("baseline_full_classification", {}).get("sum", 0),
                step1_sum=tokens_analysis.get("step1_selection_prompt", {}).get("sum", 0),
                step2_sum=tokens_analysis.get("step2_reduced_classification", {}).get("sum", 0),
                two_step_sum=tokens_analysis.get("two_step_total", {}).get("sum", 0),
                two_step_savings=tokens_analysis.get("aggregate_savings", {}).get("two_step_total_vs_baseline", 0),
                top_repos_tokens=top_repos_tokens,
                top_file_names=top_file_names,
                top_file_exts=top_file_exts,
                lang_labels_json=json.dumps([row.get("name", "") for row in top_languages]),
                lang_values_json=json.dumps([int(row.get("count", 0) or 0) for row in top_languages]),
                build_labels_json=json.dumps([row.get("name", "") for row in top_build_tools]),
                build_values_json=json.dumps([int(row.get("count", 0) or 0) for row in top_build_tools]),
                conf_labels_json=json.dumps(list(avg_confidence.keys())),
                conf_values_json=json.dumps(list(avg_confidence.values())),
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate analyses from result artifacts")
    parser.add_argument("--results-dir", default="classification_results", help="Directory containing result YAML files")
    parser.add_argument("--summaries-dir", default="summaries", help="Directory containing summary YAML files")
    parser.add_argument("--analysis-dir", default="analysis", help="Output directory for analysis files")
    parser.add_argument("--top-k", type=int, default=25, help="Top-N rows to keep for counter-based outputs")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent.parent
    results_dir = (project_root / args.results_dir).resolve()
    summaries_dir = (project_root / args.summaries_dir).resolve()
    analysis_dir = (project_root / args.analysis_dir).resolve()

    results_analysis = analyze_classification_results(results_dir, args.top_k)
    tokens_analysis = analyze_token_metrics(results_dir)
    selected_files_analysis = analyze_selected_files(summaries_dir, args.top_k)

    results_analysis_path = analysis_dir / "results-analysis.yaml"
    tokens_analysis_path = analysis_dir / "token-analysis.yaml"
    selected_files_analysis_path = analysis_dir / "selected-files-analysis.yaml"
    report_path = analysis_dir / "REPORT.md"
    dashboard_path = analysis_dir / "dashboard.html"

    write_yaml(results_analysis_path, results_analysis)
    write_yaml(tokens_analysis_path, tokens_analysis)
    write_yaml(selected_files_analysis_path, selected_files_analysis)

    print(f"Wrote {results_analysis_path}")
    print(f"Wrote {tokens_analysis_path}")
    print(f"Wrote {selected_files_analysis_path}")

    markdown_report = build_markdown_report(results_analysis, tokens_analysis, selected_files_analysis)
    html_dashboard = build_html_dashboard(results_analysis, tokens_analysis, selected_files_analysis)
    write_text(dashboard_path, html_dashboard)
    print(f"Wrote {dashboard_path}")
    write_text(report_path, markdown_report)
    print(f"Wrote {report_path}")
    print_console_summary(results_analysis, tokens_analysis, selected_files_analysis)


if __name__ == "__main__":
    main()
