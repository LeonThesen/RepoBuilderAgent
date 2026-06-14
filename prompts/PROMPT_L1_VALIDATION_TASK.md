Repository: {{REPO_URL}}
Validate installation/build evidence coverage and highlight risky gaps.
Use think at most once, only when changing strategy.
Return keys: thought, checks (map), warnings (list), selected_files (list), done (bool).
Each checks entry must include status (pass|warn|fail) and detail.
You have at most {{MAX_TOOL_CALLS}} tool calls. Use them sparingly and call finalize(answer=<your YAML answer>) before they run out — finalize early with partial evidence rather than being cut off.

CURRENT_CHECKS:
{{CURRENT_CHECKS}}
SYNTHESIS_ARTIFACT:
{{SYNTHESIS_ARTIFACT}}
SUMMARY_EVIDENCE:
{{SUMMARY_EVIDENCE}}
