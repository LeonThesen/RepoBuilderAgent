Repository: {{REPO_URL}}
Validate installation/build evidence coverage and highlight risky gaps.
Use think at most once, only when changing strategy.
Return keys: thought, checks (map), warnings (list), selected_files (list), done (bool).
Each checks entry must include status (pass|warn|fail) and detail.
You have a limited tool-call budget. When you have enough evidence — or when you are near the budget — STOP exploring and call finalize(answer=<your YAML answer>) as your last action. Always finalize, even with partial evidence; never keep exploring until you are cut off.

CURRENT_CHECKS:
{{CURRENT_CHECKS}}
SYNTHESIS_ARTIFACT:
{{SYNTHESIS_ARTIFACT}}
SUMMARY_EVIDENCE:
{{SUMMARY_EVIDENCE}}
