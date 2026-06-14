Repository: {{REPO_URL}}
Improve build strategy hypotheses and risk notes using evidence.
Use think at most once, only when changing strategy.
Return keys: thought, hypothesis_updates (list), risk_updates (list), selected_files (list), done (bool).
You have a limited tool-call budget. When you have enough evidence — or when you are near the budget — STOP exploring and call finalize(answer=<your YAML answer>) as your last action. Always finalize, even with partial evidence; never keep exploring until you are cut off.

CURRENT_HYPOTHESES:
{{CURRENT_HYPOTHESES}}

CURRENT_RISKS:
{{CURRENT_RISKS}}

SUBAGENT_SIGNALS:
{{SUBAGENT_SIGNALS}}

SUMMARY_EVIDENCE:
{{SUMMARY_EVIDENCE}}
