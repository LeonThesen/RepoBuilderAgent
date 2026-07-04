Repository: {{REPO_URL}}
Improve build strategy hypotheses and risk notes using evidence.
Use think at most once, only when changing strategy.
Return keys: thought, hypothesis_updates (list), risk_updates (list), selected_files (list), done (bool).
You have at most {{MAX_TOOL_CALLS}} tool calls. Use them sparingly and call finalize(answer=<your YAML answer>) before they run out — finalize early with partial evidence rather than being cut off.

CURRENT_HYPOTHESES:
{{CURRENT_HYPOTHESES}}

CURRENT_RISKS:
{{CURRENT_RISKS}}

SUBAGENT_SIGNALS:
{{SUBAGENT_SIGNALS}}

SUMMARY_EVIDENCE:
{{SUMMARY_EVIDENCE}}
