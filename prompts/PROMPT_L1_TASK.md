Repository: {{REPO_URL}}
Use tools to identify and inspect installation-relevant files, then return YAML keys: thought, selected_files, done.
Workflow: list_tree to orient, search_pattern to find specific file types, read_file to inspect contents.
Use think at most once, only when changing strategy. selected_files must be repo-relative paths only.
Keep at most {{MAX_FILES}} files and set done=true when you have enough evidence for install/build/verify.
You have a limited tool-call budget. When you have enough evidence — or when you are near the budget — STOP exploring and call finalize(answer=<your YAML answer>) as your last action. Always finalize, even with partial evidence; never keep exploring until you are cut off.

STRUCTURE_SUMMARY (pre-built index — use list_tree/search_pattern/read_file for live inspection):
{{STRUCTURE_SUMMARY}}
