Repository: {{REPO_URL}}
Tools find + inspect install-relevant files, then return YAML keys: thought, selected_files, done.
Flow: list_tree orient, search_pattern find file types, read_file inspect.
think at most once, only on strategy change. selected_files = repo-relative paths only.
At most {{MAX_FILES}} files; set done=true when enough evidence for install/build/verify.
At most {{MAX_TOOL_CALLS}} tool calls. Use sparingly; call finalize(answer=<your YAML answer>) before they run out — finalize early w/ partial evidence over getting cut off.

STRUCTURE_SUMMARY (pre-built index — use list_tree/search_pattern/read_file for live inspection):
{{STRUCTURE_SUMMARY}}