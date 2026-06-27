Repository: {{REPO_URL}}
Use tools to find EVERY installation/build-relevant file, then return YAML keys: thought, selected_files, done.
Workflow: list_tree to orient, search_pattern to find specific file types, read_file to inspect contents.
Use think at most once, only when changing strategy. selected_files must be repo-relative paths only.

Optimize for COVERAGE: missing a build-relevant file is far worse than selecting an extra one. Before you finalize, look for ALL of these wherever they exist (do not stop at the first manifest you find):
- the primary build manifest(s) at the repo ROOT (the file your detected build tool reads)
- their lockfile(s) and any workspace / monorepo config sitting beside them
- toolchain, language-version, or runtime-version pin files
- top-level build scripts, Dockerfiles, and build-system entry/config files
When the same filename appears both at the root and deep inside test/example/subproject trees, prefer the ROOT one — the deep copies are usually noise.

Keep at most {{MAX_FILES}} files. Set done=true only once you have captured every root-level build-relevant file you can find — not at the first sufficient one.
You have at most {{MAX_TOOL_CALLS}} tool calls; call finalize(answer=<your YAML answer>) before they run out. If forced to cut short, finalize with everything you have found so far rather than being cut off.

STRUCTURE_SUMMARY (pre-built index — use list_tree/search_pattern/read_file for live inspection):
{{STRUCTURE_SUMMARY}}
