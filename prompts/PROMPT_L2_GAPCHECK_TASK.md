Repository: {{REPO_NAME}}
The four parallel signal scanners have completed. Analyze their outputs for conflicts and gaps in a single pass — no tools, no follow-up turns.

Return keys: thought, has_manifest (bool), has_docker (bool), has_scripts (bool), has_source_deps (bool), conflicts (list of objects with keys source_a, source_b, field, detail, resolution), gaps (list of strings describing missing evidence), notes (list).

Priority for conflict resolution: Dockerfile > Manifest > Build scripts > Source code

BUILD_SIGNALS (manifest/build-declaration files found):
{{BUILD_SIGNALS}}

RUNTIME_SIGNALS (containerization/runtime files found):
{{RUNTIME_SIGNALS}}

SCRIPTS_SIGNALS (Makefile/install scripts found):
{{SCRIPTS_SIGNALS}}

SOURCE_SIGNALS (source files with implicit deps/ENV vars found):
{{SOURCE_SIGNALS}}

SUMMARY_EVIDENCE:
{{SUMMARY_EVIDENCE}}
