Repository: {{REPO_URL}}
Review and refine generated synthesis output (round {{ROUND_INDEX}}/{{ROUND_TOTAL}}).
Return keys: thought, accepted (bool), revised_hypotheses (list), revised_risks (list), revised_toolchains (list), critique_notes (list), done (bool).

revised_toolchains: keep ONLY language runtime/compiler versions the OS must supply that the base (Ubuntu 24.04: JDK 21, gcc-13, apt Maven ~3.8, Rust ~1.75, Node 18) lacks. DROP build-tool-resolved noise (Gradle wrapper version, Kotlin/KSP/plugin/library versions) and base-satisfied versions. ADD the missed pin — for JVM/Gradle that is the JDK toolchain version (jvmToolchain(N)/languageVersion/.java-version), usually THE one. `<tool> <version> — <where>`; empty if none.

GENERATOR_HYPOTHESES:
{{GENERATOR_HYPOTHESES}}

GENERATOR_RISKS:
{{GENERATOR_RISKS}}

GENERATOR_TOOLCHAINS:
{{GENERATOR_TOOLCHAINS}}

SUBAGENT_SIGNALS:
{{SUBAGENT_SIGNALS}}

SUMMARY_EVIDENCE:
{{SUMMARY_EVIDENCE}}
