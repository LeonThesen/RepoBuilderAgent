Repository: {{REPO_URL}}
Review and refine generated synthesis output (round {{ROUND_INDEX}}/{{ROUND_TOTAL}}).
Return keys: thought, accepted (bool), revised_hypotheses (list), revised_risks (list), revised_toolchains (list), critique_notes (list), done (bool).

revised_toolchains: refine the generator's required-toolchain pins to ONLY the language runtime/compiler versions the OS must supply that Ubuntu 24.04 lacks (JDK 21, gcc-13, apt Maven ~3.8, apt Rust ~1.75, Node 18). DROP build-tool-resolved noise (Gradle wrapper version, Kotlin/KSP/plugin versions, any library/dependency version) and versions the base already satisfies. ADD the real pin if the generator missed it — for a JVM/Gradle repo that is the JDK toolchain version (`jvmToolchain(N)`/`languageVersion`/`.java-version`), which is usually THE one that matters. Same `<tool> <version> — <where>` format; empty list if none.

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
