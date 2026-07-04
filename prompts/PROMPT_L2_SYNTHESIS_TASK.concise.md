Repository: {{REPO_URL}}
Improve build strategy hypotheses and risk notes using evidence.
Use think at most once, only when changing strategy.
Return keys: thought, hypothesis_updates (list), risk_updates (list), toolchain_updates (list), selected_files (list), done (bool).

toolchain_updates: ONLY the LANGUAGE RUNTIME/COMPILER version the OS must supply that the base lacks (base = Ubuntu 24.04: JDK 21, gcc-13, apt Maven ~3.8, apt Rust ~1.75, Node 18). `<tool> <version> — <where>`.
INCLUDE: JDK toolchain version (Gradle jvmToolchain(N)/languageVersion/sourceCompatibility, .java-version, Maven compiler.release), Rust (rust-toolchain.toml/.tool-versions), Node (.nvmrc/engines.node), Python, system Maven a plugin REQUIRES above apt's. For JVM/Gradle repos the JDK toolchain version is almost always THE pin — find it first.
EXCLUDE (build tool resolves these, never list): Gradle wrapper version, Kotlin/KSP/plugin versions, any library/dep version in libs.versions.toml/package.json/Cargo.toml.
Good: `jdk 17 — build.gradle.kts jvmToolchain(17)`, `maven 3.9.6 — plugin requires`. Empty if base satisfies.

At most {{MAX_TOOL_CALLS}} tool calls. Use sparingly; call finalize(answer=<your YAML answer>) before they run out — finalize early with partial evidence rather than cut off.

CURRENT_HYPOTHESES:
{{CURRENT_HYPOTHESES}}

CURRENT_RISKS:
{{CURRENT_RISKS}}

SUBAGENT_SIGNALS:
{{SUBAGENT_SIGNALS}}

SUMMARY_EVIDENCE:
{{SUMMARY_EVIDENCE}}
