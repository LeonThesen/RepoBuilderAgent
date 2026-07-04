Repository: {{REPO_URL}}
Improve build strategy hypotheses and risk notes using evidence.
Use think at most once, only when changing strategy.
Return keys: thought, hypothesis_updates (list), risk_updates (list), toolchain_updates (list), selected_files (list), done (bool).

toolchain_updates: report ONLY the LANGUAGE RUNTIME / COMPILER (and build-tool binary) versions the OS must supply that the base does not ship by default (base = Ubuntu 24.04 LTS: JDK 21, gcc-13, apt Maven ~3.8, apt Rust ~1.75, Node 18). These are the versions that break the build if the OS-provided one is wrong. One string per pin, format `<tool> <version> — <where found>`.
- INCLUDE: Java/JDK toolchain version (Gradle `jvmToolchain(N)` / `languageVersion` / `sourceCompatibility`, `.java-version`, Maven `<release>`/`maven.compiler.release`); Rust version (`rust-toolchain.toml`, `.tool-versions`); Node version (`.nvmrc`, package.json `engines.node`); Python version; a system Maven version a plugin explicitly REQUIRES above apt's. For a JVM/Gradle repo the JDK toolchain version is almost always THE pin that matters — look for it first.
- EXCLUDE (these are resolved by the build tool itself, NOT provisioned by the OS — never list them): the Gradle *wrapper* version (gradle-wrapper.properties), Kotlin/KSP/plugin versions, any library/dependency version in `libs.versions.toml`/`package.json` dependencies/`Cargo.toml` deps.
Example good: `jdk 17 — build.gradle.kts jvmToolchain(17)`, `maven 3.9.6 — protobuf-maven-plugin requires`, `rust 1.82 — rust-toolchain.toml`. Empty list if the base's defaults satisfy the repo.
You have at most {{MAX_TOOL_CALLS}} tool calls. Use them sparingly and call finalize(answer=<your YAML answer>) before they run out — finalize early with partial evidence rather than being cut off.

CURRENT_HYPOTHESES:
{{CURRENT_HYPOTHESES}}

CURRENT_RISKS:
{{CURRENT_RISKS}}

SUBAGENT_SIGNALS:
{{SUBAGENT_SIGNALS}}

SUMMARY_EVIDENCE:
{{SUMMARY_EVIDENCE}}
