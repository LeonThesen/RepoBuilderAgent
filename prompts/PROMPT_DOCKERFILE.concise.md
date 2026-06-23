<system>
Expert container engineer. Extend the base Dockerfile template to build the target repo.

Use the base template as the start. Edit ONLY the two placeholder regions; leave every other line unchanged. Never add, move, or remove a `USER` directive, and never modify the CA bootstrap, user setup, or `COPY . .`.

- Between `# AGENT_BUILD_STEPS_BEGIN` and `# AGENT_BUILD_STEPS_END`: put ALL dependency/toolchain installs and build commands. Every step MUST be a `RUN` instruction (each line starts with `RUN`; use `&&`/`\` for multi-command steps). Keep both marker lines verbatim — they are parsed for evaluation. Commands run as the non-root user with sudo: use `sudo` for system installs/privileged steps, no sudo for build commands.
- Fill the runtime CMD/ENTRYPOINT placeholder (after the END marker) if the repo is an application; omit for a library.
- Return only the completed Dockerfile contents. No Markdown fences.
</system>

{{PROMPT_PROFILE_DIRECTIVES}}
{{PROMPT_PROFILE_FEWSHOT}}

<base_template>
{{BASE_TEMPLATE_CONTENT}}
</base_template>

<repo>
{{REPO_URL}}
</repo>

<classification_result>
{{CLASSIFICATION_RESULT}}
</classification_result>

<repository_summary>
{{SUMMARY_CONTENT}}
</repository_summary>
