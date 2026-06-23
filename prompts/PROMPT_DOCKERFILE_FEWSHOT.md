<example index="1">
<user>
Classification: language C; build_tool [Makefile]; system_dependencies gcc (required), make (required);
build_steps ["make all"]; verification ["make test"]; library (no entrypoint).
Fill the AGENT_BUILD_STEPS region of the base template (everything else stays verbatim).
</user>
<assistant>
# AGENT_BUILD_STEPS_BEGIN
# Install the complete toolchain up front, in one layer.
RUN sudo apt-get update && sudo apt-get install -y \
    gcc \
    make \
    && sudo rm -rf /var/lib/apt/lists/*

# Build the library. Tests/verification (`make test`) run via the separate verify
# command downstream — not here.
RUN make all
# AGENT_BUILD_STEPS_END
# Library — no CMD/ENTRYPOINT.
</assistant>
</example>

<example index="2">
<user>
Classification: language Rust; build_tool [cargo]; system_dependencies (none beyond toolchain);
build_steps ["cargo build --release"]; verification ["cargo test --release"]; application binary `app`.
Fill the AGENT_BUILD_STEPS region of the base template (everything else stays verbatim).
</user>
<assistant>
# AGENT_BUILD_STEPS_BEGIN
# Install the Rust toolchain from Debian apt (no network toolchain installer).
RUN sudo apt-get update && sudo apt-get install -y \
    cargo \
    rustc \
    && sudo rm -rf /var/lib/apt/lists/*

# Release build, parallelised across cores.
RUN cargo build --release -j $(nproc)
# AGENT_BUILD_STEPS_END
CMD ["./target/release/app"]
</assistant>
</example>
