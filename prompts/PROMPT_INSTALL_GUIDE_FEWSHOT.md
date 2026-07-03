<example index="1">
<user>
Classification: language C; build_tool [Makefile]; system_dependencies gcc, make; library (no entrypoint).
Dockerfile build steps: `make all` then `make test`.
Verification command: `make test`.
Write the INSTALL.md.
</user>
<assistant>
# Install

Build and verify this C library locally on a Debian-based Linux host.

## Prerequisites

Install the build toolchain:

```bash
sudo apt-get update && sudo apt-get install -y gcc make
```

## Build Steps

1. Build the library and test binary:
   ```bash
   make all
   ```

## Install Artifacts

`make all` produces the static and shared libraries (`libcjson.a`, `libcjson.so`) and the public headers. To install them system-wide (optional):

```bash
sudo make install
```

## Verification

Run the test target; a clean exit means the build is good:

```bash
make test
```
</assistant>
</example>

<example index="2">
<user>
Classification: language Rust; build_tool [cargo]; application binary `app`.
Dockerfile build steps: `cargo build --release`.
Verification command: `cargo test --release`.
Write the INSTALL.md.
</user>
<assistant>
# Install

Build and verify this Rust application locally on a Debian-based Linux host.

## Prerequisites

Install the Rust toolchain (Ubuntu's apt rustc is 1.75 — too old for modern crates, so use rustup):

```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal
export PATH="$HOME/.cargo/bin:$PATH"
```

## Build Steps

1. Compile the release binary:
   ```bash
   cargo build --release
   ```

## Install Artifacts

The optimized binary is written to `target/release/app`. Copy it onto your `PATH` if desired:

```bash
sudo install -m 0755 target/release/app /usr/local/bin/app
```

## Verification

Run the test suite:

```bash
cargo test --release
```
</assistant>
</example>
