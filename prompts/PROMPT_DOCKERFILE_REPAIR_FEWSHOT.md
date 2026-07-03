<example index="1">
<user>
Build failed. Log (tail):
  fatal error: openssl/ssl.h: No such file or directory
  compilation terminated. make: *** [Makefile:42: build] Error 1
Current AGENT_BUILD_STEPS region:
  RUN sudo apt-get update && sudo apt-get install -y gcc make && sudo rm -rf /var/lib/apt/lists/*
  RUN make all
Root cause: missing OpenSSL development headers. Fix the smallest thing; keep build steps.
Return the COMPLETE corrected Dockerfile (base template preserved verbatim, only the
AGENT_BUILD_STEPS region changed).
</user>
<assistant>
# Base Dockerfile projects
# Includes: Ubuntu, CA bootstrap, git, curl, pkg-config

# Pinned by digest so the base layer is frozen: the ubuntu:24.04 tag is
# re-published on patch updates, which would silently change the base
# between eval runs and break reproducibility. Re-pin with:
#   docker pull ubuntu:24.04 && docker inspect ubuntu:24.04 --format '{{index .RepoDigests 0}}'
# (Ubuntu 24.04 LTS apt repos are stable; the digest also freezes the base image.)
FROM ubuntu@sha256:4fbb8e6a8395de5a7550b33509421a2bafbc0aab6c06ba2cef9ebffbc7092d90

# Non-interactive apt so `sudo apt-get install` never blocks on a prompt.
ENV DEBIAN_FRONTEND=noninteractive

# Make builds (as) reproducible (as possible)
ENV SOURCE_DATE_EPOCH=0
ENV CARGO_INCREMENTAL=0
ENV RUSTFLAGS="-C codegen-units=1"

# Skip Playwright browser-binary downloads during `npm/pnpm install`. A repo with
# Playwright in its dev dependency tree (e.g. vite) runs a postinstall that downloads
# 170 MiB+ of browsers and then a launch-validation that hangs indefinitely in a
# minimal container missing libnss3/libnspr4 — this stalled vite's build until the
# build timeout killed it. Browsers are irrelevant to building a package, so skip them.
# Harmless no-op for repos that do not use Playwright.
ENV PLAYWRIGHT_SKIP_BROWSER_DOWNLOAD=1

# Install CA certificates and corporate CA if provided
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Decode and install corporate CA if present
RUN if [ -n "$MANUALREPOS_CA_CERT_B64" ]; then \
      echo "$MANUALREPOS_CA_CERT_B64" | base64 -d > /usr/local/share/ca-certificates/corporate-ca.crt && \
      update-ca-certificates; \
    fi

# Install default packages (incl. sudo, so the build user can install system packages;
# xxhash provides xxh64sum, used post-build to hash artifacts for HARD verification).
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    curl \
    pkg-config \
    sudo \
    xxhash \
    && rm -rf /var/lib/apt/lists/*

# Non-root build user with passwordless sudo, so the build feels like a normal dev
# environment: build commands run as this user; use `sudo` for system package installs
# (`sudo apt-get install ...`) or privileged steps (`sudo make install`). The repo
# directory is owned by the user so the build can write artifacts into it.
RUN useradd -m -s /bin/bash manualrepos \
    && mkdir -p /home/manualrepos/repo \
    && chown manualrepos:manualrepos /home/manualrepos/repo \
    && echo 'manualrepos ALL=(ALL) NOPASSWD:ALL' > /etc/sudoers.d/manualrepos \
    && chmod 0440 /etc/sudoers.d/manualrepos
USER manualrepos
WORKDIR /home/manualrepos/repo

# Copy repository from build context (already checked out on host)
COPY --chown=manualrepos:manualrepos . .

# Dependency installs + build commands go here. EACH step is a Docker RUN instruction
# (every line starts with RUN), running as the non-root user which has passwordless sudo.
# Edit only between the markers. Example:
#   RUN curl --proto '=https' -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal
#   RUN cargo build --release
# AGENT_BUILD_STEPS_BEGIN
# Add libssl-dev + pkg-config (the missing openssl/ssl.h) to the single up-front install.
RUN sudo apt-get update && sudo apt-get install -y \
    gcc \
    make \
    libssl-dev \
    pkg-config \
    && sudo rm -rf /var/lib/apt/lists/*

RUN make all
# AGENT_BUILD_STEPS_END

# Final CMD or ENTRYPOINT (runs as manualrepos)
# Library — no CMD/ENTRYPOINT.
</assistant>
</example>

<example index="2">
<user>
Build failed. Log (tail):
  error: externally-managed-environment
  ... To install Python packages system-wide, try apt install ... (PEP 668)
Current AGENT_BUILD_STEPS region:
  RUN sudo apt-get update && sudo apt-get install -y python3 python3-pip && sudo rm -rf /var/lib/apt/lists/*
  RUN pip install .
Root cause: PEP 668 blocks system pip. Use a virtualenv rather than forcing system installs.
Return the COMPLETE corrected Dockerfile (base template preserved verbatim, only the
AGENT_BUILD_STEPS region changed).
</user>
<assistant>
# Base Dockerfile projects
# Includes: Ubuntu, CA bootstrap, git, curl, pkg-config

# Pinned by digest so the base layer is frozen: the ubuntu:24.04 tag is
# re-published on patch updates, which would silently change the base
# between eval runs and break reproducibility. Re-pin with:
#   docker pull ubuntu:24.04 && docker inspect ubuntu:24.04 --format '{{index .RepoDigests 0}}'
# (Ubuntu 24.04 LTS apt repos are stable; the digest also freezes the base image.)
FROM ubuntu@sha256:4fbb8e6a8395de5a7550b33509421a2bafbc0aab6c06ba2cef9ebffbc7092d90

# Non-interactive apt so `sudo apt-get install` never blocks on a prompt.
ENV DEBIAN_FRONTEND=noninteractive

# Make builds (as) reproducible (as possible)
ENV SOURCE_DATE_EPOCH=0
ENV CARGO_INCREMENTAL=0
ENV RUSTFLAGS="-C codegen-units=1"

# Skip Playwright browser-binary downloads during `npm/pnpm install`. A repo with
# Playwright in its dev dependency tree (e.g. vite) runs a postinstall that downloads
# 170 MiB+ of browsers and then a launch-validation that hangs indefinitely in a
# minimal container missing libnss3/libnspr4 — this stalled vite's build until the
# build timeout killed it. Browsers are irrelevant to building a package, so skip them.
# Harmless no-op for repos that do not use Playwright.
ENV PLAYWRIGHT_SKIP_BROWSER_DOWNLOAD=1

# Install CA certificates and corporate CA if provided
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Decode and install corporate CA if present
RUN if [ -n "$MANUALREPOS_CA_CERT_B64" ]; then \
      echo "$MANUALREPOS_CA_CERT_B64" | base64 -d > /usr/local/share/ca-certificates/corporate-ca.crt && \
      update-ca-certificates; \
    fi

# Install default packages (incl. sudo, so the build user can install system packages;
# xxhash provides xxh64sum, used post-build to hash artifacts for HARD verification).
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    curl \
    pkg-config \
    sudo \
    xxhash \
    && rm -rf /var/lib/apt/lists/*

# Non-root build user with passwordless sudo, so the build feels like a normal dev
# environment: build commands run as this user; use `sudo` for system package installs
# (`sudo apt-get install ...`) or privileged steps (`sudo make install`). The repo
# directory is owned by the user so the build can write artifacts into it.
RUN useradd -m -s /bin/bash manualrepos \
    && mkdir -p /home/manualrepos/repo \
    && chown manualrepos:manualrepos /home/manualrepos/repo \
    && echo 'manualrepos ALL=(ALL) NOPASSWD:ALL' > /etc/sudoers.d/manualrepos \
    && chmod 0440 /etc/sudoers.d/manualrepos
USER manualrepos
WORKDIR /home/manualrepos/repo

# Copy repository from build context (already checked out on host)
COPY --chown=manualrepos:manualrepos . .

# Dependency installs + build commands go here. EACH step is a Docker RUN instruction
# (every line starts with RUN), running as the non-root user which has passwordless sudo.
# Edit only between the markers. Example:
#   RUN curl --proto '=https' -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal
#   RUN cargo build --release
# AGENT_BUILD_STEPS_BEGIN
# Add venv support, then build/install inside an isolated environment (PEP 668 safe).
RUN sudo apt-get update && sudo apt-get install -y \
    python3 \
    python3-pip \
    python3-venv \
    && sudo rm -rf /var/lib/apt/lists/*

# Create a venv the build user owns; persist it on PATH for later RUN steps.
RUN python3 -m venv /home/manualrepos/venv
ENV PATH="/home/manualrepos/venv/bin:$PATH"

RUN pip install .
# AGENT_BUILD_STEPS_END

# Final CMD or ENTRYPOINT (runs as manualrepos)
# Application entry point.
CMD ["python3", "-m", "app"]
</assistant>
</example>
