"""Validated Dockerfile RUN-block snippets for common build toolchains.

Each function returns one or more Dockerfile lines as a string ready to paste.
Snippets target Debian/Ubuntu-based images (apt). Where network is required
(Go, Rust, .NET, Gradle) the download URL follows official release conventions.
"""

from __future__ import annotations


def _apt(packages: str, extras: str = "") -> str:
    """Return a minimal apt-get RUN block with cache cleanup."""
    body = f"apt-get update && apt-get install -y --no-install-recommends {packages} && rm -rf /var/lib/apt/lists/*"
    if extras:
        body = f"apt-get update && apt-get install -y --no-install-recommends {packages} && {extras} && rm -rf /var/lib/apt/lists/*"
    return f"RUN {body}"


def install_jdk(version: str = "17") -> str:
    """Install OpenJDK Development Kit from apt.

    Common versions: 11, 17, 21. Use install_jre for a smaller runtime-only image.
    """
    v = (version or "17").strip()
    return _apt(f"openjdk-{v}-jdk")


def install_jre(version: str = "17") -> str:
    """Install OpenJDK Runtime Environment from apt (smaller than full JDK)."""
    v = (version or "17").strip()
    return _apt(f"openjdk-{v}-jre-headless")


def install_node(version: str = "20") -> str:
    """Install Node.js via NodeSource setup script.

    Common versions: 18, 20, 22. Installs the full nodejs package including npm.
    """
    v = (version or "20").strip()
    return (
        "RUN apt-get update && apt-get install -y --no-install-recommends curl ca-certificates && \\\n"
        f"    curl -fsSL https://deb.nodesource.com/setup_{v}.x | bash - && \\\n"
        "    apt-get install -y --no-install-recommends nodejs && \\\n"
        "    rm -rf /var/lib/apt/lists/*"
    )


def install_cargo(version: str = "") -> str:
    """Install Rust toolchain and Cargo via rustup (official method).

    Uses the minimal profile for smaller image size. Sets CARGO_HOME and RUSTUP_HOME
    to /usr/local so all users share the installation.
    """
    return (
        "ENV RUSTUP_HOME=/usr/local/rustup \\\n"
        "    CARGO_HOME=/usr/local/cargo \\\n"
        "    PATH=/usr/local/cargo/bin:$PATH\n"
        "RUN apt-get update && apt-get install -y --no-install-recommends curl ca-certificates && \\\n"
        "    curl https://sh.rustup.rs -sSf | sh -s -- -y --no-modify-path --profile minimal && \\\n"
        "    chmod -R a+w $RUSTUP_HOME $CARGO_HOME && \\\n"
        "    rm -rf /var/lib/apt/lists/*"
    )


def install_go(version: str = "1.22") -> str:
    """Install the Go toolchain from the official tarball.

    Downloads go<version>.linux-amd64.tar.gz from go.dev. Adds /usr/local/go/bin
    to PATH. Common versions: 1.21, 1.22, 1.23.
    """
    v = (version or "1.22").strip()
    return (
        "RUN apt-get update && apt-get install -y --no-install-recommends wget ca-certificates && \\\n"
        f"    wget -q https://go.dev/dl/go{v}.linux-amd64.tar.gz && \\\n"
        f"    tar -C /usr/local -xzf go{v}.linux-amd64.tar.gz && \\\n"
        f"    rm go{v}.linux-amd64.tar.gz && \\\n"
        "    rm -rf /var/lib/apt/lists/*\n"
        "ENV PATH=$PATH:/usr/local/go/bin"
    )


def install_ruby(version: str = "") -> str:
    """Install Ruby from apt (uses the distribution's default Ruby version)."""
    return _apt("ruby ruby-dev")


def install_cmake(version: str = "") -> str:
    """Install CMake and the C/C++ build toolchain from apt."""
    return _apt("cmake build-essential pkg-config")


def install_maven(version: str = "") -> str:
    """Install Apache Maven from apt. Requires Java; combine with install_jdk."""
    return _apt("maven")


def install_gradle(version: str = "8.5") -> str:
    """Download and install Gradle from the official distribution.

    Installs to /opt/gradle-<version> and symlinks the binary to /usr/local/bin.
    Common versions: 7.6, 8.0, 8.5, 8.7.
    """
    v = (version or "8.5").strip()
    return (
        "RUN apt-get update && apt-get install -y --no-install-recommends wget unzip && \\\n"
        f"    wget -q https://services.gradle.org/distributions/gradle-{v}-bin.zip && \\\n"
        f"    unzip -q gradle-{v}-bin.zip -d /opt && \\\n"
        f"    ln -s /opt/gradle-{v}/bin/gradle /usr/local/bin/gradle && \\\n"
        f"    rm gradle-{v}-bin.zip && \\\n"
        "    rm -rf /var/lib/apt/lists/*"
    )


def install_build_essential(version: str = "") -> str:
    """Install GCC, G++, Make, and pkg-config from apt (C/C++ build basics)."""
    return _apt("build-essential pkg-config")


def install_elixir(version: str = "") -> str:
    """Install Elixir and Erlang/OTP from apt."""
    return _apt("elixir erlang-dev erlang-parsetools erlang-tools")


def install_dotnet(version: str = "8") -> str:
    """Install the .NET SDK via the official dotnet-install script.

    Installs to /usr/local/share/dotnet. Common versions: 6, 7, 8, 9.
    """
    v = (version or "8").strip()
    return (
        "RUN apt-get update && apt-get install -y --no-install-recommends wget ca-certificates && \\\n"
        f"    wget -qO dotnet-install.sh https://dot.net/v1/dotnet-install.sh && \\\n"
        f"    bash dotnet-install.sh --channel {v}.0 --install-dir /usr/local/share/dotnet && \\\n"
        "    ln -sf /usr/local/share/dotnet/dotnet /usr/local/bin/dotnet && \\\n"
        "    rm dotnet-install.sh && \\\n"
        "    rm -rf /var/lib/apt/lists/*"
    )


def install_php(version: str = "") -> str:
    """Install PHP-CLI, common extensions, and Composer from apt."""
    return _apt(
        "php-cli php-xml php-mbstring php-curl php-zip unzip curl",
        "curl -sS https://getcomposer.org/installer | php -- --install-dir=/usr/local/bin --filename=composer",
    )


def install_pip_requirements(version: str = "") -> str:
    """Install Python dependencies from requirements.txt.

    Assumes requirements.txt has been COPYed into the working directory first.
    Use 'COPY requirements*.txt ./' before this snippet.
    """
    return (
        "COPY requirements*.txt ./\n"
        "RUN pip install --no-cache-dir -r requirements.txt"
    )


def install_npm_ci(version: str = "") -> str:
    """Install Node.js dependencies via npm ci (reproducible, ignores package-lock changes).

    Assumes package.json and package-lock.json are COPYed first.
    """
    return (
        "COPY package*.json ./\n"
        "RUN npm ci --omit=dev"
    )


def install_yarn_frozen(version: str = "") -> str:
    """Install Node.js dependencies via yarn with frozen lockfile.

    Assumes package.json and yarn.lock are COPYed first.
    """
    return (
        "COPY package.json yarn.lock ./\n"
        "RUN yarn install --frozen-lockfile --non-interactive"
    )


def install_poetry(version: str = "") -> str:
    """Install Poetry and project dependencies.

    Assumes pyproject.toml and poetry.lock are COPYed first.
    Installs without dev dependencies in a virtualenv-free mode.
    """
    return (
        "RUN pip install --no-cache-dir poetry\n"
        "COPY pyproject.toml poetry.lock ./\n"
        "RUN poetry config virtualenvs.create false && poetry install --no-dev --no-interaction --no-ansi"
    )


def install_sbt(version: str = "") -> str:
    """Install sbt (Scala Build Tool) from apt. Requires JDK; combine with install_jdk."""
    return (
        "RUN apt-get update && apt-get install -y --no-install-recommends curl gnupg && \\\n"
        "    curl -fsSL 'https://keyserver.ubuntu.com/pks/lookup?op=get&search=0x2EE0EA64E40A89B84B2DF73499E82A75642AC823' | apt-key add - && \\\n"
        "    echo 'deb https://repo.scala-sbt.org/scalasbt/debian all main' > /etc/apt/sources.list.d/sbt.list && \\\n"
        "    apt-get update && apt-get install -y --no-install-recommends sbt && \\\n"
        "    rm -rf /var/lib/apt/lists/*"
    )


def list_actions(version: str = "") -> str:
    """Return all available snippet action names and their descriptions."""
    lines = [
        "install_jdk(version)       — OpenJDK JDK from apt (default: 17; options: 11, 17, 21)",
        "install_jre(version)       — OpenJDK JRE from apt (default: 17; smaller than full JDK)",
        "install_node(version)      — Node.js via NodeSource (default: 20; options: 18, 20, 22)",
        "install_cargo              — Rust + Cargo via rustup",
        "install_go(version)        — Go tarball from go.dev (default: 1.22)",
        "install_ruby               — Ruby from apt",
        "install_cmake              — CMake + build-essential",
        "install_maven              — Apache Maven from apt (requires JDK)",
        "install_gradle(version)    — Gradle from official distribution (default: 8.5)",
        "install_build_essential    — GCC, G++, Make, pkg-config",
        "install_elixir             — Elixir + Erlang/OTP from apt",
        "install_dotnet(version)    — .NET SDK via install script (default: 8)",
        "install_php                — PHP-CLI + common extensions + Composer",
        "install_pip_requirements   — pip install -r requirements.txt (adds COPY first)",
        "install_npm_ci             — npm ci (adds COPY package*.json first)",
        "install_yarn_frozen        — yarn install --frozen-lockfile (adds COPY first)",
        "install_poetry             — Poetry install (adds COPY pyproject.toml first)",
        "install_sbt                — sbt Scala build tool (requires JDK)",
        "list_actions               — Return this list of available actions",
    ]
    return "\n".join(lines)


_ACTIONS: dict[str, object] = {
    "install_jdk": install_jdk,
    "install_jre": install_jre,
    "install_node": install_node,
    "install_cargo": install_cargo,
    "install_go": install_go,
    "install_ruby": install_ruby,
    "install_cmake": install_cmake,
    "install_maven": install_maven,
    "install_gradle": install_gradle,
    "install_build_essential": install_build_essential,
    "install_elixir": install_elixir,
    "install_dotnet": install_dotnet,
    "install_php": install_php,
    "install_pip_requirements": install_pip_requirements,
    "install_npm_ci": install_npm_ci,
    "install_yarn_frozen": install_yarn_frozen,
    "install_poetry": install_poetry,
    "install_sbt": install_sbt,
    "list_actions": list_actions,
}


def get_snippet(action: str, version: str = "") -> str:
    """Look up a snippet by action name and return the Dockerfile text.

    Returns an error string if the action is not recognized.
    Call list_actions to see all available actions.
    """
    fn = _ACTIONS.get((action or "").strip().lower())
    if fn is None:
        known = ", ".join(sorted(_ACTIONS))
        return f"ERROR: unknown action '{action}'. Known actions: {known}"
    return fn(version)  # type: ignore[call-arg,operator]
