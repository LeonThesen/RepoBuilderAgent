"""Validated Dockerfile RUN-block snippets for common build toolchains.

Each function returns one or more Dockerfile lines as a string ready to paste.
Snippets target the project's Debian/Ubuntu base image, where the build runs as a
NON-ROOT user (manualrepos) with passwordless sudo — so every privileged step (apt,
writes under /usr/local, /opt) is prefixed with `sudo`. Where network is required
(Go, Gradle, .NET) the download URL follows official release conventions.
"""

from __future__ import annotations

# The non-root build user's home, where user-scoped toolchains (rustup) install.
_USER_HOME = "/home/manualrepos"


def _apt(packages: str, extras: str = "") -> str:
    """Return an apt-get RUN block (sudo, non-root base) with cache cleanup."""
    body = f"sudo apt-get update && sudo apt-get install -y --no-install-recommends {packages}"
    if extras:
        body += f" && {extras}"
    body += " && sudo rm -rf /var/lib/apt/lists/*"
    return f"RUN {body}"


def install_apt(packages: str = "") -> str:
    """Install an arbitrary set of apt packages (space-separated) the right way.

    Example: get_dockerfile_snippet("install_apt", "libssl-dev zlib1g-dev pkg-config").
    Use this for system dev libraries (libssl-dev, zlib1g-dev, libffi-dev, libopenblas-dev,
    gfortran, ninja-build, autoconf, automake, libtool, protobuf-compiler, ...).
    """
    pkgs = (packages or "").strip()
    if not pkgs:
        return "ERROR: install_apt requires a space-separated package list (the `version` arg)."
    return _apt(pkgs)


def install_jdk(version: str = "") -> str:
    """Install the JDK from apt via the UNVERSIONED `default-jdk` meta-package.

    NOTE: the base image ALREADY ships `default-jdk` on PATH with JAVA_HOME set, so a
    JVM build normally needs NO JDK install at all — use this only if the base JDK was
    somehow removed. The `version` arg is intentionally ignored: on Ubuntu 24.04 the
    unversioned `default-jdk` resolves to openjdk-21 (an in-range LTS), which satisfies
    most toolchains. Do not chase a repo's declared JDK version with a pinned package;
    for a newer JDK than Ubuntu 24.04 ships, let the build's toolchain auto-download fetch it.
    """
    return _apt("default-jdk")


def install_jre(version: str = "") -> str:
    """Install the JRE from apt via the unversioned `default-jre` meta-package.

    The base image already ships default-jdk; see install_jdk. The `version` arg is
    ignored — on Ubuntu 24.04 the unversioned `default-jre` resolves to an in-range LTS JRE.
    """
    return _apt("default-jre")


def install_node(version: str = "20") -> str:
    """Install Node.js via NodeSource setup script. Common versions: 18, 20, 22."""
    v = (version or "20").strip()
    return (
        "RUN sudo apt-get update && sudo apt-get install -y --no-install-recommends curl ca-certificates && \\\n"
        f"    curl -fsSL https://deb.nodesource.com/setup_{v}.x | sudo -E bash - && \\\n"
        "    sudo apt-get install -y --no-install-recommends nodejs && \\\n"
        "    sudo rm -rf /var/lib/apt/lists/*"
    )


def install_pnpm(version: str = "") -> str:
    """Install the pnpm package manager globally (requires Node.js; install_node first)."""
    return "RUN sudo corepack enable && corepack prepare pnpm@latest --activate"


def install_cargo(version: str = "") -> str:
    """Install current Rust + Cargo via rustup, WITHOUT sudo so it lands in the build
    user's home and stays on PATH.

    Ubuntu 24.04's apt `cargo`/`rustc` is 1.75 — too old for modern crates and Cargo.lock
    v4 (`failed to parse lock file`), so rustup is the reliable choice. Never `sudo` the
    installer (it would land in root's home, unreachable by the non-root build user).
    """
    return (
        "RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal\n"
        "ENV PATH=/home/manualrepos/.cargo/bin:$PATH"
    )


def install_go(version: str = "1.22") -> str:
    """Install Go from the official tarball into /usr/local/go. Common versions: 1.21, 1.22, 1.23."""
    v = (version or "1.22").strip()
    return (
        "RUN sudo apt-get update && sudo apt-get install -y --no-install-recommends curl ca-certificates && \\\n"
        f"    curl -fsSL https://go.dev/dl/go{v}.linux-amd64.tar.gz -o /tmp/go.tgz && \\\n"
        "    sudo tar -C /usr/local -xzf /tmp/go.tgz && rm /tmp/go.tgz && \\\n"
        "    sudo rm -rf /var/lib/apt/lists/*\n"
        "ENV PATH=/usr/local/go/bin:$PATH"
    )


def install_ruby(version: str = "") -> str:
    """Install Ruby and bundler from apt."""
    return _apt("ruby-full", "sudo gem install bundler --no-document")


def install_cmake(version: str = "") -> str:
    """Install CMake and the C/C++ build toolchain from apt."""
    return _apt("cmake build-essential pkg-config")


def install_maven(version: str = "") -> str:
    """Install Apache Maven from apt. Requires Java; combine with install_jdk."""
    return _apt("maven")


def install_gradle(version: str = "8.5") -> str:
    """Download and install Gradle from the official distribution into /opt.

    Common versions: 7.6, 8.0, 8.5, 8.7. Requires Java; combine with install_jdk.
    """
    v = (version or "8.5").strip()
    return (
        "RUN sudo apt-get update && sudo apt-get install -y --no-install-recommends wget unzip && \\\n"
        f"    wget -q https://services.gradle.org/distributions/gradle-{v}-bin.zip -O /tmp/gradle.zip && \\\n"
        "    sudo unzip -q /tmp/gradle.zip -d /opt && rm /tmp/gradle.zip && \\\n"
        f"    sudo ln -s /opt/gradle-{v}/bin/gradle /usr/local/bin/gradle && \\\n"
        "    sudo rm -rf /var/lib/apt/lists/*"
    )


def install_build_essential(version: str = "") -> str:
    """Install GCC, G++, Make, and pkg-config from apt (C/C++ build basics)."""
    return _apt("build-essential pkg-config")


def install_autotools(version: str = "") -> str:
    """Install the GNU Autotools build chain (autoconf, automake, libtool) from apt."""
    return _apt("autoconf automake libtool pkg-config build-essential")


def install_elixir(version: str = "") -> str:
    """Install Elixir and Erlang/OTP from apt."""
    return _apt("elixir erlang-dev erlang-parsetools erlang-tools")


def install_dotnet(version: str = "8") -> str:
    """Install the .NET SDK via the official dotnet-install script. Common versions: 6, 7, 8, 9."""
    v = (version or "8").strip()
    return (
        "RUN sudo apt-get update && sudo apt-get install -y --no-install-recommends wget ca-certificates && \\\n"
        "    wget -qO /tmp/dotnet-install.sh https://dot.net/v1/dotnet-install.sh && \\\n"
        f"    sudo bash /tmp/dotnet-install.sh --channel {v}.0 --install-dir /usr/local/share/dotnet && \\\n"
        "    sudo ln -sf /usr/local/share/dotnet/dotnet /usr/local/bin/dotnet && rm /tmp/dotnet-install.sh && \\\n"
        "    sudo rm -rf /var/lib/apt/lists/*"
    )


def install_php(version: str = "") -> str:
    """Install PHP-CLI, common extensions, and Composer from apt."""
    return _apt(
        "php-cli php-xml php-mbstring php-curl php-zip unzip curl",
        "curl -sS https://getcomposer.org/installer | php -- --install-dir=/tmp && sudo mv /tmp/composer.phar /usr/local/bin/composer",
    )


def install_pip_requirements(version: str = "") -> str:
    """Install Python dependencies from requirements.txt (COPY it in first)."""
    return (
        "COPY requirements*.txt ./\n"
        "RUN pip install --no-cache-dir -r requirements.txt"
    )


def install_npm_ci(version: str = "") -> str:
    """Install Node.js dependencies via npm ci (COPY package*.json first)."""
    return (
        "COPY package*.json ./\n"
        "RUN npm ci --omit=dev"
    )


def install_yarn_frozen(version: str = "") -> str:
    """Install Node.js dependencies via yarn with a frozen lockfile (COPY package.json yarn.lock first)."""
    return (
        "COPY package.json yarn.lock ./\n"
        "RUN yarn install --frozen-lockfile --non-interactive"
    )


def install_pnpm_frozen(version: str = "") -> str:
    """Install Node.js dependencies via pnpm with a frozen lockfile (requires install_pnpm; COPY lockfile first)."""
    return (
        "COPY package.json pnpm-lock.yaml ./\n"
        "RUN pnpm install --frozen-lockfile"
    )


def install_poetry(version: str = "") -> str:
    """Install Poetry and project dependencies (COPY pyproject.toml poetry.lock first)."""
    return (
        "RUN pip install --no-cache-dir poetry\n"
        "COPY pyproject.toml poetry.lock ./\n"
        "RUN poetry config virtualenvs.create false && poetry install --no-dev --no-interaction --no-ansi"
    )


def install_sbt(version: str = "") -> str:
    """Install sbt (Scala Build Tool) from its apt repo. Requires JDK; combine with install_jdk."""
    return (
        "RUN sudo apt-get update && sudo apt-get install -y --no-install-recommends curl gnupg && \\\n"
        "    curl -fsSL 'https://keyserver.ubuntu.com/pks/lookup?op=get&search=0x2EE0EA64E40A89B84B2DF73499E82A75642AC823' | sudo apt-key add - && \\\n"
        "    echo 'deb https://repo.scala-sbt.org/scalasbt/debian all main' | sudo tee /etc/apt/sources.list.d/sbt.list && \\\n"
        "    sudo apt-get update && sudo apt-get install -y --no-install-recommends sbt && \\\n"
        "    sudo rm -rf /var/lib/apt/lists/*"
    )


def list_actions(version: str = "") -> str:
    """Return all available snippet action names and their descriptions."""
    lines = [
        "install_apt(packages)      — Install arbitrary apt packages (pass space-separated list as the arg)",
        "install_jdk               — default-jdk (unversioned; base already ships it; version arg ignored — do not version-chase)",
        "install_jre               — default-jre (unversioned; base already ships the JDK; smaller than full JDK)",
        "install_node(version)      — Node.js via NodeSource (default: 20; options: 18, 20, 22)",
        "install_pnpm               — pnpm package manager via corepack (requires Node)",
        "install_cargo              — Rust + Cargo via apt (rustup alternative in docstring)",
        "install_go(version)        — Go tarball from go.dev into /usr/local (default: 1.22)",
        "install_ruby               — Ruby + bundler from apt",
        "install_cmake              — CMake + build-essential + pkg-config",
        "install_autotools          — autoconf + automake + libtool + build-essential",
        "install_maven              — Apache Maven from apt (requires JDK)",
        "install_gradle(version)    — Gradle from official distribution (default: 8.5; requires JDK)",
        "install_build_essential    — GCC, G++, Make, pkg-config",
        "install_elixir             — Elixir + Erlang/OTP from apt",
        "install_dotnet(version)    — .NET SDK via install script (default: 8)",
        "install_php                — PHP-CLI + common extensions + Composer",
        "install_pip_requirements   — pip install -r requirements.txt (adds COPY first)",
        "install_npm_ci             — npm ci (adds COPY package*.json first)",
        "install_yarn_frozen        — yarn install --frozen-lockfile (adds COPY first)",
        "install_pnpm_frozen        — pnpm install --frozen-lockfile (adds COPY first; needs install_pnpm)",
        "install_poetry             — Poetry install (adds COPY pyproject.toml first)",
        "install_sbt                — sbt Scala build tool (requires JDK)",
        "list_actions               — Return this list of available actions",
    ]
    return "\n".join(lines)


_ACTIONS: dict[str, object] = {
    "install_apt": install_apt,
    "install_jdk": install_jdk,
    "install_jre": install_jre,
    "install_node": install_node,
    "install_pnpm": install_pnpm,
    "install_cargo": install_cargo,
    "install_go": install_go,
    "install_ruby": install_ruby,
    "install_cmake": install_cmake,
    "install_autotools": install_autotools,
    "install_maven": install_maven,
    "install_gradle": install_gradle,
    "install_build_essential": install_build_essential,
    "install_elixir": install_elixir,
    "install_dotnet": install_dotnet,
    "install_php": install_php,
    "install_pip_requirements": install_pip_requirements,
    "install_npm_ci": install_npm_ci,
    "install_yarn_frozen": install_yarn_frozen,
    "install_pnpm_frozen": install_pnpm_frozen,
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
