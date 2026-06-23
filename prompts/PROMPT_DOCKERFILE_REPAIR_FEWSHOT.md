<example index="1">
<user>
Build failed. Log (tail):
  fatal error: openssl/ssl.h: No such file or directory
  compilation terminated. make: *** [Makefile:42: build] Error 1
Current AGENT_BUILD_STEPS region:
  RUN sudo apt-get update && sudo apt-get install -y gcc make && sudo rm -rf /var/lib/apt/lists/*
  RUN make all
Root cause: missing OpenSSL development headers. Fix the smallest thing; keep build steps.
</user>
<assistant>
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
</user>
<assistant>
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
</assistant>
</example>
