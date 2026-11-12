# CPU-only sandbox for reader and programmer agents: Python with NumPy, SciPy and CPU PyTorch.
#   docker build -t wsr-sandbox sandbox/
# With the built-in harness the container runs with --network none: the model is called from the host.
# The agent CLIs (opencode, Claude Code, Codex) are included for the external harnesses, which need network
# access to reach their model; drop those lines for a smaller image.
FROM python:3.12-slim-bookworm

RUN apt-get update && apt-get install -y --no-install-recommends nodejs npm ca-certificates procps coreutils \
    && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch \
    && pip install --no-cache-dir numpy scipy safetensors
ARG AGENT_CLIS="opencode-ai @anthropic-ai/claude-code @openai/codex"
RUN if [ -n "$AGENT_CLIS" ]; then npm install -g $AGENT_CLIS; fi
# the agent's Python calls go through the logging wrapper in /work/bin, which execs /usr/bin/python3
RUN ln -sf /usr/local/bin/python3 /usr/bin/python3
RUN useradd -m -u 1000 agent
USER agent
WORKDIR /work
