# CPU-only sandbox for reader agents: Python with NumPy and CPU PyTorch, and the opencode agent CLI.
# Run it without GPU access and without internet access; the reader's model endpoint is the only service it needs.
FROM python:3.12-slim-bookworm

RUN apt-get update && apt-get install -y --no-install-recommends nodejs npm ca-certificates procps \
    && rm -rf /var/lib/apt/lists/*
RUN npm install -g opencode-ai
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch \
    && pip install --no-cache-dir numpy scipy safetensors
# the agent's Python calls go through the logging wrapper in /work/bin, which execs /usr/bin/python3
RUN ln -sf /usr/local/bin/python3 /usr/bin/python3
