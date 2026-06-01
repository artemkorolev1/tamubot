FROM python:3.14-slim-bookworm

# System deps + make (for the project's Makefile targets) + gh + git/curl/jq.
RUN apt-get update && apt-get install -y --no-install-recommends \
      git \
      curl \
      jq \
      make \
      ca-certificates \
      gnupg \
  && rm -rf /var/lib/apt/lists/*

# GitHub CLI
RUN curl -fsSL https://cli.github.com/packages/githubcli-archive-keyring.gpg \
      | dd of=/usr/share/keyrings/githubcli-archive-keyring.gpg \
  && echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/githubcli-archive-keyring.gpg] https://cli.github.com/packages stable main" \
      > /etc/apt/sources.list.d/github-cli.list \
  && apt-get update && apt-get install -y --no-install-recommends gh \
  && rm -rf /var/lib/apt/lists/*

# UID/GID alignment so bind-mounted files share an owner with the host.
ARG AGENT_UID=1000
ARG AGENT_GID=1000
RUN groupadd -o -g $AGENT_GID agent \
  && useradd -o -u $AGENT_UID -g $AGENT_GID -m -d /home/agent agent
USER ${AGENT_UID}:${AGENT_GID}

# Claude Code CLI (standalone installer, no Node required).
RUN curl -fsSL https://claude.ai/install.sh | bash

# .local/bin contains both the Claude CLI and the pip --user console scripts.
ENV PATH="/home/agent/.local/bin:$PATH"

# Verification tooling — these power `make lint` / `make typecheck` / `make test`.
# Heavy runtime deps (torch, transformers, bitsandbytes, flash-linear-attention,
# opendataloader-pdf, dagster, docling-slim) are intentionally NOT installed —
# they need CUDA / Java / large wheels that don't belong in an ephemeral CPU
# sandbox. mypy is configured with ignore_missing_imports = true, so typecheck
# will pass even with the heavy deps absent. `make test` may surface
# ModuleNotFoundError for those deps — the prompts treat that as expected and
# commit on the strength of lint + typecheck.
RUN pip install --user --no-cache-dir \
      ruff \
      mypy \
      pytest \
      pytest-asyncio \
      pydantic \
      python-dotenv \
      pyyaml

WORKDIR /home/agent
# Sandcastle bind-mounts the worktree at /home/agent/workspace and overrides
# the working directory there at container start.
ENTRYPOINT ["sleep", "infinity"]
