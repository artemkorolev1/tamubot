FROM python:3.14-slim
WORKDIR /workspace

ENV PIP_USER=0

# System deps + Bun
RUN apt-get update && apt-get install -y \
    git curl build-essential nodejs npm unzip \
    && rm -rf /var/lib/apt/lists/* \
    && curl -fsSL https://bun.sh/install | BUN_INSTALL=/usr/local bash

# Python deps (only re-runs when requirements.txt / pyproject.toml change)
COPY requirements.txt pyproject.toml ./
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir playwright

# Playwright: install Chromium browser + its OS deps
ENV PLAYWRIGHT_BROWSERS_PATH=/opt/playwright
RUN playwright install --with-deps chromium

# Source code + editable install (cheap; re-runs on src/ changes)
COPY src/ src/
RUN pip install --no-cache-dir -e ".[v4]"

# Non-root user (required for claude --dangerously-skip-permissions)
RUN useradd -m -s /bin/bash claude && chown -R claude:claude /workspace

# Bake ccstatusline config so it persists across container restarts
COPY .ccstatusline-settings.json /home/claude/.config/ccstatusline/settings.json
RUN chown -R claude:claude /home/claude/.config

USER claude

# Claude Code CLI (native install — always pulls latest on rebuild)
RUN curl -fsSL https://claude.ai/install.sh | bash
ENV PATH="/home/claude/.local/bin:${PATH}"

CMD ["bash"]
