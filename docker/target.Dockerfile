# Sandbox image for target agents + upstream MCP servers (SPEC section 10).
# Build context: repository root.  All base images are pinned by digest.
FROM ghcr.io/astral-sh/uv:0.13.6@sha256:af22a52f5eafef3d78e551599d405c02df61919d18028f0cea8eda93bc9986d0 AS uv

# --- Node: install the pinned filesystem MCP server from the lockfile ---------------------
FROM node:24-slim@sha256:be8204d4bd327363357f532187b752ff62759c31bedf98e9e18fc08c9cbd605c AS mcp
WORKDIR /opt/mcp
COPY docker/mcp/package.json docker/mcp/package-lock.json ./
RUN npm ci --omit=dev --ignore-scripts && rm -rf /root/.npm

# --- Runtime -----------------------------------------------------------------------------
FROM python:3.12-slim@sha256:6b25cbe6577518889d553cdd5a5da97d259348d8f6e45b527dd731caaba27a07

RUN apt-get update \
    && apt-get install -y --no-install-recommends git ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY --from=uv /uv /uvx /usr/local/bin/
COPY --from=mcp /usr/local/bin/node /usr/local/bin/node
COPY --from=mcp /opt/mcp /opt/mcp

ENV UV_TOOL_DIR=/opt/tools/venvs \
    UV_TOOL_BIN_DIR=/opt/tools/bin \
    UV_PYTHON_DOWNLOADS=never \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy
RUN uv tool install mcp-server-git==2026.10.19

WORKDIR /app
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project --no-editable
COPY src ./src
RUN uv sync --frozen --no-dev --no-editable && rm -rf src

ENV PATH=/app/.venv/bin:/opt/tools/bin:/usr/local/bin:/usr/bin:/bin \
    HOME=/tmp \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MCP_CHAOS_SCRATCH=/scratch \
    MCP_CHAOS_UPSTREAM_ENV=docker \
    GIT_CONFIG_COUNT=1 \
    GIT_CONFIG_KEY_0=safe.directory \
    GIT_CONFIG_VALUE_0=/scratch

RUN mkdir -p /scratch /run/in /run/out && chmod 0555 /app
USER 10001:10001
WORKDIR /scratch
ENTRYPOINT ["python", "-m", "mcp_chaos.targets.entry"]
