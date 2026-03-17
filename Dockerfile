# syntax=docker/dockerfile:1.7
# CPU base (Debian slim). For GPU, use the GPU Dockerfile below or override in compose.
FROM python:3.10-slim AS base

ARG UID=1000
ARG GID=1000

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    UV_LINK_MODE=copy

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
        git build-essential curl ca-certificates ffmpeg \
        libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*


# Workdir and user
WORKDIR /workspace

RUN set -eux; \
    GN="$(getent group "${GID}" | cut -d: -f1 || true)"; \
    if [ -z "${GN}" ]; then \
        groupadd -g "${GID}" app; \
        GN="app"; \
    fi; \
    id -u "${UID}" >/dev/null 2>&1 || useradd -m -u "${UID}" -g "${GN}" -s /bin/bash app; \
    mkdir -p /workspace; chown -R "${UID}:${GID}" /workspace

USER app

# Install uv for the non-root user and ensure it's on PATH
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/home/app/.local/bin:${PATH}"

# Create a user-owned virtual environment and use it as default Python
ENV VENV_PATH="/home/app/.venv"
RUN python -m venv "$VENV_PATH"
ENV PATH="$VENV_PATH/bin:${PATH}"

# Copy pyproject and source for layer caching
COPY --chown=app:app pyproject.toml README.md /workspace/
COPY --chown=app:app src /workspace/src

# Install project (editable) + dev tools (kernel), using user-local uv path
RUN ~/.local/bin/uv pip install -e . && \
    ~/.local/bin/uv pip install jupyter ipykernel && \
    python -m ipykernel install --user --name p10-semi --display-name "p10-semi"

# Copy remaining (notebooks/scripts/config)
COPY --chown=app:app notebooks /workspace/notebooks

# Default command: bash
CMD ["bash"]