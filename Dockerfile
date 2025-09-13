# syntax=docker/dockerfile:1.7
# ---------- builder ----------
FROM python:3.11-slim AS builder

ARG PIP_NO_BINARY=:none:
ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_DEFAULT_TIMEOUT=100 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# OS deps (add as needed)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential git && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy project metadata first for layer caching
COPY pyproject.toml ./
# If you use a lock file (pdm.lock/poetry.lock/hatch.lock), copy it too for better caching
# COPY pdm.lock ./
# COPY poetry.lock ./

# Install build utilities
RUN python -m pip install --upgrade pip wheel build

# Build a wheel from your project
# This invokes your pyproject backend (setuptools/hatch/pdm/…) to produce dist/*.whl
COPY . .
RUN python -m build --wheel --no-isolation

# ---------- runtime ----------
FROM python:3.11-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    GRADIO_SERVER_NAME=0.0.0.0 \
    GRADIO_SERVER_PORT=7860

# Optional: smaller locale, no cache
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy only built wheel(s) and install
COPY --from=builder /app/dist/*.whl /tmp/
RUN pip install --no-cache-dir /tmp/*.whl

# If your app needs extra runtime data files, copy them
# COPY assets/ ./assets/

# Create non-root user
RUN useradd -m appuser && chown -R appuser:appuser /app
USER appuser

EXPOSE 7860

# Ensure your app reads PORT env and binds 0.0.0.0
# Replace app.py with your actual entrypoint
ENV PORT=7860
CMD ["python", "app.py"]
