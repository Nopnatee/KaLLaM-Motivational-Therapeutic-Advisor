# syntax=docker/dockerfile:1
FROM python:3.11-slim AS app

# Minimal env
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    GRADIO_SERVER_NAME=0.0.0.0 \
    GRADIO_SERVER_PORT=8080 \
    PORT=8080

# Allow overriding entry script
ARG APP_FILE=gui/chatbot_dev_app.py
ENV APP_FILE=${APP_FILE}

WORKDIR /app

# OS deps (git is often needed for pip VCS URLs)
RUN apt-get update \
 && apt-get install -y --no-install-recommends git \
 && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md LICENSE ./

RUN python -m pip install --upgrade pip setuptools wheel

# Bring in source last to avoid invalidating earlier cache layers
COPY src ./src
COPY gui ./gui
COPY scripts ./scripts

# Install your package
RUN pip install --no-cache-dir .

EXPOSE 8080

# Launch the chosen app file; App Runner sets $PORT for us
CMD ["sh", "-c", "python $APP_FILE"]
