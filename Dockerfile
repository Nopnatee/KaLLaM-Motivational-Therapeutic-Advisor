# syntax=docker/dockerfile:1

FROM python:3.11-slim AS app

# Environment
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    GRADIO_SERVER_NAME=0.0.0.0 \
    GRADIO_SERVER_PORT=8080 \
    PORT=8080

# Allow overriding the app entry file at build/run time
ARG APP_FILE=gui/chatbot_dev_app.py
ENV APP_FILE=${APP_FILE}

WORKDIR /app

# System deps (minimal). Add more if your libs require.
RUN apt-get update \
 && apt-get install -y --no-install-recommends git \
 && rm -rf /var/lib/apt/lists/*

# Copy project files
COPY pyproject.toml README.md LICENSE ./
COPY src ./src
# Copy app scripts and GUIs (so the entry script is present)
COPY gui ./gui
COPY scripts ./scripts

# Install Python deps and the package
RUN python -m pip install --upgrade pip \
 && pip install .

EXPOSE 8080

# Note: Gradio reads GRADIO_SERVER_NAME/PORT; many platforms also use $PORT
# Use a shell to allow $APP_FILE expansion
CMD ["sh", "-c", "python $APP_FILE"]
