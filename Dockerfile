# Build stage
FROM python:3.9-slim as builder

# Install build dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    libusb-1.0-0-dev \
    && rm -rf /var/lib/apt/lists/*

# Install poetry
RUN curl -sSL https://install.python-poetry.org | python3 -
ENV PATH="/root/.local/bin:$PATH"

WORKDIR /app

# Copy dependency files
COPY pyproject.toml poetry.lock ./

# Create minimal README if not exists (required by Poetry)
RUN echo "# Coral Vision" > README.md || true

# Install dependencies (without dev dependencies)
RUN poetry config virtualenvs.in-project true && \
    poetry install --no-root --without dev

# Runtime stage
FROM python:3.9-slim

# Install Edge TPU runtime library and dependencies
RUN apt-get update && apt-get install -y \
    curl \
    gnupg \
    ca-certificates \
    && curl -fsSL https://packages.cloud.google.com/apt/doc/apt-key.gpg | gpg --dearmor -o /usr/share/keyrings/coral-edgetpu-archive-keyring.gpg \
    && echo "deb [signed-by=/usr/share/keyrings/coral-edgetpu-archive-keyring.gpg] https://packages.cloud.google.com/apt coral-edgetpu-stable main" | tee /etc/apt/sources.list.d/coral-edgetpu.list \
    && apt-get update && apt-get install -y \
    libedgetpu1-std \
    libusb-1.0-0 \
    libgomp1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgl1 \
    espeak \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy virtual environment from builder
COPY --from=builder /app/.venv /app/.venv

# Copy README from builder (required by Poetry metadata)
COPY --from=builder /app/README.md ./

# Copy application code
COPY coral_vision ./coral_vision
COPY pyproject.toml poetry.lock ./

# Install the application using pip (Poetry not needed in runtime)
ENV PATH="/app/.venv/bin:$PATH"
RUN /app/.venv/bin/pip install -e . --no-deps

# Create data directory for models only
RUN mkdir -p /app/data/models

# Environment variables
ENV PYTHONUNBUFFERED=1
ENV DATA_DIR=/app/data
ENV DB_HOST=postgres
ENV DB_PORT=5432
ENV DB_NAME=coral_vision
ENV DB_USER=coral
ENV DB_PASSWORD=coral

# Expose Flask port
EXPOSE 5000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:5000/health || exit 1

# Entrypoint script to handle USE_EDGETPU and SSL environment variables
COPY <<'EOF' /app/entrypoint.sh
#!/bin/bash
set -e

# Build command arguments
ARGS="serve --host 0.0.0.0 --port 5000"

# Add Edge TPU flag if enabled
if [ "${USE_EDGETPU}" = "true" ]; then
    ARGS="$ARGS --use-edgetpu"
fi

# Add SSL certificates if SSL is enabled
if [ "${USE_SSL}" = "true" ]; then
    if [ -n "${SSL_CERT}" ] && [ -n "${SSL_KEY}" ]; then
        if [ -f "${SSL_CERT}" ] && [ -f "${SSL_KEY}" ]; then
            ARGS="$ARGS --ssl-cert ${SSL_CERT} --ssl-key ${SSL_KEY}"
            echo "🔒 SSL certificates found, enabling HTTPS"
        else
            echo "⚠️  SSL_CERT or SSL_KEY files not found, using HTTP"
        fi
    elif [ -f "/app/certs/cert.pem" ] && [ -f "/app/certs/key.pem" ]; then
        # Auto-detect certificates in /app/certs directory
        ARGS="$ARGS --ssl-cert /app/certs/cert.pem --ssl-key /app/certs/key.pem"
        echo "🔒 Auto-detected SSL certificates, enabling HTTPS"
    else
        echo "⚠️  USE_SSL=true but no certificates found, using HTTP"
    fi
fi

# Run the application
exec coral-vision $ARGS
EOF

RUN chmod +x /app/entrypoint.sh

# Run the application
ENTRYPOINT ["/app/entrypoint.sh"]
