# Multi-stage Dockerfile for MLOps DistillBERT API

# Stage 1: Build stage
FROM python:3.9-slim as builder

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Create virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# Stage 2: Production stage
FROM python:3.9-slim as production

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/opt/venv/bin:$PATH" \
    HOST=0.0.0.0 \
    PORT=8000 \
    WORKERS=4

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy virtual environment from builder stage
COPY --from=builder /opt/venv /opt/venv

# Create non-root user
RUN groupadd -r mlops && useradd -r -g mlops -s /bin/bash mlops

# Create app directory
WORKDIR /app

# Create necessary directories
RUN mkdir -p /app/models /app/logs /app/config && \
    chown -R mlops:mlops /app

# Copy application code
COPY --chown=mlops:mlops src/ ./src/
COPY --chown=mlops:mlops config/ ./config/
COPY --chown=mlops:mlops requirements.txt .

# Copy entrypoint script
COPY --chown=mlops:mlops docker/entrypoint.sh .
RUN chmod +x entrypoint.sh

# Switch to non-root user
USER mlops

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:${PORT}/health || exit 1

# Expose port
EXPOSE ${PORT}

# Set entrypoint
ENTRYPOINT ["./entrypoint.sh"]
CMD ["start"]