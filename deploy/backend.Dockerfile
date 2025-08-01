# Build stage
FROM mcr.microsoft.com/mirror/docker/library/python:3.11-slim AS builder

# Set working directory
WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies with retry mechanism
COPY backend/requirements.txt .
RUN pip install --no-cache-dir pip==23.3.1 setuptools==69.0.3 wheel==0.42.0 && \
    pip install --no-cache-dir --timeout 300 --retries 5 -r requirements.txt || \
    (pip install --no-cache-dir --timeout 300 --retries 5 cryptography==42.0.2 && \
     pip install --no-cache-dir --timeout 300 --retries 5 -r requirements.txt)

# Production stage
FROM mcr.microsoft.com/mirror/docker/library/python:3.11-slim

# Set working directory
WORKDIR /app

# Install runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libpq5 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy Python dependencies from builder
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Create non-root user first
RUN useradd -m appuser

# Copy application code
COPY backend/ /app/backend/
COPY alembic.ini .
COPY migrations/ migrations/

# Create logs directory and set permissions
RUN mkdir -p /app/logs && \
    chown -R appuser:appuser /app /app/logs

# Switch to non-root user
USER appuser

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app:/app/backend \
    PATH="/home/appuser/.local/bin:$PATH"

# Add health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:${PORT:-10000}/health || exit 1

# Expose port
EXPOSE ${PORT:-10000}

# Start Gunicorn
CMD ["gunicorn", "--bind", "0.0.0.0:10000", \
     "--workers", "4", \
     "--threads", "2", \
     "--timeout", "60", \
     "--worker-class", "gevent", \
     "--worker-connections", "1000", \
     "--access-logfile", "-", \
     "--error-logfile", "-", \
     "--log-level", "info", \
     "--capture-output", \
     "backend.app:app"] 