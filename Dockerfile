# ==============================================================================
# OpenAlgo Dockerfile - Python 3.14.0 with FastAPI/Uvicorn
# ==============================================================================

# ------------------------------ Builder Stage ------------------------------ #
FROM python:3.14.0-slim-bookworm AS builder

# Install system dependencies for building Python packages
RUN apt-get update && apt-get install -y --no-install-recommends \
        curl \
        build-essential \
        gcc \
        libffi-dev \
        libssl-dev \
        && apt-get clean \
        && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Create virtual environment and install dependencies
RUN python -m venv /app/.venv \
    && /app/.venv/bin/pip install --no-cache-dir --upgrade pip \
    && /app/.venv/bin/pip install --no-cache-dir -r requirements.txt \
    && /app/.venv/bin/pip install --no-cache-dir uvicorn[standard] \
    && rm -rf /root/.cache

# --------------------------------------------------------------------------- #


# ------------------------------ Production Stage --------------------------- #
FROM python:3.14.0-slim-bookworm AS production

# Set timezone to IST (Asia/Kolkata)
RUN apt-get update && apt-get install -y --no-install-recommends \
        tzdata \
        curl \
        && ln -fs /usr/share/zoneinfo/Asia/Kolkata /etc/localtime \
        && dpkg-reconfigure -f noninteractive tzdata \
        && apt-get clean \
        && rm -rf /var/lib/apt/lists/*

# Create non-root user for security
RUN useradd --create-home --shell /bin/bash appuser

# Set working directory
WORKDIR /app

# Copy virtual environment from builder
COPY --from=builder --chown=appuser:appuser /app/.venv /app/.venv

# Copy application code
COPY --chown=appuser:appuser . .

# Create required directories with proper ownership
RUN mkdir -p /app/logs /app/data /app/reports /app/db /app/keys \
    && chown -R appuser:appuser /app/logs /app/data /app/reports /app/db /app/keys

# ------------------------------ Runtime Environment ------------------------ #
ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONFAULTHANDLER=1 \
    TZ=Asia/Kolkata \
    APP_MODE=production

# --------------------------------------------------------------------------- #

# Switch to non-root user
USER appuser

# Expose the application port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run the application with Uvicorn
CMD ["uvicorn", "openalgo.main:app", "--host", "0.0.0.0", "--port", "8000"]
