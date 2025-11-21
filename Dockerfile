# ============================================================================
# Multi-Stage Dockerfile for ML Operations Environment
# ============================================================================
# This Dockerfile uses multi-stage builds to create optimized images for
# different services while sharing common base layers for efficiency.
#
# Build targets:
#   - ml-model: Machine learning model service
#   - dashboard: Streamlit dashboard service
#
# Usage:
#   docker build --target ml-model -t ml-model .
#   docker build --target dashboard -t dashboard .
# ============================================================================


# ============================================
# BASE STAGE - Shared Foundation
# ============================================
# This stage contains all common dependencies and setup shared by all services.
# By keeping this as a separate stage, Docker can cache and reuse these layers.

FROM python:3.11-slim AS base

# Build-time arguments for version tracking
ARG VERSION_TAG="unknown"
ARG GIT_COMMIT="unknown"
ARG BUILD_TIMESTAMP="unknown"

# Set as environment variables so they're available at runtime
ENV IMAGE_VERSION=${VERSION_TAG} \
    IMAGE_COMMIT=${GIT_COMMIT} \
    IMAGE_BUILD_TIME=${BUILD_TIMESTAMP}

# Copy uv package manager from official image (ultra-fast Python package installer)
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Set working directory
WORKDIR /app

# Install common system dependencies
# gcc/g++: Required for compiling Python packages with C extensions (numpy, pandas, etc.)
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Configure environment variables
ENV UV_SYSTEM_PYTHON=1 \
    UV_COMPILE_BYTECODE=1 \
    PYTHONUNBUFFERED=1

# Copy shared models folder (MongoDB connector used by all services)
# This ensures all services have access to database connection logic
COPY models/ ./models/


# ============================================
# ML MODEL SERVICE TARGET
# ============================================
# This stage builds the machine learning model service.
# Extends the base stage with ML-specific dependencies and code.

FROM base AS ml-model

# Copy and install ML service dependencies
COPY ml_model/requirements.txt .
RUN uv pip install --no-cache -r requirements.txt

# Copy application code
COPY ml_model/ ./ml_model/

# Expose port for API (if needed)
EXPOSE 5000

# Run the ML pipeline
# Note: Adjust the entrypoint based on your actual main file location
CMD ["python", "-m", "ml_model.main"]


# ============================================
# STREAMLIT DASHBOARD SERVICE TARGET
# ============================================
# This stage builds the Streamlit dashboard service.
# Extends the base stage with dashboard-specific dependencies and code.

FROM base AS dashboard

# Install additional dependencies needed for Streamlit
# curl: Required for health checks
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy and install dashboard dependencies
COPY dashboard/requirements.txt .
RUN uv pip install --no-cache -r requirements.txt

# Copy application code
COPY dashboard/ ./dashboard/

# Expose Streamlit default port
EXPOSE 8501

# Health check to ensure Streamlit is running properly
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health || exit 1

# Run Streamlit application
CMD ["streamlit", "run", "dashboard/dashboard.py", "--server.port=8501", "--server.address=0.0.0.0"]
