FROM python:3.9-slim

# Copy uv from official image
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Set environment variables for uv
ENV UV_SYSTEM_PYTHON=1
ENV UV_COMPILE_BYTECODE=1

# Copy requirements first for better caching
COPY classes/requirements.txt .

# Install dependencies using uv (much faster than pip)
RUN uv pip install --no-cache -r requirements.txt

# Copy shared models folder
COPY models/ ./models/

# Copy classes application files
COPY classes/ ./classes/

# Expose port for API
EXPOSE 5000

# Run the ML pipeline
CMD ["python", "-m", "classes.main"]
