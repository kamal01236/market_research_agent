# Use official Python image with AMD64 architecture
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy project files
COPY pyproject.toml .
COPY config.yml .
COPY market_research_agent/ ./market_research_agent/
COPY README.md .
COPY LICENSE .

# Install Python dependencies
RUN pip install --no-cache-dir ".[test,dev]"

# Set environment variables
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

# Default command (can be overridden)
CMD ["market-research", "serve", "--config", "config.yml"]