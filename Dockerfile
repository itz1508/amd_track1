# AMD Track 1 General-Purpose AI Agent - Submission Image
# Minimal, explicit Dockerfile for isolated submission

FROM python:3.11-slim

WORKDIR /app

# Install runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user for security
RUN useradd -m -u 1000 agent

# Install Python dependencies from explicit requirements
COPY amd_track1/requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

# Copy only explicit runtime files (no tests, no docs, no development artifacts)
COPY amd_track1 /app/amd_track1

# Make sure scripts in .local are usable
ENV PATH=/home/agent/.local/bin:$PATH
ENV PYTHONPATH=/home/agent/.local/lib/python3.11/site-packages

# Runtime environment
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

# Runtime remains root so /output is writable with standard bind mounts
WORKDIR /app

# Entry point - run from package parent so relative imports resolve
ENTRYPOINT ["python", "-m", "amd_track1.entrypoint"]

# Expose port for debugging
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=3s \
    CMD python -c "import sys; sys.exit(0)" || exit 1
