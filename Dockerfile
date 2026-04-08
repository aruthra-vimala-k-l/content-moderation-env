# -----------------------------------------------------------------------
# Content Moderation OpenEnv Environment
# Hugging Face Spaces / Docker deployment
#
# Build:  docker build -t content-moderation-env .
# Run:    docker run -p 7860:7860 content-moderation-env
# -----------------------------------------------------------------------

FROM python:3.11-slim

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user (HF Spaces requirement)
RUN useradd -m -u 1000 appuser

WORKDIR /app

# Install Python dependencies first (layer cache)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application source
COPY app.py .
COPY environment.py .
COPY models.py .
COPY tasks.py .
COPY openenv.yaml .
COPY inference.py .

# Ownership
RUN chown -R appuser:appuser /app
USER appuser

# HF Spaces exposes port 7860
EXPOSE 7860

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=20s --retries=3 \
    CMD curl -f http://localhost:7860/health || exit 1

# Start the FastAPI server
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860", "--log-level", "info"]
