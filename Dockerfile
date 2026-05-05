FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy and install Python dependencies first (for layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY kno/ ./kno/

# The container runs the FastAPI production app on port 8080
# Cloud Run injects PORT env var (default 8080)
ENV PORT=8080

# Use Vertex AI (no API key needed — uses GCP service account)
ENV GOOGLE_GENAI_USE_VERTEXAI=1

# Start the FastAPI production app
CMD ["sh", "-c", "uvicorn kno.prod_app:app --host 0.0.0.0 --port ${PORT}"]
