FROM python:3.10-slim

WORKDIR /app

# Install system dependencies for building Python packages + curl/zstd for the Ollama installer
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    zstd \
    && rm -rf /var/lib/apt/lists/*

# Install Ollama and pre-pull the embedding model used by retriever.py,
# so PDF-chunk retrieval works the same as it does locally.
RUN curl -fsSL https://ollama.com/install.sh | sh
RUN ollama serve & \
    sleep 5 && \
    ollama pull nomic-embed-text

# Copy requirements first (for better layer caching)
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY . .

# Create data directory for SQLite database
RUN mkdir -p /app/data

RUN chmod +x /app/entrypoint.sh

# Expose port (Hugging Face Spaces uses 7860)
EXPOSE 7860

# Set environment variables for Hugging Face Spaces
ENV PORT=7860 \
    FLASK_ENV=production

# Start Ollama in the background, then the Flask app
CMD ["/app/entrypoint.sh"]
