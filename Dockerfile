FROM python:3.10-slim

WORKDIR /app

# Install system dependencies for building Python packages
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (for better layer caching)
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY . .

# Create data directory for SQLite database
RUN mkdir -p /app/data

# Expose port (Hugging Face Spaces uses 7860)
EXPOSE 7860

# Set environment variables for Hugging Face Spaces
ENV PORT=7860 \
    FLASK_ENV=production

# Run the Flask app on port 7860
CMD ["python", "app.py"]
