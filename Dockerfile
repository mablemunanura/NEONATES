# Use Python 3.11 slim image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (for better caching)
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY app.py .

# Copy model files from notebooks/MODELS directory
COPY notebooks/MODELS/ notebooks/MODELS/

# Expose Gradio port
EXPOSE 7860

# Run Gradio
CMD ["python", "app.py"]
