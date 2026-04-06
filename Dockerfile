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
RUN pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cpu -r requirements.txt

# Copy application files
COPY app.py .
COPY fusion_inference.py .

# Copy model files from root directory
COPY cnn_lstm_audio_model_scripted.pt .
COPY elm_model.pkl .
COPY fusion_model_config.json .

# Expose Streamlit port
EXPOSE 8501

# Set Streamlit config
ENV STREAMLIT_SERVER_PORT=8501
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0
ENV STREAMLIT_SERVER_HEADLESS=true

# Run Streamlit
CMD ["streamlit", "run", "app.py"]
