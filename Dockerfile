# Base image matching active environment (Python 3.10)
FROM python:3.10-slim

# Prevent Python from writing .pyc files and enable unbuffered logging
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

# Set working directory inside container
WORKDIR /app

# Install system dependencies required for build/compilation
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first to leverage Docker layer caching
COPY requirements.txt /app/

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code, configs, models, data, and MLflow artifacts
COPY src/ /app/src/
COPY configs/ /app/configs/
COPY models/ /app/models/
COPY data/ /app/data/
COPY mlruns/ /app/mlruns/
COPY mlflow.db /app/

# Expose Streamlit default port
EXPOSE 8501

# Configure Streamlit environment settings for container operation
ENV STREAMLIT_SERVER_PORT=8501
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0

# Set entry point command to run the Streamlit application
CMD ["streamlit", "run", "src/app.py"]