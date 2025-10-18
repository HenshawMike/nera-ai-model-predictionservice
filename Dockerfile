# Use an official Python runtime as a parent image
FROM python:3.10-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_DEFAULT_TIMEOUT=100 \
    POETRY_VERSION=1.4.2

# Set the working directory in the container
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy only requirements first to leverage Docker cache
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY . .

# Create necessary directories
RUN mkdir -p uploads cleaned_data \
    && mkdir -p models/model_v1 models/v1

# Copy model files
COPY model.joblib /app/model.joblib


# Verify model files are present
RUN python setup_models.py

## Set environment variables
ENV PORT=8000
ENV ENVIRONMENT=production

# Expose the port
EXPOSE $PORT

# Start the FastAPI server
# Using 'sh -c' ensures $PORT is expanded properly by the shell
CMD ["sh", "-c", "uvicorn app:app --host 0.0.0.0 --port $PORT"]
