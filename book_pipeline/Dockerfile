# Use official Python image as base
FROM python:3.13.3-alpine

# Set working directory
WORKDIR /app

# Install system dependencies using Alpine's package manager
RUN apk add --no-cache \
    poppler-utils \
    tesseract-ocr \
    libgl1-mesa-glx \
    g++ \
    make \
    openblas-dev

# Copy requirements file
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p data output cache logs

# Set environment variables
ENV PYTHONUNBUFFERED=1

# Expose ports for API and metrics
EXPOSE 8000 8001

# Run the FastAPI server
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]