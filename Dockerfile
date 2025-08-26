# Use Python 3.12 slim image as base
FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies (use CPU-only PyTorch for smaller image)
RUN pip install --no-cache-dir \
    torch==2.4.1+cpu \
    torchvision==0.19.1+cpu \
    -f https://download.pytorch.org/whl/torch_stable.html && \
    pip install --no-cache-dir \
    flask>=2.3.0 \
    pillow>=9.0.0 \
    numpy>=1.24.0 \
    diffusers>=0.21.0 \
    transformers>=4.30.0 \
    tqdm>=4.65.0 \
    matplotlib>=3.7.0 \
    lion-pytorch>=0.1.0

# Copy application code
COPY . .

# Create data directory for MNIST downloads
RUN mkdir -p data

# Expose the Flask port
EXPOSE 5000

# Set environment variables
ENV FLASK_APP=app.py
ENV FLASK_ENV=production

# Run the application
CMD ["python", "app.py"]