#!/bin/bash

# Railway Deployment Script
# This script prepares the app for Railway deployment

echo "Preparing for Railway deployment..."

# Railway specific configurations
export PORT=${PORT:-5000}
export RAILWAY_STATIC_URL=${RAILWAY_STATIC_URL:-""}

# Install dependencies
echo "Installing dependencies..."
pip install -r requirements.txt

# Create model directories
mkdir -p my_conv my_model my_moe_model vq_transformer_model my_diffusion_model

echo "Starting Flask application..."
python app.py