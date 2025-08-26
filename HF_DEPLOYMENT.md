# HuggingFace Spaces Deployment Guide

This repository includes everything needed to deploy the MNIST Digit Generator to HuggingFace Spaces using Docker.

## Quick Deploy to HuggingFace Spaces

1. **Create a new Space on HuggingFace:**
   - Go to https://huggingface.co/spaces
   - Click "Create new Space"
   - Choose "Docker" as the SDK
   - Name your space and set it as public or private

2. **Clone your HF Space repository:**
   ```bash
   git clone https://huggingface.co/spaces/YOUR_USERNAME/YOUR_SPACE_NAME
   cd YOUR_SPACE_NAME
   ```

3. **Copy the necessary files from this repository:**
   ```bash
   # Copy these files to your HF Space repository:
   - Dockerfile
   - requirements.txt
   - app.py
   - templates/
   - *.py (all Python model files)
   - healthcheck.py
   ```

4. **Create a README.md for your Space:**
   ```yaml
   ---
   title: MNIST Digit Generator
   emoji: 🔢
   colorFrom: blue
   colorTo: purple
   sdk: docker
   pinned: false
   license: mit
   app_port: 7860
   ---
   
   # MNIST Digit Generator
   
   Interactive web application that generates MNIST-style digits using various deep learning models.
   ```

5. **Push to your HF Space:**
   ```bash
   git add .
   git commit -m "Add MNIST Digit Generator"
   git push
   ```

6. **Your Space will automatically build and deploy!**

## Alternative: Direct Container Registry

You can also use the pre-built images from GitHub Container Registry:

```bash
docker pull ghcr.io/krll-corp/image_generator:latest
docker run -p 7860:7860 ghcr.io/krll-corp/image_generator:latest
```

## Configuration

The application automatically:
- Binds to `0.0.0.0:7860` (HF Spaces requirement)
- Loads available models on startup
- Gracefully handles missing models
- Provides a web interface for digit generation

## Model Notes

The container includes the neural network architectures but not the trained weights. Models will be detected as "unavailable" until trained weights are provided. This is by design to keep the container size reasonable.

To include pre-trained models, mount them as volumes or add them to the container during build.