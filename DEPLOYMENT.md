# Deployment Guide for MNIST Image Generator

This guide provides multiple deployment options for the MNIST Image Generator, addressing the issue of large model weights (~700MB+) that prevent deployment on platforms like Vercel.

## Deployment Options

### 1. Railway (Recommended for ML Apps)

Railway is excellent for ML applications with generous resource limits.

**Steps:**
1. Fork/clone this repository
2. Connect to Railway: https://railway.app
3. Create new project from GitHub repo
4. Railway will automatically detect the Python app
5. Set environment variables:
   ```
   FLASK_ENV=production
   PORT=5000
   DEVICE=cpu
   ```
6. Deploy! Railway handles the rest.

**Pros:** 
- Up to 32GB RAM, 32 vCPUs
- Generous storage for model files
- Great for ML workloads
- Simple deployment

**Cons:**
- Paid service after free tier

### 2. Render

Good Docker support and ML-friendly infrastructure.

**Steps:**
1. Push code to GitHub
2. Connect to Render: https://render.com
3. Create new Web Service
4. Use Docker environment
5. Set build command: `docker build -t image-generator .`
6. Set start command: `python app.py`

**Pros:**
- Free tier available
- Docker support
- Good performance

**Cons:**
- Free tier limitations

### 3. Google Cloud Run

Serverless container platform that scales to zero.

**Steps:**
1. Install Google Cloud SDK
2. Build and push container:
   ```bash
   gcloud builds submit --tag gcr.io/PROJECT-ID/image-generator
   gcloud run deploy --image gcr.io/PROJECT-ID/image-generator --platform managed
   ```

**Pros:**
- Pay per request
- Scales to zero
- Handles large containers

**Cons:**
- Cold start latency
- Requires GCP knowledge

### 4. DigitalOcean App Platform

Simple container deployment with good ML support.

**Steps:**
1. Connect GitHub repository
2. Choose Docker deployment
3. Set environment variables
4. Deploy

### 5. Hugging Face Spaces

Perfect for ML demos and prototypes.

**Steps:**
1. Create account at https://huggingface.co/spaces
2. Create new Space with Docker
3. Upload code and Dockerfile
4. Models can be stored on Hugging Face Hub

**Pros:**
- ML-focused platform
- Great for demos
- Model hosting integration

**Cons:**
- Limited resources on free tier

### 6. AWS ECS/Fargate

Enterprise-grade container orchestration.

**Steps:**
1. Create ECR repository
2. Build and push Docker image
3. Create ECS cluster and service
4. Configure load balancer

**Pros:**
- Highly scalable
- Enterprise features
- AWS ecosystem

**Cons:**
- Complex setup
- Higher costs

## Local Development with Docker

```bash
# Build the image
docker build -t image-generator .

# Run with model volume
docker run -p 5000:5000 -v $(pwd)/models:/app/models image-generator
```

## Model Storage Strategies

### Option 1: Direct Deployment
- Include models in Docker image
- Simple but large image size
- Good for self-contained deployments

### Option 2: Model Download on Startup
- Download models from cloud storage on container start
- Smaller image, longer startup time
- Good for platforms with storage limits

### Option 3: External Model Storage
- Store models in cloud storage (AWS S3, GCS, etc.)
- Mount as volume or download as needed
- Most flexible approach

## Environment Variables

```bash
FLASK_ENV=production          # production/development
PORT=5000                    # Server port
HOST=0.0.0.0                # Server host
DEVICE=cpu                   # cpu/cuda/mps/auto
MODEL_PATH=/app/models       # Path to model files
MAX_MEMORY_MB=2048          # Memory limit
```

## Performance Optimization

1. **Use CPU-optimized builds** for platforms without GPU
2. **Implement model lazy loading** to reduce startup time
3. **Add health check endpoints** for container orchestration
4. **Use gunicorn** for production WSGI server
5. **Implement caching** for frequently used models

## Security Considerations

1. Don't commit model files to git (use Git LFS if needed)
2. Use environment variables for sensitive configuration
3. Implement rate limiting for public deployments
4. Use HTTPS in production
5. Validate all user inputs

## Monitoring and Logging

1. Add structured logging
2. Implement health checks
3. Monitor memory usage (models are memory-intensive)
4. Set up alerts for failures

## Cost Optimization

1. **Railway**: Best for development/small scale
2. **Render**: Good free tier for testing
3. **Google Cloud Run**: Pay per request, good for sporadic usage
4. **Traditional VPS**: Best for consistent usage

Choose based on your usage patterns and budget requirements.