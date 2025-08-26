# Quick Deployment Commands

## Railway
```bash
# 1. Install Railway CLI
npm install -g @railway/cli

# 2. Login and deploy
railway login
railway link
railway up
```

## Render
```bash
# 1. Connect GitHub repo to Render
# 2. Choose Docker deployment
# 3. Set environment variables in Render dashboard
```

## Google Cloud Run
```bash
# 1. Install gcloud CLI
# 2. Build and deploy
gcloud builds submit --tag gcr.io/PROJECT-ID/image-generator
gcloud run deploy image-generator \
  --image gcr.io/PROJECT-ID/image-generator \
  --platform managed \
  --port 5000 \
  --memory 4Gi \
  --cpu 2 \
  --max-instances 10
```

## Docker Local Testing
```bash
# Build and test locally
docker build -t image-generator .
docker run -p 5000:5000 -e FLASK_ENV=development image-generator
```

## Hugging Face Spaces
```bash
# Run the deployment script
./deploy-huggingface.sh
# Then upload to HF Spaces with Docker runtime
```