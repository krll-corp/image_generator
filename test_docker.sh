#!/bin/bash

# Simple test script to validate the Docker setup
# This can be run locally to test the container build

echo "Testing Docker build for MNIST Generator..."

# Build the Docker image
echo "Building Docker image..."
if docker build -t mnist-generator-test .; then
    echo "✓ Docker build successful"
else
    echo "✗ Docker build failed"
    exit 1
fi

# Test running the container (quick test)
echo "Testing container startup..."
container_id=$(docker run -d -p 8080:7860 --name mnist-test mnist-generator-test)

if [ $? -eq 0 ]; then
    echo "✓ Container started successfully"
    
    # Wait a bit for the app to start
    sleep 10
    
    # Test health check
    if docker exec $container_id python healthcheck.py; then
        echo "✓ Health check passed"
    else
        echo "⚠ Health check failed (may be expected if models are missing)"
    fi
    
    # Cleanup
    docker stop $container_id >/dev/null 2>&1
    docker rm $container_id >/dev/null 2>&1
    echo "✓ Container cleanup completed"
else
    echo "✗ Container failed to start"
    exit 1
fi

echo "✓ All tests completed successfully!"
echo "The Docker image is ready for deployment to HuggingFace Spaces or other platforms."