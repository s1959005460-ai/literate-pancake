#!/bin/bash

# FedGNN deployment script
set -e

# Configuration
IMAGE_NAME="fedgnn-advanced"
CONTAINER_NAME="fedgnn-container"
DATASET_DIR="./data"
PORT=8000

# Build Docker image
echo "Building Docker image..."
docker build -t $IMAGE_NAME .

# Create data directory if it doesn't exist
mkdir -p $DATASET_DIR

# Run container
echo "Starting container..."
docker run -d \
  --name $CONTAINER_NAME \
  --gpus all \
  -p $PORT:8000 \
  -v $DATASET_DIR:/data \
  $IMAGE_NAME

echo "Deployment complete. Container $CONTAINER_NAME is running."
echo "Monitoring dashboard available at: http://localhost:$PORT"