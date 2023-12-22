#!/bin/bash
# export DOCKERHUB_USERNAME=simongarciamorillo
# chmod +x ./scripts/bash/run_app.sh
# ./scripts/bash/run_app.sh
# Web-app url: http://localhost:8501

# Set repository variables
REPOSITORY_NAME=ed-ml-docker
VERSION=v1.0.0

# Set host variables
HOST_PATH=$(pwd)

# Clean containers
docker rm -f model_serving_container_$VERSION
docker rm -f run_app_container_$VERSION

# Create a Docker network (required for app.py to ping the endpoint generated in model_serving.py)
docker network create my_network

# Run model_serving_container from model_serving_image
# -p 5000:5000 \
(
    docker run \
    --name model_serving_container_$VERSION \
    --network my_network \
    -v $HOST_PATH:/app \
    -d $DOCKERHUB_USERNAME/$REPOSITORY_NAME:model_serving_$VERSION
)

# Sleep 5 seconds
sleep 5

# Run the app_container from app_image 
(
    docker run \
    --name run_app_container_$VERSION \
    --network my_network \
    -p 8501:8501 \
    -v $HOST_PATH:/app \
    -d $DOCKERHUB_USERNAME/$REPOSITORY_NAME:run_app_$VERSION
)

# Check active containers
docker ps