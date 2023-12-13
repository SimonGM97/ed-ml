#!/bin/bash
# export DOCKERHUB_TOKEN=dckr_pat_7yTPWD938TSdkt3CA04IpW5wLXw
# chmod +x ./scripts/_bash/run_app.sh
# ./scripts/_bash/run_app.sh

# Set repository variables
REPOSITORY_NAME=ed-ml-docker
USERNAME=simongarciamorillo
VERSION=v.1.0.0

# Set host variables
HOST_PATH=$(pwd)

# Clean containers
docker rm -f $(docker ps -aq)

# Clean local images
docker rmi -f $(docker images -q)

# Create a Docker network (required for app.py to ping the endpoint generated in model_serving.py)
docker network create my_network

# login to docker
echo $DOCKERHUB_TOKEN | docker login -u $USERNAME --password-stdin

# Pull images from dockerhub
docker pull $USERNAME/$REPOSITORY_NAME:model_serving_$VERSION
docker pull $USERNAME/$REPOSITORY_NAME:app_$VERSION

# Run model_serving_container from model_serving_image
(
    docker run \
    --name model_serving_container_$VERSION \
    --network my_network \
    -v $HOST_PATH:/app \
    -d $USERNAME/$REPOSITORY_NAME:model_serving_$VERSION
)

# Sleep 5 seconds
sleep 5

# Run the app_container from app_image 
(
    docker run \
    --name app_container_$VERSION \
    --network my_network \
    -p 8501:8501 \
    -v $HOST_PATH:/app \
    -d $USERNAME/$REPOSITORY_NAME:app_$VERSION
)

# Check active containers
docker ps

# Open web app: http://localhost:8501