#!/bin/bash
# export DOCKERHUB_USERNAME=simongarciamorillo
# chmod +x ./scripts/bash/model_building_workflow.sh
# ./scripts/bash/model_building_workflow.sh max_evals

# Set up variables passed
MAX_EVALS=${1:-None}

# Show variables passed
echo "Max evals chosen: $MAX_EVALS"

# Set repository variables
REPOSITORY_NAME=ed-ml-docker
VERSION=v1.0.0

# Set host variables
HOST_PATH=$(pwd)

# Clean containers
docker rm -f data_processing_container_$VERSION
docker rm -f model_tuning_container_$VERSION
docker rm -f model_updating_container_$VERSION

# Create a Docker network (required to host the tracking server)
docker network create my_network

# Run workflow
(
    REPOSITORY_NAME=$REPOSITORY_NAME \
    USERNAME=$DOCKERHUB_USERNAME \
    VERSION=$VERSION \
    MAX_EVALS=$MAX_EVALS \
    HOST_PATH=$HOST_PATH \
    docker-compose -f docker/docker-compose.yaml up
)

