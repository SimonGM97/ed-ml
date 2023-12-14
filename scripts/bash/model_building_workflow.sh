#!/bin/bash
# chmod +x ./scripts/bash/model_building_workflow.sh
# ./scripts/bash/model_building_workflow.sh

# Set repository variables
REPOSITORY_NAME=ed-ml-docker
USERNAME=simongarciamorillo
VERSION=v.1.0.0

# Set script variables
MAX_EVALS=20

# Set host variables
HOST_PATH=$(pwd)

# Clean containers
docker rm -f $(docker ps -aq)

# Clean local images
docker rmi -f $(docker images -q)

# Create a Docker network (required to host the tracking server)
docker network create my_network

# login to docker
echo $DOCKERHUB_TOKEN | docker login -u $USERNAME --password-stdin

# Pull images from dockerhub
docker pull $USERNAME/$REPOSITORY_NAME:data_processing_$VERSION
docker pull $USERNAME/$REPOSITORY_NAME:model_tuning_$VERSION
docker pull $USERNAME/$REPOSITORY_NAME:model_updating_$VERSION

# Run model_tuning_container from model_tuning_image
# (
#     docker run \
#     --name data_processing_container_$VERSION \
#     -v $HOST_PATH:/app \
#     -d $USERNAME/$REPOSITORY_NAME:data_processing_$VERSION
# )
# (
#     docker run \
#     --name model_tuning_container_$VERSION \
#     --network my_network -p 5050:5050 \
#     -v $HOST_PATH:/app \
#     -d $USERNAME/$REPOSITORY_NAME:model_tuning_$VERSION \
#     --max_evals $MAX_EVALS
# )
# (
#     docker run 
#     --name model_updating_container_$VERSION 
#     -v $HOST_PATH:/app 
#     -d $USERNAME/$REPOSITORY_NAME:model_updating_$VERSION
# )

# Run workflow
(
    REPOSITORY_NAME=$REPOSITORY_NAME \
    USERNAME=$USERNAME \
    VERSION=$VERSION \
    MAX_EVALS=$MAX_EVALS \
    HOST_PATH=$HOST_PATH \
    docker-compose -f docker/docker-compose.yaml up
)

