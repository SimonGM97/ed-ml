#!/bin/bash
# export DOCKERHUB_USERNAME=simongarciamorillo
# export DOCKERHUB_TOKEN=token_key
# chmod +x ./scripts/bash/image_pulling.sh
# ./scripts/bash/image_pulling.sh

# Set repository variables
REPOSITORY_NAME=ed-ml-docker
VERSION=v1.0.0

# Clean containers
docker rm -f $(docker ps -aq)

# Clean local images
docker rmi -f $(docker images -q)

# login to docker
echo $DOCKERHUB_TOKEN | docker login -u $DOCKERHUB_USERNAME --password-stdin

# Pull images from dockerhub
docker pull $DOCKERHUB_USERNAME/$REPOSITORY_NAME:data_processing_$VERSION
docker pull $DOCKERHUB_USERNAME/$REPOSITORY_NAME:model_tuning_$VERSION
docker pull $DOCKERHUB_USERNAME/$REPOSITORY_NAME:model_updating_$VERSION

docker pull $DOCKERHUB_USERNAME/$REPOSITORY_NAME:model_serving_$VERSION
docker pull $DOCKERHUB_USERNAME/$REPOSITORY_NAME:inference_$VERSION
docker pull $DOCKERHUB_USERNAME/$REPOSITORY_NAME:run_app_$VERSION
