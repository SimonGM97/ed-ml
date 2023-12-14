#!/bin/bash
# chmod +x ./scripts/bash/image_building.sh
# ./scripts/bash/image_building.sh

# Set repository variables
REPOSITORY_NAME=ed-ml-docker
USERNAME=simongarciamorillo
VERSION=v.1.0.0

# Make scripts executable
# chmod +x ./scripts/data_processing/data_processing.py
# chmod +x ./scripts/model_tuning/model_tuning.py
# chmod +x ./scripts/model_updating/model_updating.py

chmod +x ./scripts/model_serving/model_serving.py
chmod +x app.py

# Clean containers
docker rm -f $(docker ps -aq)

# Clean local images
docker rmi -f $(docker images -q)

# Build Docker images
# docker build -t data_processing_image:$VERSION -f docker/data_processing/Dockerfile .
# docker build -t model_tuning_image:$VERSION -f docker/model_tuning/Dockerfile .
# docker build -t model_updating_image:$VERSION -f docker/model_updating/Dockerfile .

docker build -t model_serving_image:$VERSION -f docker/model_serving/Dockerfile .
docker build -t app_image:$VERSION -f docker/app/Dockerfile .

# login to docker
echo $DOCKERHUB_TOKEN | docker login -u $USERNAME --password-stdin

# Tag docker images
# docker tag data_processing_image:$VERSION $USERNAME/$REPOSITORY_NAME:data_processing_$VERSION
# docker tag model_tuning_image:$VERSION $USERNAME/$REPOSITORY_NAME:model_tuning_$VERSION
# docker tag model_updating_image:$VERSION $USERNAME/$REPOSITORY_NAME:model_updating_$VERSION

docker tag model_serving_image:$VERSION $USERNAME/$REPOSITORY_NAME:model_serving_$VERSION
docker tag app_image:$VERSION $USERNAME/$REPOSITORY_NAME:app_$VERSION

# Push images to repository
# docker push $USERNAME/$REPOSITORY_NAME:data_processing_$VERSION
# docker push $USERNAME/$REPOSITORY_NAME:model_tuning_$VERSION
# docker push $USERNAME/$REPOSITORY_NAME:model_updating_$VERSION

docker push $USERNAME/$REPOSITORY_NAME:model_serving_$VERSION
docker push $USERNAME/$REPOSITORY_NAME:app_$VERSION

# List dockerhub images
docker images $USERNAME/$REPOSITORY_NAME
