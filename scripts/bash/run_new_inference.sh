#!/bin/bash
# export DOCKERHUB_USERNAME=simongarciamorillo
# chmod +x ./scripts/bash/run_new_inference.sh
# ./scripts/bash/run_new_inference.sh course_name user_uuids particion pick_random

# Set up variables passed
COURSE_NAME=${1:-None}
USER_UUIDS=${2:-None}
PARTICION=${3:-None}
PICK_RANDOM=${4:-False}

# Show variables passed
echo "Course chosen: $COURSE_NAME"
echo "Students chosen: $USER_UUIDS"
echo "Particion chosen: $PARTICION"
echo "Pick random: $PICK_RANDOM"

# Set repository variables
REPOSITORY_NAME=ed-ml-docker
VERSION=v1.0.0

# Set host variables
HOST_PATH=$(pwd)

# Clean container
docker rm -f inference_container_$VERSION

# Create a Docker network (required for app.py to ping the endpoint generated in model_serving.py)
docker network create my_network

# Run model_serving_container from model_serving_image
# -p 5000:5000 \
# -d model_serving_image_$VERSION
# (
#     docker run \
#     --name model_serving_container_$VERSION \
#     --network my_network \
#     -v $HOST_PATH:/app \
#     -d $DOCKERHUB_USERNAME/$REPOSITORY_NAME:model_serving_$VERSION
# )

# Sleep 5 seconds
sleep 5

# Run the inference_container from inference_image 
# -d inference_image:$VERSION \
(
    docker run \
    --name inference_container_$VERSION \
    --network my_network \
    -v $HOST_PATH:/app \
    -d $DOCKERHUB_USERNAME/$REPOSITORY_NAME:inference_$VERSION \
    --course_name $COURSE_NAME \
    --user_uuids $USER_UUIDS \
    --particion $PARTICION \
    --pick_random $PICK_RANDOM
)

