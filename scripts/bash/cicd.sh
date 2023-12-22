#!/bin/bash
# chmod +x ./scripts/bash/cicd.sh
# ./scripts/bash/cicd.sh

# Set variables
COMMIT_MESSAGE="Adding documentation changes"
BRANCH_NAME="main"

# Build & pull docker images
chmod +x ./scripts/bash/image_building.sh
./scripts/bash/image_building.sh

# Pull Docker images
chmod +x ./scripts/bash/image_pulling.sh
./scripts/bash/image_pulling.sh

# Stage changes
git add .

# Commit changes
git commit -m $COMMIT_MESSAGE

# Pull changes
git pull origin $BRANCH_NAME

# Run Data Processing Tests
python3 -m unittest test/test_data_processing.py

# Run MLPipeline Tests
python3 -m unittest test/test_ml_pipeline.py

# Push changes
git push origin $BRANCH_NAME