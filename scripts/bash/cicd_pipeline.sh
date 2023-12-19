#!/bin/bash
# chmod +x ./scripts/bash/cicd_pipeline.sh
# ./scripts/bash/cicd_pipeline.sh

# Set variables
COMMIT_MESSAGE="Adding documentation changes"
BRANCH_NAME="main"

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