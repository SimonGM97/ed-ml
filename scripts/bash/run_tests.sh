#!/bin/bash
# chmod +x ./scripts/_bash/run_tests.sh
# ./scripts/_bash/run_tests.sh

# Run Data Processing Tests
python3 -m unittest test/test_data_processing.py

# Run MLPipeline Tests
python3 -m unittest test/test_ml_pipeline.py