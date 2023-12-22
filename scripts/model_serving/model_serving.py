from config.params import Params
from ed_ml.modeling.model_registry import ModelRegistry
from ed_ml.pipeline.pipeline import MLPipeline

from flask import Flask, jsonify, request
import pandas as pd
import numpy as np
import json

app = Flask(__name__)

# Your API endpoint URL would consist /predict
@app.route('/predict', methods=['POST'])
def predict() -> json:
    """
    Function that will:
        - Load the champion ML model into memory when the application kicks off.
        - Set up the API endpoint required to ping the champion model and obtain new inferences.
    
    :return: (json) New inferences derived from raw_data parsed as a json file.
    """
    try:
        json_ = request.json
        raw_df = pd.DataFrame(json_).replace('nan', np.nan)

        # Use the service name of the inference container as the URL
        predictions = ml_pipeline.inference_pipeline(
            model=champion,
            raw_df=raw_df
        )
        return jsonify(predictions)
    except Exception as e:
        print(f"Error: {e}")
        return jsonify({"error": str(e)}), 500


# .venv/bin/python scripts/model_serving/model_serving.py
if __name__ == '__main__':
    # Load champion Model
    ml_registry = ModelRegistry(
        load_from_local_registry=Params.local_registry
    )
    
    champion = ml_registry.prod_model

    # Define Pipeline
    ml_pipeline = MLPipeline()

    # Extract host & port
    host = Params.request_url.split('/')[-2].split(':')[0]
    port = Params.request_url.split('/')[-2].split(':')[-1]

    # Run app
    app.run(host=host, port=port)