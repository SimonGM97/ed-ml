from config.params import Params
from ed_ml.modeling.model_registry import ModelRegistry
from ed_ml.pipeline.pipeline import MLPipeline

from flask import Flask, jsonify, request
import pandas as pd
import numpy as np
import sys

app = Flask(__name__)

# Your API endpoint URL would consist /predict
@app.route('/predict', methods=['POST'])
def predict():
    """
    Function that serves two key purposes:
        - Model Loading: It will load our persisted machine learning model into 
          memory as soon as our application kicks off.
        - API Endpoint Creation: This function will also set up an API endpoint. 
          This endpoint will be responsible for accepting input variables, intelligently 
          converting them into the required format for model processing, and then 
          delivering the valuable predictions we seek.
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

    # Run app
    port = Params.request_url.split('/')[-2].split(':')[-1]
    app.run(host='0.0.0.0', port=port, debug=True)