from config.params import Params
from ed_ml.modeling.model_registry import ModelRegistry
import plotly.graph_objects as go
import streamlit as st
import pandas as pd
import numpy as np


def build_model_registry_page():
    # Instanciate Registry
    registry = ModelRegistry(
        load_from_local_registry=Params.local_registry
    )

    # Load Models    
    models = (
        [registry.prod_model] 
        + registry.staging_models
        + registry.dev_models
    )

    # Write new inference text
    st.markdown(
        '<p style="font-family:sans-serif; color:#183f59; font-size: 25px; font-weight: bold; text-align: left;"'
        '>Model Registry',
        unsafe_allow_html=True
    )

    # Write a line
    st.write("-----")

     # Define row
    row0, row1, row2 = st.columns([1, 8, 1])

    # Find Summary Models DF
    models_df = pd.DataFrame({
        'Model ID': [model.model_id for model in models],
        'Stage': [model.stage for model in models],
        'Algorithm': [model.algorithm for model in models],
        'F1 Score (test)': [model.f1_score for model in models],
        'Precision (test)': [model.precision_score for model in models],
        'Recall (test)': [model.recall_score for model in models],
        'ROC AUC (test)': [model.roc_auc_score for model in models],
        'Accuracy (test)': [model.accuracy_score for model in models] 
    })
    print(models_df)

    def highlight_row(row):
        color = 'background-color: white'
        if row['Stage'] == 'production':
            color = 'background-color: #93A3BC'
        return [color] * len(row)
    
    config_cols = [
        'F1 Score (test)', 'Precision (test)', 'Recall (test)', 
        'ROC AUC (test)', 'Accuracy (test)'
    ]

    models_df[config_cols] = 100 * models_df[config_cols].round(3)

    row1.dataframe(
        models_df.style.apply(highlight_row, axis=1), 
        column_config={
            col: st.column_config.ProgressColumn(
                col,
                format="%.1f%%",
                min_value=50,
                max_value=100
            ) for col in config_cols
        },
        use_container_width=True,
        hide_index=True
    )