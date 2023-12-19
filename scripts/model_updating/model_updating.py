from config.params import Params
from ed_ml.modeling.model_registry import ModelRegistry
from ed_ml.pipeline.pipeline import MLPipeline
from ed_ml.data_processing.feature_engineering import FeatureEngineer
from tqdm import tqdm


def main():
    print('Running model_updating.main\n\n')
    
    # Instanciate FeatureEngineer
    FE = FeatureEngineer(load_dataset=True)

    # Instanciate MLPipeline
    pipeline = MLPipeline()
    
    # Prepare ML datasets
    pipeline.prepare_datasets(
        ml_df=FE.df.copy(),
        train_test_ratio=Params.train_test_ratio
    )

    # Instanciate ModelRegistry
    model_registry = ModelRegistry(
        load_from_local_registry=Params.local_registry
    )

    print(model_registry)
    
    # Update development models
    print('Updating Development Models.\n')
    for model in tqdm(model_registry.dev_models):
        # Update model
        model = pipeline.updating_pipeline(
            model=model,
            eval_metric=Params.eval_metric,
            refit_model=Params.refit_model
        )

        # Save Model
        model.save()
    
    # Update staging models
    print('Updating Staging Models.\n')
    for model in tqdm(model_registry.staging_models):
        # Update model
        model = pipeline.updating_pipeline(
            model=model,
            eval_metric=Params.eval_metric,
            refit_model=Params.refit_model
        )

        # Save Model
        model.save()
    
    # Update champion Model
    print('Updating Champion Model.\n')
    champion = pipeline.updating_pipeline(
        model=model_registry.prod_model,
        eval_metric=Params.eval_metric,
        refit_model=Params.refit_model
    )

    # Save champion Model
    champion.save()

    # Select Champion & Challengers
    model_registry.update_model_stages(
        n_candidates=Params.n_candidates,
        update_champion=True
    )

    # Clean Registry
    model_registry.clean_registry()

    # Show Registry
    print(model_registry)


# .venv/bin/python scripts/model_updating/model_updating.py
if __name__ == '__main__':
    main()
    
    # import shap
    # import numpy as np
    # import pickle
    # import os
    # import pandas as pd

    # FE = FeatureEngineer(load_dataset=True)

    # # Instanciate MLPipeline
    # pipeline = MLPipeline()
    
    # # Prepare ML datasets
    # pipeline.prepare_datasets(
    #     ml_df=FE.df.copy(),
    #     train_test_ratio=Params.train_test_ratio
    # )

    # # Instanciate ModelRegistry
    # model_registry = ModelRegistry(
    #     load_from_local_registry=Params.local_registry
    # )

    # # Set up tracking server
    # model_registry.set_up_tracking_server()

    # model = model_registry.dev_models[0]
    # print(model.shap_values)

    # model.find_feature_importance(
    #     X_test=pipeline.X_test,
    #     find_new_shap_values=Params.find_new_shap_values
    # )
    # print(model.shap_values)
    
    # model.save()

    # explainer = shap.TreeExplainer(model.model)

    # Calculate shap values
    # model.shap_values: np.ndarray = explainer.shap_values(pipeline.X_test)

    # with open("shap_values_model_attr.pickle", 'wb') as handle:
    #     pickle.dump({'shap_values': model.shap_values}, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # Load model
    # with open("shap_values_model_attr.pickle", 'rb') as handle:
    #     model.shap_values: np.ndarray = pickle.load(handle)['shap_values']

    # model.save()
