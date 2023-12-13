from config.params import Params
from ed_ml.modeling.model_registry import ModelRegistry
from ed_ml.pipeline.pipeline import MLPipeline
from ed_ml.data_processing.feature_engineering import FeatureEngineer
from pprint import pprint
from tqdm import tqdm


def main():
    print('Running model_tuning.\n\n')

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
        n_candidates=Params.n_candidates,
        local_registry=Params.local_registry
    )

    # Define Models to evaluate
    eval_models = (
        model_registry.dev_models
        + model_registry.staging_models
        + [model_registry.prod_model]
    )

    print('Evaluating Models.\n')

    for model in tqdm(eval_models):
        # Evaluate Model on test set
        model.evaluate_test(
            X_test=pipeline.X_test,
            y_test=pipeline.y_test,
            eval_metric=Params.eval_metric
        )

        # Find Model feature importance
        # if model.stage == 'production':
        #     model.find_feature_importance(
        #         test_features=pipeline.X_test
        #     )

        # Save Model
        model.save()

    # Select Champion & Challengers
    model_registry.update_model_stages(update_champion=True)

    # Clean Registry
    model_registry.clean_registry()

    # Show Registry
    print(model_registry)


# .venv/bin/python scripts/model_updating/model_updating.py
if __name__ == '__main__':
    main()