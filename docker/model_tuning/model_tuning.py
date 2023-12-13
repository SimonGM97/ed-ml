from config.params import Params
from ed_ml.data_processing.feature_engineering import FeatureEngineer
from ed_ml.modeling.model_tuning import ModelTuner
from ed_ml.utils.add_parser_arguments import add_parser_arguments
import argparse
import warnings

warnings.filterwarnings("ignore")


def main(
    algorithms: str,
    eval_metric: str,
    val_splits: int,
    train_test_ratio: float,
    n_candidates: int,
    local_registry: bool,
    max_evals: int,
    timeout_mins: int,
    loss_threshold: float,
    min_performance: float
):
    # Instanciate FeatureEngineer
    FE = FeatureEngineer(load_dataset=True)

    # Instanciate ModelTuner
    tuner = ModelTuner(
        algorithms=algorithms,
        eval_metric=eval_metric,
        val_splits=val_splits,
        train_test_ratio=train_test_ratio,
        n_candidates=n_candidates,
        local_registry=local_registry,
        max_evals=max_evals,
        timeout_mins=timeout_mins,
        loss_threshold=loss_threshold,
        min_performance=min_performance
    )
    
    # Run ModelTuner
    tuner.run(
        ml_df=FE.df.copy(),
        soft_debug=True,
        deep_debug=False
    )

    # Select Champion & Challengers
    tuner.model_registry.update_model_stages(update_champion=True)

    # Clean Registry
    tuner.model_registry.clean_registry()

    # Show Registry
    print(tuner.model_registry)


# .venv/bin/python scripts/model_tuning/model_tuning.py --arg_name arg_value
if __name__ == '__main__':
    # Define parser
    parser = argparse.ArgumentParser(description='Model tuning script.')

    # Add expected arguments
    parser = add_parser_arguments(parser=parser)

    # Extract arguments
    args = parser.parse_args()

    # Run main with parsed parguments
    main(
        algorithms=args.algorithms,
        eval_metric=args.eval_metric,
        val_splits=args.val_splits,
        train_test_ratio=args.train_test_ratio,
        n_candidates=args.n_candidates,
        local_registry=args.local_registry,
        max_evals=args.max_evals,
        timeout_mins=args.timeout_mins,
        loss_threshold=args.loss_threshold,
        min_performance=args.min_performance
    )

    