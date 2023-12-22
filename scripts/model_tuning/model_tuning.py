from config.params import Params
from ed_ml.data_processing.feature_engineering import FeatureEngineer
from ed_ml.modeling.model_tuner import ModelTuner
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
    min_performance: float,
    balance_train: bool,
    balance_method: str,
):
    print(f'Running model_tuning.main with parameters:\n'
          f'    - algorithms: {algorithms} ({type(algorithms)})\n'
          f'    - eval_metric: {eval_metric} ({type(eval_metric)})\n'
          f'    - val_splits: {val_splits} ({type(val_splits)})\n'
          f'    - train_test_ratio: {train_test_ratio} ({type(train_test_ratio)})\n'
          f'    - n_candidates: {n_candidates} ({type(n_candidates)})\n'
          # f'    - local_registry: {local_registry} ({type(local_registry)})\n'
          f'    - max_evals: {max_evals} ({type(max_evals)})\n'
          f'    - timeout_mins: {timeout_mins} ({type(timeout_mins)})\n'
          f'    - loss_threshold: {loss_threshold} ({type(loss_threshold)})\n'
          f'    - min_performance: {min_performance} ({type(min_performance)})\n'
          f'    - balance_train: {balance_train} ({type(balance_train)})\n'
          f'    - balance_method: {balance_method} ({type(balance_method)})\n\n')
    
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
        balance_train=balance_train,
        balance_method=balance_method,
        soft_debug=True,
        deep_debug=False
    )


# .venv/bin/python scripts/model_tuning/model_tuning.py --max_evals 20
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
        min_performance=args.min_performance,
        balance_train=args.balance_train,
        balance_method=args.balance_method
    )
