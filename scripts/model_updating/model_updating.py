from ed_ml.modeling.model_registry import ModelRegistry
from ed_ml.modeling.model import Model
from ed_ml.pipeline.pipeline import MLPipeline
from ed_ml.utils.add_parser_arguments import add_parser_arguments
from tqdm import tqdm
import argparse


def main(
    eval_metric: str,
    refit_model: bool,
    optimize_cutoff: bool,
    find_new_shap_values: bool,
    local_registry: bool,
    n_candidates: bool
):
    print(f'Running model_updating.main with parameters:\n'
          f'    - eval_metric: {eval_metric} ({type(eval_metric)})\n'
          f'    - refit_model: {refit_model} ({type(refit_model)})\n'
          f'    - optimize_cutoff: {optimize_cutoff} ({type(optimize_cutoff)})\n'
          f'    - find_new_shap_values: {find_new_shap_values} ({type(find_new_shap_values)})\n'
          # f'    - local_registry: {local_registry} ({type(local_registry)})\n'
          f'    - n_candidates: {n_candidates} ({type(n_candidates)})\n\n')

    def update_model(model_: Model):
        # Update model
        model_ = pipeline.updating_pipeline(
            model=model_,
            eval_metric=eval_metric,
            refit_model=refit_model,
            optimize_cutoff=optimize_cutoff,
            find_new_shap_values=find_new_shap_values
        )

        # Save Model
        model_.save()
    
    # Instanciate MLPipeline
    pipeline = MLPipeline(load_datasets=True)

    # Instanciate ModelRegistry
    model_registry = ModelRegistry(
        load_from_local_registry=local_registry
    )
    
    # Update development models
    print('Updating Development Models.\n')
    for model in tqdm(model_registry.dev_models):
        update_model(model)
    
    # Update staging models
    print('Updating Staging Models.\n')
    for model in tqdm(model_registry.staging_models):
        update_model(model)
    
    # Update champion Model
    print('Updating Champion Model.\n')
    update_model(model_registry.prod_model)

    # Select Champion & Challengers
    model_registry.update_model_stages(
        n_candidates=n_candidates,
        update_champion=True
    )

    # Clean Registry
    model_registry.clean_registry()

    # Show Registry
    print(model_registry)


# .venv/bin/python scripts/model_updating/model_updating.py
if __name__ == '__main__':
    # Define parser
    parser = argparse.ArgumentParser(description='Model updating script.')

    # Add expected arguments
    parser = add_parser_arguments(parser=parser)

    # Extract arguments
    args = parser.parse_args()

    # Run main with parsed parguments
    main(
        eval_metric=args.eval_metric,
        refit_model=args.refit_model,
        optimize_cutoff=args.optimize_cutoff,
        find_new_shap_values=args.find_new_shap_values,
        local_registry=args.local_registry,
        n_candidates=args.n_candidates
    )
