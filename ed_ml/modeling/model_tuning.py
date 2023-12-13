from config.params import Params
from ed_ml.modeling.model import Model
from ed_ml.modeling.model_registry import ModelRegistry
from ed_ml.pipeline.pipeline import MLPipeline

import mlflow
from hyperopt import fmin, hp, tpe, STATUS_OK
from hyperopt.fmin import generate_trials_to_calculate
from hyperopt.pyll.base import scope

import pandas as pd
import numpy as np
import time
from pprint import pprint
from typing import List
import warnings

warnings.filterwarnings("ignore")


class ModelTuner:

    int_parameters = [
        # random_forest
        'random_forest.n_estimators',
        'random_forest.max_depth',
        'random_forest.min_samples_split',

        # lightgbm
        'lightgbm.n_estimators',
        'lightgbm.max_depth',
        'lightgbm.min_child_samples',
        'lightgbm.num_leaves'
    ]

    choice_parameters = {
        # random_forest
        'random_forest.max_features': ['1.0', 'sqrt'],

        # lightgbm
        'lightgbm.boosting_type': ['gbdt', 'dart'],

        # xgboost
        'xgboost.booster': ['gbtree', 'dart']
    }

    # Model Type Choices
    model_type_choices = [
        # Random Forest Search Space
        {
            "algorithm": 'random_forest',
            "random_forest.n_estimators": scope.int(hp.quniform('random_forest.n_estimators', 15, 125, 1)), # 5, 350, 1
            "random_forest.max_depth": scope.int(hp.quniform('random_forest.max_depth', 5, 90, 1)), # 1, 175, 1
            "random_forest.min_samples_split": scope.int(hp.quniform('random_forest.min_samples_split', 5, 150, 1)),
            "random_forest.max_features": hp.choice('random_forest.max_features', choice_parameters['random_forest.max_features']),
        },
        
        # Lightgbm Search Space
        {
            "algorithm": 'lightgbm',
            "lightgbm.boosting_type": hp.choice('lightgbm.boosting_type', choice_parameters['lightgbm.boosting_type']),
            "lightgbm.n_estimators": scope.int(hp.quniform('lightgbm.n_estimators', 15, 125, 1)), # 5, 350, 1
            "lightgbm.max_depth": scope.int(hp.quniform('lightgbm.max_depth', 5, 90, 1)), # 1, 175, 1
            "lightgbm.min_child_samples": scope.int(hp.quniform('lightgbm.min_child_samples', 5, 150, 1)),
            "lightgbm.learning_rate": hp.loguniform('lightgbm.learning_rate', np.log(0.001), np.log(0.3)),
            "lightgbm.num_leaves": scope.int(hp.quniform('lightgbm.num_leaves', 5, 90, 1)), # 5, 150, 1
            "lightgbm.colsample_bytree": hp.uniform('lightgbm.colsample_bytree', 0.6, 1)
        },

        # XGBoost Search Space
        {
            "algorithm": 'xgboost',
            "xgboost.booster": hp.choice('xgboost.booster', choice_parameters['xgboost.booster']),
            "xgboost.eta": hp.loguniform('xgboost.eta', np.log(0.005), np.log(0.4)), # learning_rate
            "xgboost.n_estimators": scope.int(hp.quniform('xgboost.n_estimators', 15, 100, 1)), # 5, 250, 1
            "xgboost.max_depth": scope.int(hp.quniform('xgboost.max_depth', 5, 80, 1)), # 1, 120, 1
            "xgboost.colsample_bytree": hp.uniform('xgboost.colsample_bytree', 0.6, 1),
            "xgboost.lambda": hp.loguniform('xgboost.lambda', np.log(0.001), np.log(5)),
            "xgboost.alpha": hp.loguniform('xgboost.alpha', np.log(0.001), np.log(5)),
            "xgboost.max_leaves": scope.int(hp.quniform('xgboost.max_leaves', 5, 80, 1)), # 5, 120, 1
        }
    ]
    
    def __init__(
        self,
        algorithms: list = None,
        eval_metric: str = None,
        val_splits: int = None,
        train_test_ratio: float = None,
        n_candidates: int = None,
        local_registry: bool = True,
        max_evals: int = None,
        timeout_mins: int = None,
        loss_threshold: float = None,
        min_performance: float = None
    ) -> None:
        # Define attributes
        self.algorithms = algorithms
        self.eval_metric = eval_metric
        self.val_splits = val_splits
        self.train_test_ratio = train_test_ratio

        self.n_candidates = n_candidates
        self.max_evals = max_evals
        self.timeout_mins = timeout_mins
        self.loss_threshold = loss_threshold
        self.min_performance = min_performance

        # Set up hyper-opt search space
        self.model_type_choices = [
            choice for choice in self.model_type_choices if choice['algorithm'] in self.algorithms
        ]
        
        self.search_space = {
            "model_type": hp.choice('model_type', self.model_type_choices)
        }

        # Instanciate MLPipeline
        self.ml_pipeline: MLPipeline = MLPipeline()
        
        # Instanciate ModelRegistry
        self.model_registry: ModelRegistry = ModelRegistry(
            n_candidates=self.n_candidates,
            local_registry=local_registry
        )

        # Define dev_models
        self.dev_models: List[Model] = (
            self.model_registry.dev_models
            + self.model_registry.staging_models
        )

        if self.model_registry.prod_model is not None:
            self.dev_models.extend([self.model_registry.prod_model])

        # Sort dev_models, based on validation performance
        self.dev_models.sort(key=lambda model: model.val_score, reverse=True)

    def parameter_configuration(
        self, 
        parameters_list: list[dict],
        complete_parameters: bool = False,
        choice_parameters: str = 'index', 
        debug: bool = False
    ):
        if choice_parameters not in ['index', 'values']:
            raise Exception(f'Invalid "choice_parameters": {choice_parameters}.\n\n')

        int_types = [int, np.int64, np.int32] #, float, np.float32 ,np.float64]

        for parameters in parameters_list:
            # Check "algorithm" parameter
            if 'algorithm' not in parameters.keys() and type(parameters['model_type']) in int_types:
                parameters['algorithm'] = self.algorithms[parameters['model_type']]
            elif 'algorithm' in parameters.keys() and type(parameters['algorithm']) == str and type(parameters['model_type']) in int_types:
                parameters['model_type'] = self.algorithms.index(parameters['algorithm'])

            # Check "model_type" parameter
            if parameters['model_type'] is None:
                parameters['model_type'] = self.algorithms.index(parameters['algorithm'])
            if type(parameters['model_type']) == dict:
                parameters.update(**parameters['model_type'])
                parameters['model_type'] = self.algorithms.index(parameters['algorithm'])

        # Complete Dummy Parameters
        if complete_parameters:
            dummy_list = []
            for model_type in self.model_type_choices:
                dummy_list.extend(list(model_type.keys()))

            for parameters in parameters_list:
                for dummy_parameter in dummy_list:
                    if dummy_parameter not in parameters.keys():
                        parameters[dummy_parameter] = 0
        else:
            for parameters in parameters_list:
                filtered_keys = list(self.search_space.keys())
                filtered_keys += list(self.model_type_choices[parameters['model_type']].keys())

                dummy_parameters = parameters.copy()
                for parameter in dummy_parameters.keys():
                    if parameter not in filtered_keys:
                        parameters.pop(parameter)

        # Check Choice Parameters
        if choice_parameters == 'index':                   
            for parameters in parameters_list:
                choice_keys = [k for k in self.choice_parameters.keys() 
                               if k in parameters.keys() and type(parameters[k]) not in int_types]
                for choice_key in choice_keys:
                    parameters[choice_key] = self.choice_parameters[choice_key].index(parameters[choice_key])
        else:            
            for parameters in parameters_list:
                choice_keys = [k for k in self.choice_parameters.keys() 
                               if k in parameters.keys() and type(parameters[k]) in int_types]
                for choice_key in choice_keys:
                    parameters[choice_key] = self.choice_parameters[choice_key][parameters[choice_key]]

        # Check int parameters
        for parameters in parameters_list:
            for parameter in parameters:
                if parameter in self.int_parameters and parameters[parameter] is not None:
                    parameters[parameter] = int(parameters[parameter])

        return parameters_list

    def prepare_hyper_parameters(
        self,
        parameters: dict
    ):
        hyper_param_choices = [d for d in self.model_type_choices if d['algorithm'] == parameters['algorithm']][0]

        parameters['hyper_parameters'] = {
            hyper_param: parameters.pop(hyper_param) 
            for hyper_param in hyper_param_choices.keys()
            if hyper_param != 'algorithm'
        }

        parameters['hyper_parameters'] = {
            k.replace(f'{parameters["algorithm"]}.', ''): v
            for k, v in parameters['hyper_parameters'].items()
        }

        return parameters
    
    def prepare_parameters(
        self,
        parameters: dict,
        debug: bool = False
    ):
        if debug:
            t1 = time.time()
            print("parameters:\n"
                  "{")
            for key in parameters:
                if key != 'selected_features':
                    print(f"    '{key}': {parameters[key]}")
            print('}\n\n')

        parameters = self.parameter_configuration(
            parameters_list=[parameters],
            complete_parameters=False,
            choice_parameters='values',
            debug=False
        )[0]

        # Add version, stage & model_type
        if 'version' not in parameters.keys():
            parameters['version'] = '0.0'
        if 'stage' not in parameters.keys():
            parameters['stage'] = 'development'
        if 'model_type' in parameters.keys():
            parameters.pop('model_type')

        # Prepare Hyper Parameters
        parameters = self.prepare_hyper_parameters(
            parameters=parameters
        )

        if debug:
            print("new parameters:\n"
                  "{")
            for key in parameters:
                if key == 'selected_features' and parameters[key] is not None:
                    print(f"    '{key}' (len): {len(parameters[key])}")
                else:
                    print(f"    '{key}': {parameters[key]}")
            print('}\n')
            print(f'Time taken to prepare parameters: {round(time.time() - t1, 1)} sec.\n\n')
        
        return parameters
    
    def objective(
        self, 
        parameters: dict,
        debug: bool = False
    ):
        try:
            # Parameter configuration
            parameters = self.prepare_parameters(
                parameters=parameters,
                debug=debug
            )

            # Run Model Build Pipeline
            model: Model = self.ml_pipeline.build_pipeline(
                ml_params=parameters,
                eval_metric=self.eval_metric,
                splits=self.val_splits,
                debug=debug
            )
            
            # Log dev model
            if model.val_score >= self.min_performance:
                model.log_model()
            
            # Return Loss
            return {'loss': -model.val_score, 'status': STATUS_OK}
        except Exception as e:
            print(f'[WARNING] Skipping iteration.\n'
                  f'Exception: {e}\n'
                  f'Parameters:\n{parameters}\n\n')
            return {'loss': np.inf, 'status': STATUS_OK}

    def find_dev_models(self):
        self.dev_models = []
        for algorithm in self.algorithms:
            # Query algorithm runs & sort by validation performance
            filter_string = f"tag.algorithm = '{algorithm}'"

            runs_df: pd.DataFrame = mlflow.search_runs(
                experiment_names=[Params.experiment_name],
                filter_string=filter_string,
                order_by=['metrics.val_score']
            )

            # Find top candidates
            top_n = int(np.floor(self.n_candidates/len(self.algorithms)))
            top_runs = runs_df.iloc[:top_n]

            # Retrieve logged models and metadata
            for run_id in top_runs['run_id']:
                # Retrieve run
                run = Params.ml_client.get_run(run_id=run_id)

                # Instanciate Model
                model = Model(
                    artifact_uri=run.info.artifact_uri,
                    algorithm=run.data.tags.get('algorithm')
                )

                # Load model from logged run
                model.load_from_registry(load_model_from_tracking_server=True)

                # Append dev_model
                self.dev_models.append(model)

            # Delete unperformant runs
            drop_runs = runs_df.iloc[top_n:]
            for drop_id in drop_runs['run_id']:
                print(f'deleting run_id: {drop_id}')
                Params.ml_client.delete_run(run_id=drop_id)
            print('\n')

    def evaluate_dev_models(
        self,
        debug: bool = False
    ):
        for model in self.dev_models:
            # Evaluate model on test set & find model.test_score
            model.evaluate_test(
                X_test=self.ml_pipeline.X_test,
                y_test=self.ml_pipeline.y_test,
                eval_metric=self.eval_metric,
                debug=False
            )

            # Log model
            model.log_model()
        
        if debug:
            print('self.dev_models:')
            for model in self.dev_models:
                print(f'    - {model.model_id} ({model.stage}) '
                      f'(val_score: {round(model.val_score, 4)} '
                      f'test_score: {round(model.test_score, 4)}).')
            print('\n\n')

    def run(
        self,
        ml_df: pd.DataFrame,
        soft_debug: bool = False, 
        deep_debug: bool = False
    ):
        if deep_debug:
            soft_debug = True

        # Set up local tracking server
        self.model_registry.set_up_tracking_server()

        # Prepare Datasets
        self.ml_pipeline.prepare_datasets(
            ml_df=ml_df,
            train_test_ratio=self.train_test_ratio
        )

        # Define warm start
        warm_models = [model for model in self.dev_models if model.algorithm in self.algorithms]
        if len(warm_models) > 0 and warm_models[0].warm_start_params is not None:
            best_parameters_to_evaluate = self.parameter_configuration(
                parameters_list=[warm_models[0].warm_start_params],
                complete_parameters=True,
                choice_parameters='index',
                debug=soft_debug
            )
            trials = generate_trials_to_calculate(best_parameters_to_evaluate)

            if soft_debug:
                print(f'best_parameters_to_evaluate:')
                pprint(best_parameters_to_evaluate)
                print('\n\n')
        else:
            trials = None
        
        # Run hyperopt searching engine
        print(f'\n\nTuning Models (max_evals: {self.max_evals}):\n')
        try:
            result = fmin(
                fn=self.objective,
                space=self.search_space,
                algo=tpe.suggest,
                max_evals=self.max_evals,
                timeout=self.timeout_mins * 60,
                loss_threshold=self.loss_threshold,
                trials=trials,
                verbose=True,
                show_progressbar=True,
                early_stop_fn=None
            )
        except Exception as e:
            print(f'[WARNING] Exception occured while tuning hyperparameters.\n'
                  f'Exception: {e}\n\n')

        # Define dev_models
        self.find_dev_models()

        # Evaluate dev_models
        print("Evaluating Dev Models:\n")
        self.evaluate_dev_models(debug=soft_debug)

        # Save Models
        for model in self.dev_models:
            model.save()
        
        # Add Dev Models to local model registry
        self.model_registry.local_registry['development'] = [
            m.model_id for m in self.dev_models
            if m.stage == 'development'
        ]

    def __repr__(self) -> str:
        i = 1
        for model in self.dev_models[:5]:
            print(f'Dev Model {i}:')
            print(model)
            print('\n\n')
            i += 1
        return ''

