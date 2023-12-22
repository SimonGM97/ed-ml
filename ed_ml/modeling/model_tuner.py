from config.params import Params
from ed_ml.modeling.model import Model
from ed_ml.modeling.model_registry import ModelRegistry
from ed_ml.pipeline.pipeline import MLPipeline
from ed_ml.utils.timing import timing

import mlflow
from hyperopt import fmin, hp, tpe, STATUS_OK
from hyperopt.fmin import generate_trials_to_calculate
from hyperopt.pyll.base import scope

import pandas as pd
import numpy as np
import time
from functools import partial
from pprint import pprint
from typing import List
import warnings

warnings.filterwarnings("ignore")


class ModelTuner:
    """
    Class designed to find the most performant classification ML models, leveraging hyperopt's TPE based
    search engine to optimize both the model flavor (or algorithm) & set of hyperparameters in order 
    to train robust models with strong generalization capabilities.
    """

    # Integer Serch Parameters
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

    # Choice Search Parameters
    choice_parameters = {
        # random_forest
        'random_forest.max_features': ['1.0', 'sqrt'],
        'random_forest.criterion': ['gini', 'entropy', 'log_loss'],

        # lightgbm
        'lightgbm.boosting_type': ['gbdt', 'dart'],

        # xgboost
        'xgboost.objective': ['binary:logistic', 'binary:logitraw', 'binary:hinge'],
        'xgboost.booster': ['gbtree', 'dart']
    }

    # Model Type Choices
    model_type_choices = [
        # Random Forest Search Space
        {
            "algorithm": 'random_forest',
            "random_forest.criterion": hp.choice('random_forest.criterion', choice_parameters['random_forest.criterion']),
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
            "xgboost.objective": hp.choice('xgboost.objective', choice_parameters['xgboost.objective']),
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
        """
        Initialize ModelTuner.

        :param `algorithms`: (list) Model flavors to iterate over. 
            - Currently available options: random_forest, lightgbm, xgboost.
        :param `eval_metric`: (str) Name of the metric utilized to evaluate ML models over the validation set.
            - Note: this is also the metric which will be optimized by the TPE algorithm.
        :param `val_splits`: (int) Number of splits utilized in for cross validation of ML model candidates.
        :param `train_test_ratio`: (float) Proportion of data to keep as the test set; relative to the complete
         dataset.
        :param `n_candidates`: (int) Number of development models that will be chosen as potential candidates.
        :param `local_registry`: (bool) Wether or not to load models from the file system or MLflow Model Registry.
        :param `max_evals`: (int) Number of maximum iterations that the hyperopt.fmin() function will be allowed to
         search before finding the most performant candidates.
        :param `timeout_mins`: (int) Number of minutes that the hyperopt.fmin() function will be allowed to run 
         before finding the most performant candidates.
        :param `loss_theshold`: (float) Theshold performance at which, if reached, the optimization algorithm will
         sease searching for a better model.
        :param `min_performance`: (float) Minimum performance required for a candidate model to be logged in the 
         mlflow tracking server.
        """
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
            load_from_local_registry=local_registry
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
        choice_parameters: str = 'index'
    ) -> List[dict]:
        """
        Method designed to interprete and complete the keys, values and indexes for each iteration of the search 
        parameters; following expected input & output structure expected by hyperopt.

        :param `parameters_list`: (list) List of parameters to standardize.
        :param `complete_parameters`: (bool) Wether or not to complete any missing keys in the parameters.
        :param `choice_parameters`: (str) Used to set how the choice parameters will be outputed.

        :return: (pd.DataFrame) List of parameters with standardized structure.
        """
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
    ) -> dict:
        """
        Method that standardizes the structure of hyper-parameters, so that it can be consumed while 
        instanciating new ML classification models.

        :param `parameters`: (dict) Parameters with hyper-parameters to standardize.

        :return: (dict) Parameters with standardized hyper-parameters.
        """
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
    ) -> dict:
        """
        Method designed to standardize the structure, complete required keys and interpret values 
        of the set of parameters & hyperparameters that are being searched by the hyperopt TPE 
        powered seach engine.

        :param `parameters`: (dict) Parameters with raw structure, uncomplete keys & uninterpreted values.
        :param `debug`: (bool) Wether or not to show input and output parameters for debugging purposes.

        :return: (dict) Parameters with standardized structure, complete keys & interpreted values.
        """
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
            choice_parameters='values'
        )[0]

        # Add version, stage, cutoff & drop model_type
        if 'version' not in parameters.keys():
            parameters['version'] = 0
        if 'stage' not in parameters.keys():
            parameters['stage'] = 'development'
        if 'cutoff' not in parameters.keys():
            parameters['cutoff'] = 0.5
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
    ) -> dict:
        """
        Method defined as the objective function for the hyperopt's TPE based search engine; which will:
            - Standardize, complete & interprete inputed parameters
            - Leverage MLPipeline to build a ML classification model with the inputed parameters
            - Log the resulting model in the mlflow tracking server, if the validation performance is over
              a defined threshold.
            - Output the validation performance (mean cross validation score) as the loss function.

        :param `parameters`: (dict) Parameters with raw structure, uncomplete keys & uninterpreted values.
        :param `debug`: (bool) Wether or not to show intermediate logs for debugging purposes.

        :return: (dict) Loss function with the validation performance of the ML classification model.
        """
        # try:
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
        # except Exception as e:
        #     print(f'[WARNING] Skipping iteration.\n'
        #           f'Exception: {e}\n'
        #           f'Parameters:\n{parameters}\n\n')
        #     return {'loss': np.inf, 'status': STATUS_OK}

    def find_dev_models(self) -> None:
        """
        Method that finds the most performant development models built by the search engine, by:
            - Querying the top mlflow tracking server runs for each model flavor/algorithm, based on the mean
              cross validation score.
            - Deleting unperformant runs from the tracking server.
            - Add found dev models to local registry and save changes made.
        """
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
        
        # Add Dev Models to model local IDs registry
        self.model_registry.local_registry['development'] = [
            m.model_id for m in self.dev_models
            if m.stage == 'development'
        ]

        # Save model local IDs registry
        self.model_registry.save_local_registry()

    def evaluate_dev_models(
        self,
        debug: bool = False
    ) -> None:
        """
        Method that evaluates the development models on the test set, defined in the MLPipeline.

        :param `debug`: (bool) Wether or not to show self.dev_models performances logs for debugging purposes.
        """
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

    def save_dev_models(self) -> None:
        """
        Method that will save development models in:
            - Tracking server
            - File system
        """
        for model in self.dev_models:
            model.save()

    @timing
    def run(
        self,
        ml_df: pd.DataFrame,
        balance_train: bool = True,
        balance_method: str = 'SMOT',
        use_warm_start: bool = True,
        soft_debug: bool = False, 
        deep_debug: bool = False
    ) -> None:
        """
        Main method that orchestrates the processes required to track, train and evaluate performant
        development models.

        This method will:
            - Set up the mlflow tracking server (currently hosted locally).
            - Define a balanced training set (which will be later divided in the cross validation section).
            - Define a test set (unbalanced, in order to accurately depict the real group distributions).
            - (Optional) Set up a "warm start" for the search engine, leveraging a performant solution found 
              on a previous run.
                - Note: utilizing warm start will find performant solutions potentially from the first iteration,
                  but the search algorithm is predisposed to find local minima.
            - Run the hyperopt's TPE based search engine.
            - Load the most performant development models, based on the mean cross validation score (on the 
              validation set).
            - Evaluate the development models on the unbalanced test set.
            - Save development models.
        
        :param `ml_df`: (pd.DataFrame) Engineered DataFrame outputted by the FeatureEngineer class.
        :param `balance_train`: (bool) Wether or not to balance train datasets.
        :param `balance_method`: (str) Methodology utilized to balance train datasets.
        :param `use_warm_start`: (bool) Wether or not to utilize the optional warm start functionality.
        :param `soft_debug`: (bool) Wether or not to show general intermediate logs for debugging purposes.
        :param `deep_debug`: (bool) Wether or not to show intermediate logs in the objective function, for 
         debugging purposes.
        """
        if deep_debug:
            soft_debug = True

        # Set up local mlflow tracking server
        self.model_registry.set_up_tracking_server()

        # Prepare Datasets
        self.ml_pipeline.prepare_datasets(
            ml_df=ml_df,
            train_test_ratio=self.train_test_ratio,
            balance_train=balance_train,
            balance_method=balance_method,
        )

        # Define warm start
        warm_models = [model for model in self.dev_models if model.algorithm in self.algorithms]
        if (
            use_warm_start
            and len(warm_models) > 0 
            and warm_models[0].warm_start_params is not None
        ):
            best_parameters_to_evaluate = self.parameter_configuration(
                parameters_list=[warm_models[0].warm_start_params],
                complete_parameters=True,
                choice_parameters='index'
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

        fmin_objective = partial(
            self.objective,
            debug=deep_debug
        )

        try:
            result = fmin(
                fn=fmin_objective,
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
        self.save_dev_models()

    def __repr__(self) -> str:
        i = 1
        for model in self.dev_models[:5]:
            print(f'Dev Model {i}:')
            print(model)
            print('\n\n')
            i += 1
        return ''

