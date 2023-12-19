from config.params import Params

from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier

from sklearn.model_selection import cross_val_score
from sklearn.metrics import (
    make_scorer, 
    f1_score, 
    precision_score, 
    recall_score, 
    roc_auc_score,
    accuracy_score
)

import mlflow
import shap
import joblib

import pandas as pd
import numpy as np

import secrets
import string
import os
import pickle
from pprint import pprint

from typing import List, Tuple
import warnings

warnings.filterwarnings("ignore")


class Model:
    """
    Class designed to homogenize the methods for building, evaluating, tracking & registering multiple types
    of ML classification models with different flavors/algorithms & hyperparameters, in a unified fashion. 
    """
    
    # Attributes to load as picke files, from the file system
    load_pickle = [
        # General Parameters
        'algorithm',
        'fitted',

        # Register Parameters
        'model_id',
        'artifact_uri',
        'version',
        'stage',

        # Model Parameters
        'hyper_parameters',

        # Feature Importance
        'shap_values',
        'importance_method',

        # Performance
        'f1_score',
        'precision_score',
        'recall_score',
        'roc_auc_score',
        'accuracy_score',
        'cv_scores',
        'test_score'
    ]
    
    # Attributes to load as csv files, from the file system
    load_csv = [
        'feature_importance_df'
    ]

    def __init__(
        self,
        model_id: str = None,
        artifact_uri: str = None,
        version: int = 1,
        stage: str = 'staging',
        algorithm: str = None,
        hyper_parameters: dict = {}
    ) -> None:
        """
        Initialize Model.

        :param `model_id`: (str) ID to tag & identify a Model instance.
        :param `artifact_uri`: (str) URI required to load a pickled model from the mlflow tracking server.
        :param `version`: (int) Model version, which increases by one each time the model gets re-fitted.
        :param `stage`: (str) Model stage, which can be either "development", "staging" or "production".
        :param `algorithm`: (str) Also known as model flavor. Current options are "random_forest", "lightgbm" 
         & "xgboost".
        :param `hyper_parameters`: (dict) Dictionart containing key-value pairs of the model hyper-parameters.
        """
        # General Parameters
        self.algorithm = algorithm

        # Register Parameters
        if model_id is not None:
            self.model_id = model_id
        else:
            self.model_id = ''.join(secrets.choice(string.ascii_letters) for i in range(10))

        self.artifact_uri = artifact_uri
        
        self.version = version
        self.stage = stage

        # Model Parameters
        self.hyper_parameters = self.correct_hyper_parameters(hyper_parameters)
        if 'max_features' in self.hyper_parameters and self.hyper_parameters['max_features'] == '1.0':
            self.hyper_parameters['max_features'] = 1.0
        
        # Load Parameters
        self.model: RandomForestClassifier or XGBClassifier or LGBMClassifier = None
        self.fitted: bool = False

        self.f1_score: float = 0
        self.precision_score: float = 0
        self.recall_score: float = 0
        self.roc_auc_score: float = 0
        self.accuracy_score: float = 0
        
        self.cv_scores: np.ndarray = np.ndarray([])
        self.test_score: float = 0

        self.feature_importance_df: pd.DataFrame = pd.DataFrame(columns=['feature', 'shap_value'])
        self.importance_method: str = None
        self.shap_values: np.ndarray = None
    
    @property
    def warm_start_params(self) -> dict:
        """
        Defines the parameters required for a warm start on the ModelTuner.run() method.
        Can be accesses as an attribute.

        :return: (dict) warm parameters.
        """
        algorithms: List[str] = Params.algorithms

        params = {
            # General Parameters
            'algorithm': self.algorithm,

            # Register Parameters            
            'model_id': self.model_id,
            'version': self.version,
            'stage': self.stage,

            # Others
            'model_type': algorithms.index(self.algorithm)
        }

        # Hyper-Parameters
        params.update(**{
            f'{self.algorithm}.{k}': v for k, v in self.hyper_parameters.items()
        })

        return params

    @property
    def metrics(self) -> dict:
        """
        Defines the test and validation metrics to be logged in the mlflow tracking server.
        Can be accessed as an attribute.

        :return: (dict) validation & test metrics.
        """
        return {
            'f1_score': self.f1_score,
            'precision_score': self.precision_score,
            'recall_score': self.recall_score,
            'roc_auc_score': self.roc_auc_score,
            'accuracy_score': self.accuracy_score,

            'val_score': self.val_score,
            'test_score': self.test_score
        }

    @property
    def artifacts(self) -> dict:
        """
        Defines the dictionary of attributes to be saved as artifacts in the mlflow tracking server.
        Can be accessed as an attribute.

        :return: (dict) Dictionary to save as artifacts.
        """
        return {
            'model_id': self.model_id,
            'fitted': self.fitted,
            'cv_scores': self.cv_scores,
            'feature_importance_df': self.feature_importance_df
        }

    @property
    def tags(self) -> dict:
        """
        Defines the tags to be saved in the mlflow tracking server.
        Can be accessed as an attribute.

        :return: (dict) Dictionary of tags.
        """
        return {
            'algorithm': self.algorithm,
            'stage': self.stage,
            'version': self.version
        }

    @property
    def val_score(self) -> float:
        """
        Defines the validation score as the mean value of the cross validation results.
        Can be accessed as an attribute.

        :return: (float) mean cross validation score.
        """
        if self.cv_scores is not None:
            return self.cv_scores.mean()
        return None

    @property
    def run_id(self) -> str:
        """
        Finds the run_id, which is accessed throughout the self.artifact_uri.
        Can be accessed as an attribute.

        :param: (str) Run ID.
        """
        if self.artifact_uri is None:
            return None
        else:
            splits = self.artifact_uri.split('/')
            mlruns_idx = splits.index('mlruns')
            run_id = splits[mlruns_idx+2]
            return run_id

    @property
    def file_name(self) -> str:
        """
        Defines the file name in which to save the self.model in the file system.
        Can be accessed as an attribute.

        :return: (str) file name
        """
        if self.algorithm == 'random_forest':
            return f"{self.model_id}_random_forest_model.pickle"

        elif self.algorithm == 'lightgbm':
            return f"{self.model_id}_lightgbm_model.pickle"
        
        elif self.algorithm == 'xgboost':
            return f"{self.model_id}_xgboost_model.pickle"
    
    @property
    def model_name(self) -> str:
        """
        Defines the model name used in the mlflow tracking server and mlflow model registry.
        Can be accessed as an attribute.

        :return: (str) model name.
        """
        return f"{self.model_id}_{self.algorithm}_model"

    def correct_hyper_parameters(
        self,
        hyper_parameters: dict,
        debug: bool = False
    ) -> dict:
        """
        Method that completes pre-defined hyperparameters.

        :param `hyper_parameters`: (dict) hyper_parameters that might not contain pre-defined hyperparameters.
        :param `debug`: (bool) Wether or not to show output hyper_parameters for debugging purposes.

        :return: (dict) hyper_parameters containing pre-defined hyperparameters.
        """
        if self.algorithm == 'random_forest':
            hyper_parameters.update(**{
                'oob_score': False,
                'n_jobs': -1,
                'random_state': 23111997
            })

        elif self.algorithm == 'lightgbm':
            hyper_parameters.update(**{
                "importance_type": 'split',
                "random_state": 23111997,
                "verbose": -1,
                "n_jobs": -1
            })

        elif self.algorithm == 'xgboost':
            hyper_parameters.update(**{
                "verbosity": 0,
                "use_rmm": True,
                "device": 'cuda', # 'cpu', 'cuda' # cuda -> GPU
                "nthread": -1,
                "n_gpus": -1,
                "max_delta_step": 0,
                "gamma": 0,
                "subsample": 1, # hp.uniform('xgboost.subsample', 0.6, 1)
                "sampling_method": 'uniform',
                "random_state": 23111997,
                "n_jobs": -1
            })

        if debug and hyper_parameters is not None:
            print("hyper_parameters:\n"
                  "{")
            for key in hyper_parameters.keys():
                print(f"    '{key}': {hyper_parameters[key]} ({type(hyper_parameters[key])})")
            print('}\n\n')

        return hyper_parameters

    def build(
        self,
        debug: bool = False
    ) -> None:
        """
        Method to instanciate the specified ML classification model, based on the model flavor/alrorithm
        & hyper-parameters.

        :param `debug`: (bool) Wether or not to show output hyper_parameters for debugging purposes.
        """
        if self.algorithm == 'random_forest':
            self.model = RandomForestClassifier(**self.hyper_parameters)
        
        elif self.algorithm == 'lightgbm':
            self.model = LGBMClassifier(**self.hyper_parameters)

        elif self.algorithm == 'xgboost':
            self.model = XGBClassifier(**self.hyper_parameters)

        else:
            raise Exception(f'Invalid algorithm: {self.algorithm}!')
        
        self.fitted = False
        
        if debug:
            print(f'self.model: {self.model}\n')

    def fit(
        self,
        y_train: pd.DataFrame = None,
        X_train: pd.DataFrame = None
    ) -> None:
        """
        Method to fit self.model.

        :param `y_train`: (pd.DataFrame) Binary & balanced train target.
        :param `X_train`: (pd.DataFrame) Train features.
        """
        self.model.fit(
            X_train.values.astype(float), 
            y_train[Params.target_column].values.astype(int).ravel()
        )
        
        # Update Version
        if not self.fitted:
            self.fitted = True
            self.version = 1
        else:
            self.version += 1

        # Update Last Fitting Date
        self.last_fitting_date = y_train.index[-1]

    def evaluate_val(
        self,
        y_train: pd.DataFrame,
        X_train: pd.DataFrame,
        eval_metric: str,
        splits: int,
        debug: bool = False
    ) -> None:
        """
        Method that will define a score metric (based on the eval_metric parameter) and will leverage
        the cross validation technique to obtain the validation scores.

        :param `y_train`: (pd.DataFrame) binary & balanced train target.
        :param `X_train`: (pd.DataFrame) Train features.
        :param `eval_metric`: (str) Metric to measure on each split of the cross validation.
        :param `splits`: (int) Number of splits to perform in the cross validation.
        :param `debug`: (bool) Wether or not to show self.cv_scores, for debugging purposes.
        """
        # Define scorer
        if eval_metric == 'f1_score':
            scorer = make_scorer(f1_score)
        elif eval_metric == 'precision':
            scorer = make_scorer(precision_score)
        elif eval_metric == 'recall':
            scorer = make_scorer(recall_score)
        elif eval_metric == 'roc_auc':
            scorer = make_scorer(roc_auc_score)
        elif eval_metric == 'accuracy':
            scorer = make_scorer(accuracy_score)
        else:
            raise Exception(f'Invalid "eval_metric": {eval_metric}.\n\n')

        # Evaluate Model using Cross Validation
        self.cv_scores = cross_val_score(
            self.model, 
            X_train.values.astype(float), 
            y_train[Params.target_column].values.astype(int).ravel(),
            cv=splits, 
            scoring=scorer
        )
        
        if debug:
            print(f'self.cv_scores: {self.cv_scores}\n\n')

    def evaluate_test(
        self,
        y_test: pd.DataFrame,
        X_test: pd.DataFrame,
        eval_metric: str,
        debug: bool = False
    ) -> None:
        """
        Method that will predict test set values and define the following test metrics:
            - self.f1_score
            - self.precision_score
            - self.recall_score
            - self.roc_auc_score
            - self.accuracy_score
            - self.test_score (utilized to define champion model)

        :param `y_test`: (pd.DataFrame) Binary & un-balanced test target.
        :param `X_test`: (pd.DataFrame) Test features.
        :param `eval_metric`: (str) Metric utilized to define the self.test_score attribute.
        :param `debug`: (bool) Wether or not to show self.test_score, for debugging purposes.
        """
        # Predict test values
        test_preds = self.model.predict(X_test.values)

        # Evaluate F1 Score
        self.f1_score = f1_score(y_test.values.astype(int), test_preds)

        # Evaluate Precision Score
        self.precision_score = precision_score(y_test.values.astype(int), test_preds)

        # Evaluate Recall Score
        self.recall_score = recall_score(y_test.values.astype(int), test_preds)

        # ROC AUC Score
        self.roc_auc_score = roc_auc_score(y_test.values.astype(int), test_preds)

        # Accuracy Score
        self.accuracy_score = accuracy_score(y_test.values.astype(int), test_preds)

        # Define test score
        if eval_metric == 'f1_score':
            self.test_score = self.f1_score
        elif eval_metric == 'precision':
            self.test_score = self.precision_score
        elif eval_metric == 'recall':
            self.test_score = self.recall_score
        elif eval_metric == 'roc_auc':
            self.test_score = self.roc_auc_score
        elif eval_metric == 'accuracy':
            self.test_score = self.accuracy_score
        else:
            raise Exception(f'Invalid "eval_metric": {eval_metric}.\n\n')

        if debug:
            print(f'self.test_score ({eval_metric}): {self.test_score}\n')

    def predict(
        self,
        X: pd.DataFrame
    ) -> np.ndarray:
        """
        Method for realizing new category inferences.

        :param `X`: (pd.DataFrame) New features to make inferences on.

        :return: (np.ndarray) New category inferences.
        """
        return self.model.predict(X.values.astype(float))
    
    def predict_proba(
        self,
        X: pd.DataFrame
    ) -> np.ndarray:
        """
        Method for realizing new probabilistic inferences.

        :param `X`: (pd.DataFrame) New features to make inferences on.

        :return: (np.ndarray) New probabilistic inferences.
        """
        return self.model.predict_proba(X.values.astype(float))

    def find_feature_importance(
        self,
        X_test: pd.DataFrame,
        find_new_shap_values: bool = False,
        debug: bool = False
    ) -> None:
        """
        Method that utilizes the shap library to calculate feature impotances on the test dataset 
        (whenever possible).

        :param `test_features`: (pd.DataFrame) Test features.
        :param `find_new_shap_values`: (bool) Wether or not to calculate new shaply values.
        :param `debug`: (bool) Wether or not to show top feature importances, for debugging purposes.
        """
        try:
            if find_new_shap_values or self.shap_values is None:
                print(f'Calculating new shaply values for {self.model_id}.\n\n')
                # Instanciate explainer
                explainer = shap.TreeExplainer(self.model)

                # Calculate shap values
                self.shap_values: np.ndarray = explainer.shap_values(X_test)

            # Find the sum of feature values
            shap_sum = np.abs(self.shap_values).mean(0).sum(0)

            # Find shap feature importance DataFrame
            importance_df = pd.DataFrame({
                'feature': X_test.columns.tolist(),
                'importance': shap_sum
            })

            self.importance_method = 'shap'
        except Exception as e:
            print(f'[WARNING] Unable to calculate shap feature importance on {self.model_id} ({self.algorithm}).\n'
                  f'Exception: {e}\n'
                  f'Re-trying with a native approach.\n\n')
            
            # Define DataFrame to describe importances on (utilizing native feature importance calculation method)
            importance_df = pd.DataFrame({
                'feature': X_test.columns.tolist(),
                'importance': self.model.feature_importances_.tolist()
            })

            self.shap_values = None
            self.importance_method = f'native_{self.algorithm}'

        # Sort DataFrame by shap_value
        importance_df.sort_values(by=['importance'], ascending=False, ignore_index=True, inplace=True)

        # Find shap cumulative percentage importance
        importance_df['cum_perc'] = importance_df['importance'].cumsum() / importance_df['importance'].sum()

        # Assign result to the self.feature_importance_df attribute
        self.feature_importance_df = importance_df
        
        if debug:
            print(f'Shap importance df (top 20): \n{importance_df.iloc[:20]}\n\n')

    def diagnose_model(
        self
    ) -> dict:
        """
        Work in Process.
        """
        return {
            'passed': True
        }

    def save(self) -> None:
        """
        Method used to save the Model attributes on file system, mlflow tracking server and
        mlflow model registry.
        """
        # Save Model to file system
        self.save_to_file_system()

        # Save Model to tracking system
        self.log_model()

        # Register Model
        if self.stage != 'development':
            self.register_model()

    def save_to_file_system(self) -> None:
        """
        Method that will save Model's attributes in file system.
        """
        # Save .pickle files
        model_attr = {key: value for (key, value) in self.__dict__.items() if key in self.load_pickle}

        with open(os.path.join(Params.model_attr_path, f"{self.model_id}_model_attr.pickle"), 'wb') as handle:
            pickle.dump(model_attr, handle, protocol=pickle.HIGHEST_PROTOCOL)

        # Save .parquet files
        for attr_name in self.load_csv:
            df: pd.DataFrame = getattr(self, attr_name)
            if df is not None:
                df.to_csv(os.path.join(
                    Params.model_attr_path, f"{self.model_id}_model_{attr_name}.csv"
                ))

        # Save self.model
        if self.model is not None:
            save_path = os.path.join(Params.model_attr_path, self.file_name)
            joblib.dump(self.model, save_path)
    
    def log_model(self) -> None:
        """
        Method that will log the following attributes on mlflow tracking server:
            - self.model
            - self.hyper_parameters
            - self.metrics
            - self.artifacts
            - self.tags
        """
        def log_fun():
            # Find run_id
            run_id = mlflow.active_run().info.run_id

            # Log model
            if self.algorithm == 'random_forest':
                mlflow.sklearn.log_model(self.model, self.model_name)
            
            elif self.algorithm == 'lightgbm':
                mlflow.lightgbm.log_model(self.model, self.model_name)

            elif self.algorithm == 'xgboost':
                mlflow.xgboost.log_model(self.model, self.model_name)

            # Find model path
            model_path = os.path.join('mlruns', str(Params.experiment_id), run_id, 'artifacts', self.model_name)

            # Check that the model was in fact logged
            if not os.path.exists(model_path):
                # Create directory
                os.makedirs(model_path)

                # Save model manually
                joblib.dump(self.model, os.path.join(model_path, 'model.pkl'))

            # Log hyper_paramters
            mlflow.log_params(self.hyper_parameters)

            # Log performance
            mlflow.log_metrics(self.metrics)
            
            # Save artifacts locally
            artifacts_path = os.path.join(Params.artifacts_path, f'artifacts.pickle')
            with open(artifacts_path, 'wb') as file:
                pickle.dump(self.artifacts, file)

            # Log artifacts
            mlflow.log_artifact(artifacts_path, artifact_path=f'artifacts')

            # Find artifacts path
            mlflow_artifacts_path = os.path.join('mlruns', str(Params.experiment_id), run_id, 'artifacts', 'artifacts')

            # Check if artifacts were in fact logged
            if not os.path.exists(mlflow_artifacts_path):
                # Create directorty
                os.makedirs(mlflow_artifacts_path)

                # Save artifacts manually
                with open(os.path.join(mlflow_artifacts_path, 'artifacts.pickle'), 'wb') as file:
                    pickle.dump(self.artifacts, file)

            # Set algorithm, stage & version tags
            mlflow.set_tags(self.tags)

            # Set self.model_uri
            self.artifact_uri = mlflow.active_run().info.artifact_uri

        # Start mlflow run to log results
        try:
            # Try to log model using self.run_id
            with mlflow.start_run(run_id=self.run_id, experiment_id=Params.experiment_id):
                log_fun()
        except Exception as e:
            print(f'[WARNING] Unable to start run {self.run_id} (experiment: {Params.experiment_id}).'
                  f'Exception: {e}\n\n')
            # Re-try utilizing no pre-specified run_id
            with mlflow.start_run(run_id=None, experiment_id=Params.experiment_id):
                log_fun()

    def register_model(self) -> None:
        """
        Method that will register the model in the mlflow model registry. It will additionally set the tags,
        current version and current stage.
        """
        # Delete registerd model (if it already exists)
        reg_model_names = [reg.name for reg in mlflow.search_registered_models()]
        if self.model_name in reg_model_names:
            Params.ml_client.delete_registered_model(name=self.model_name)

        # Register model
        model_uri = f'runs:/{self.run_id}/{self.model_name}'
        name = self.model_name
        tags = {
            'algorithm': self.algorithm,
            'validation_score': self.val_score,
            'test_score': self.test_score
        }

        print(f'\nregistering: {name} (from: {model_uri})')
        mlflow.register_model(
            model_uri=model_uri,
            name=name,
            tags=tags
        )

        # Define stage & version
        Params.ml_client.transition_model_version_stage(
            name=self.model_name,
            version=self.version,
            stage=self.stage
        )

    def load_from_file_system(self) -> None:
        """
        Method that will load model attributes from the file system.
        """
        # Load .pickle files
        try:
            with open(os.path.join(Params.model_attr_path, f"{self.model_id}_model_attr.pickle"), 'rb') as handle:
                model_attr: dict = pickle.load(handle)

            for attr_key, attr_value in model_attr.items():
                if attr_key in self.load_pickle:
                    setattr(self, attr_key, attr_value)
        except Exception as e:
            print(f'[WARNING] Unable to load model attr (f"{self.model_id}_model_attr.pickle).\n'
                  f'Exception: {e}\n\n')
            
        # Load .parquet files
        for attr_name in self.load_csv:
            try:
                setattr(self, attr_name, pd.read_csv(
                    os.path.join(Params.model_attr_path, f"{self.model_id}_model_{attr_name}.csv")
                ))
            except Exception as e:
                print(f'[WARNING] Unable to load {self.model_id}_model_{attr_name}.parquet.\n\n')

        # Load self.model
        try:
            load_path = os.path.join(Params.model_attr_path, self.file_name)
            self.model = joblib.load(load_path)
        except Exception as e:
            print(f'[WARNING] Unable to load model ({self.model_id}).\n'
                  f'Exception: {e}\n\n')
    
    def load_from_registry(
        self,
        load_model_from_tracking_server: bool = False
    ) -> None:
        """
        Method that will load tags, parameters, metrics and artifacts from the mlflow tracking server,
        and will load self.model from the mlflow model registry (if specified).

        :param `load_model_from_tracking_server`: (bool) Wether or not to load self.model from the mlflow
         tracking server or the model registry.
        """
        # Find run_id
        run = Params.ml_client.get_run(run_id=self.run_id)

        # Load tags
        for tag_name in self.tags.keys():
            # Assign attribute
            setattr(self, tag_name, run.data.tags.get(tag_name))

        # Load hyper_parameters
        self.hyper_parameters = run.data.params

        # Load metrics
        for metric_name, metric_val in run.data.metrics.items():
            if metric_name != 'val_score':
                # Set attribute
                setattr(self, metric_name, metric_val)

        # Load artifacts
        artifact_path = os.path.join(
            'mlruns', str(Params.experiment_id), self.run_id, 'artifacts', 'artifacts', 'artifacts.pickle'
        )
        
        with open(artifact_path, 'rb') as handle:
            artifacts: dict = pickle.load(handle)
        
        for art_name, art_val in artifacts.items():
            # Set attribute
            try:
                setattr(self, art_name, art_val)
            except Exception as e:
                print(f'[WARNING] Unable to load {art_name}.\n'
                      f'Exception: {e}\n\n')

        # Load model
        if load_model_from_tracking_server:
            # Load model from tracking server
            model_uri = os.path.join(
                'mlruns', str(Params.experiment_id), self.run_id, 'artifacts', self.model_name
            )
        else:
            # Load model from MLflow Model Registry
            model_uri = f'models:/{self.model_name}/{self.version}'

        try:
            if self.algorithm == 'random_forest':
                self.model = mlflow.sklearn.load_model(model_uri)
            elif self.algorithm == 'lightgbm':
                self.model = mlflow.lightgbm.load_model(model_uri)
            elif self.algorithm == 'xgboost':
                self.model = mlflow.xgboost.load_model(model_uri)
        except:
            model_path = os.path.join(
                'mlruns', str(Params.experiment_id), self.run_id, 'artifacts', self.model_name, 'model.pkl'
            )
            self.model = joblib.load(model_path)

    def __repr__(self) -> str:
        print('Model:\n')

        # Register Parameters
        print(f'Attributes:')
        pprint({
            'Model ID': self.model_id,
            'Artifact URI': self.artifact_uri,
            'Model Name': self.model_name,
            'Model File Name': self.file_name,
            'Version': self.version,
            'Stage': self.stage,
            'Algorithm': self.algorithm,
            'Hyper Parameters': self.hyper_parameters,
            'Test Score': self.test_score,
            'Validation Score': self.val_score 
        })
        print('\n\n')

        return ''
    