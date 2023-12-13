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

        # Performance
        'f1_score',
        'precision_score',
        'recall_score',
        'roc_auc_score',
        'accuracy_score',
        'cv_scores',
        'test_score'
    ]
    load_parquet = [
        'feature_importance_df'
    ]

    def __init__(
        self,
        model_id: str = None,
        artifact_uri: str = None,
        version: str = 1,
        stage: str = 'staging',
        algorithm: str = None,
        hyper_parameters: dict = {}
    ) -> None:
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

        self.file_name = self.find_file_name()
        self.model_name = self.find_model_name()

        # Model Parameters
        self.hyper_parameters = self.correct_hyper_parameters(hyper_parameters)
        if 'max_features' in self.hyper_parameters and self.hyper_parameters['max_features'] == '1.0':
            self.hyper_parameters['max_features'] = 1.0
        
        # Load Parameters
        self.model = None
        self.fitted: bool = False

        self.f1_score: float = 0
        self.precision_score: float = 0
        self.recall_score: float = 0
        self.roc_auc_score: float = 0
        self.accuracy_score: float = 0
        
        self.cv_scores: np.ndarray = np.ndarray([])
        self.test_score: float = 0

        self.feature_importance_df: pd.DataFrame = pd.DataFrame(columns=['feature', 'shap_value'])
    
    @property
    def warm_start_params(self):
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
    def metrics(self):
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
    def artifacts(self):
        return {
            'model_id': self.model_id,
            'file_name': self.file_name,
            'model_name': self.model_name,
            'fitted': self.fitted,
            'cv_scores': self.cv_scores,
            'feature_importance_df': self.feature_importance_df
        }

    @property
    def tags(self):
        return {
            'algorithm': self.algorithm,
            'stage': self.stage,
            'version': self.version
        }

    @property
    def val_score(self):
        return self.cv_scores.mean()

    @property
    def run_id(self):
        if self.artifact_uri is None:
            return None
        else:
            splits = self.artifact_uri.split('/')
            mlruns_idx = splits.index('mlruns')
            run_id = splits[mlruns_idx+2]
            return run_id

    def find_file_name(self):
        if self.algorithm == 'random_forest':
            return f"{self.model_id}_random_forest_model.pickle"

        elif self.algorithm == 'lightgbm':
            return f"{self.model_id}_lightgbm_model.pickle"
        
        elif self.algorithm == 'xgboost':
            return f"{self.model_id}_xgboost_model.pickle"
        
    def find_model_name(self):
        return f"{self.model_id}_{self.algorithm}_model"

    def correct_hyper_parameters(
        self,
        hyper_parameters: dict,
        debug: bool = False
    ) -> dict:
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
    ):        
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
    ):
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
        X_train: pd.DataFrame,
        y_train: pd.DataFrame,
        eval_metric: str,
        splits: int,
        debug: bool = False
    ) -> None:
        # Define scorer
        if eval_metric == 'f1_score':
            scorer = make_scorer(f1_score)

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
        X_test: pd.DataFrame,
        y_test: pd.DataFrame,
        eval_metric: str,
        debug: bool = False
    ) -> None:
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

        if eval_metric == 'f1_score':
            self.test_score = self.f1_score

        if debug:
            print(f'self.test_score: {self.test_score}\n')

    def predict(
        self,
        X: pd.DataFrame
    ):
        return self.model.predict(X.values.astype(float))
    
    def predict_proba(
        self,
        X: pd.DataFrame
    ):
        return self.model.predict_proba(X.values.astype(float))

    def find_feature_importance(
        self,
        test_features: pd.DataFrame,
        debug: bool = False
    ):
        try:
            explainer = shap.TreeExplainer(self.model)

            shap_values: np.ndarray = explainer.shap_values(test_features)
            shap_sum: np.ndarray = np.abs(shap_values).mean(axis=0)
            
            shap_importance_df = pd.DataFrame({
                'feature': test_features.columns.tolist(),
                'shap_value': shap_sum.tolist()
            })

            shap_importance_df.sort_values(by=['shap_value'], ascending=False, ignore_index=True, inplace=True)
            shap_importance_df['cum_perc'] = shap_importance_df['shap_value'].cumsum() / shap_importance_df['shap_value'].sum()

            self.feature_importance_df = shap_importance_df

            if debug:
                print(f'Shap importance df (top 20): \n{shap_importance_df.iloc[:20]}\n\n')
        except:                    
            if self.algorithm in ['random_forest', 'xgboost']:
                # booster = self.model.get_booster()
                # gain_score = booster.get_score(importance_type='gain')
                importance_df = pd.DataFrame({
                    'feature': test_features.columns.tolist(),
                    'importance': self.model.feature_importances_.tolist()
                })
                importance_df.sort_values(by=['importance'], ascending=False, ignore_index=True, inplace=True)
                importance_df['cum_perc'] = importance_df['importance'].cumsum() / importance_df['importance'].sum()

                self.feature_importance_df = importance_df
            else:
                self.feature_importance_df = pd.DataFrame(columns=['feature', 'importance', 'cum_perc'])
                print(f'[WARNING] Unable to generate self.feature_importance_df for {self.model_id}.\n')

    def diagnose_model(
        self
    ):
        # TODO: complete
        return {
            'passed': True
        }

    def save(self):
        # Save Model to file system
        self.save_to_file_system()

        # Save Model to tracking system
        self.log_model()

        # Register Model
        if self.stage != 'development':
            self.register_model()

    def save_to_file_system(self):
        """
        Step 1) Save .pickle files
        """
        model_attr = {key: value for (key, value) in self.__dict__.items() if key in self.load_pickle}

        with open(os.path.join(Params.model_attr_path, f"{self.model_id}_model_attr.pickle"), 'wb') as handle:
            pickle.dump(model_attr, handle, protocol=pickle.HIGHEST_PROTOCOL)

        """
        Step 2) Save .parquet files
        """
        for attr_name in self.load_parquet:
            df: pd.DataFrame = getattr(self, attr_name)
            if df is not None:
                df.to_csv(os.path.join(
                    Params.model_attr_path, f"{self.model_id}_model_{attr_name}.csv"
                ))

        """
        Step 4) Save self.model
        """
        if self.model is not None:
            save_path = os.path.join(Params.model_attr_path, self.file_name)
            joblib.dump(self.model, save_path)
    
    def log_model(self):
        def log_fun():
            # Find run_id
            run_id = mlflow.active_run().info.run_id

            # Find model path
            model_path = os.path.join('mlruns', str(Params.experiment_id), run_id, 'artifacts', self.model_name)

            # Save model locally
            if not os.path.exists(model_path):
                os.makedirs(model_path)
            
            joblib.dump(self.model, os.path.join(model_path, 'model.pkl'))

            # Log model
            if self.algorithm == 'random_forest':
                mlflow.sklearn.log_model(self.model, self.model_name)
            
            elif self.algorithm == 'lightgbm':
                mlflow.lightgbm.log_model(self.model, self.model_name)

            elif self.algorithm == 'xgboost':
                mlflow.xgboost.log_model(self.model, self.model_name)

            # Log hyper_paramters
            mlflow.log_params(self.hyper_parameters)

            # Log performance
            mlflow.log_metrics(self.metrics)
            
            # Save artifacts locally
            artifacts_path = os.path.join(Params.artifacts_path, f'artifacts.pickle')
            with open(artifacts_path, 'wb') as file:
                pickle.dump(self.artifacts, file)

            mlflow_artifacts_path = os.path.join('mlruns', str(Params.experiment_id), run_id, 'artifacts', 'artifacts')

            if not os.path.exists(mlflow_artifacts_path):
                os.makedirs(mlflow_artifacts_path)

            with open(os.path.join(mlflow_artifacts_path, 'artifacts.pickle'), 'wb') as file:
                pickle.dump(self.artifacts, file)

            # Log artifacts
            mlflow.log_artifact(artifacts_path, artifact_path=f'artifacts')

            # Set algorithm, stage & version tags
            mlflow.set_tags(self.tags)

            # Set self.model_uri
            self.artifact_uri = mlflow.active_run().info.artifact_uri

        # Start mlflow run to log results
        try:
            with mlflow.start_run(run_id=self.run_id, experiment_id=Params.experiment_id):
                log_fun()
        except Exception as e:
            print(f'[WARNING] Unable to start run {self.run_id} (experiment: {Params.experiment_id}).'
                  f'Exception: {e}\n\n')
            with mlflow.start_run(run_id=None, experiment_id=Params.experiment_id):
                log_fun()

    def register_model(self):
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

    def load_from_file_system(self):
        # Step 1) Load .pickle files
        try:
            with open(os.path.join(Params.model_attr_path, f"{self.model_id}_model_attr.pickle"), 'rb') as handle:
                model_attr: dict = pickle.load(handle)

            for attr_key, attr_value in model_attr.items():
                if attr_key in self.load_pickle:
                    setattr(self, attr_key, attr_value)

            self.file_name = self.find_file_name()
            self.model_name = self.find_model_name()
        except Exception as e:
            print(f'[WARNING] Unable to load model attr (f"{self.model_id}_model_attr.pickle).\n'
                f'Exception: {e}\n\n')
            
        # Step 2) Load .parquet files
        for attr_name in self.load_parquet:
            try:
                setattr(self, attr_name, pd.read_csv(
                    os.path.join(Params.model_attr_path, f"{self.model_id}_model_{attr_name}.csv")
                ))
            except Exception as e:
                print(f'[WARNING] Unable to load {self.model_id}_model_{attr_name}.parquet.\n\n')

        # Step 4) Load self.model
        try:
            load_path = os.path.join(Params.model_attr_path, self.file_name)
            self.model = joblib.load(load_path)
        except Exception as e:
            print(f'[WARNING] Unable to load model ({self.model_id}).\n'
                f'Exception: {e}\n\n')
    
    def load_from_registry(
        self,
        load_model_from_tracking_server: bool = False
    ):
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
            setattr(self, art_name, art_val)

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
    