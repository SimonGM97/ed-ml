import configparser
from typing import List
from pprint import pprint
import mlflow
import os


class Params:

    initialized: bool = False

    # DataProcessing
    raw_data_path: str
    cleaned_data_path: str
    ml_data_path: str

    ignore_features: List[str]
    datetime_columns: List[str]

    # Modeling
    model_attr_path: str
    model_registry_path: str

    target_column: str
    algorithms: List[str]
    eval_metric: str 
    val_splits: int 
    train_test_ratio: float 

    n_candidates: int 
    max_evals: int
    timeout_mins: int
    loss_threshold: float 
    min_performance: float
    
    # Inference
    request_url: str
    inference_path: str 

    @classmethod
    def load(cls) -> None:
        if cls.initialized:
            return
        
        # Create an instance of the ConfigParser class
        config = configparser.ConfigParser()  

        # Read the contents of the `config.ini` file:
        config.read(os.path.join("config", "config.ini"))

        # Access parameters from DataProcessing
        cls.raw_data_path = os.path.join(*config.get("DataProcessing", "raw_data_path").split(', '))
        cls.cleaned_data_path = os.path.join(*config.get("DataProcessing", "cleaned_data_path").split(', '))
        cls.ml_data_path = os.path.join(*config.get("DataProcessing", "ml_data_path").split(', '))

        cls.ignore_features = config.get("DataProcessing", "ignore_features").split(', ')
        cls.datetime_columns = config.get("DataProcessing", "datetime_columns").split(', ')

        # Access parameters from Modelling
        cls.model_attr_path = os.path.join(*config.get("Modeling", "model_attr_path").split(', '))

        cls.target_column = config.get("Modeling", "target_column")
        cls.algorithms = config.get("Modeling", "algorithms").split(', ')
        cls.eval_metric = config.get("Modeling", "eval_metric")
        cls.val_splits = int(config.get("Modeling", "val_splits"))
        cls.train_test_ratio = float(config.get("Modeling", "train_test_ratio"))

        cls.n_candidates = int(config.get("Modeling", "n_candidates"))
        cls.max_evals = int(config.get("Modeling", "max_evals"))
        cls.timeout_mins = int(config.get("Modeling", "timeout_mins"))
        cls.loss_threshold = float(config.get("Modeling", "loss_threshold"))
        cls.min_performance = float(config.get("Modeling", "min_performance"))

        # Mlflow
        cls.tracking_url = config.get("Mlflow", "tracking_url")

        local_registry = config.get("Mlflow", "local_registry")
        if local_registry == 'True':
            cls.local_registry = True
        else:
            cls.local_registry = False
        
        cls.local_registry_path = os.path.join(*config.get("Mlflow", "local_registry_path").split(', '))
        cls.artifacts_path = os.path.join(*config.get("Mlflow", "artifacts_path").split(', '))
        cls.experiment_name = config.get("Mlflow", "experiment_name")

        # Define client
        cls.ml_client = mlflow.tracking.MlflowClient(tracking_uri=f"file://{os.getcwd()}/mlruns")

        # Define Experiment
        experiment = cls.ml_client.get_experiment_by_name(name=cls.experiment_name)
        if experiment is None:
            cls.ml_client.create_experiment(name=cls.experiment_name)
        
        # Get experiment_id
        cls.experiment_id = int(cls.ml_client.get_experiment_by_name(name=cls.experiment_name).experiment_id)        
        
        # Access parameters from Inference
        cls.request_url = config.get("Inference", "request_url")
        cls.inference_path = os.path.join(*config.get("Inference", "inference_path").split(', '))

        # Access parameters from Default
        cls.raw_df = config.get("Default", "raw_df")
        if cls.raw_df == 'None':
            cls.raw_df = None

        save = config.get("Default", "save")
        if save == 'True':
            cls.save = True
        else:
            cls.save = False


if not Params.initialized:
    Params.load()

