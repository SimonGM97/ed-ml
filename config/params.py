import configparser
from typing import List
from pprint import pprint
import mlflow
import os


class Params:

    initialized: bool = False

    # Versioning
    version: str

    # DataProcessing
    raw_data_path: str
    cleaned_data_path: str
    ml_data_path: str
    
    target_column: str
    datetime_columns: list

    # Modeling
    model_attr_path: str

    algorithms: str
    eval_metric: str
    val_splits: int
    train_test_ratio: float
    n_candidates: int
    max_evals: int
    timeout_mins: int
    loss_threshold: float
    min_performance: float

    # Mlflow
    tracking_url: str
    local_registry: bool
    local_registry_path: str
    artifacts_path: str
    experiment_name: str
    
    # Inference
    request_url: str
    inference_path: str
    course_name: str
    user_uuids: list
    course_uuids: list
    particion: int
    pick_random: bool

    # Updating
    refit_model: bool
    find_new_shap_values: bool

    # Default
    raw_df: str
    save: bool

    @classmethod
    def load(cls) -> None:
        def read_param(key: str, val: str, as_type: str):
            # Extract paramater from config file
            param = config.get(key, val)

            # None
            if param == 'None':
                return None
            
            # Path
            if val.endswith('path'):
                return os.path.join(*param.split(', '))

            # String
            if as_type == str:
                return param
            
            # Boolean
            if as_type == bool:
                if param == 'True':
                    return True
                return False

            # Integer
            if as_type == int:
                return int(param)
            
            # Float
            if as_type == float:
                return float(param)
            
            # List
            if as_type == list:
                return param.split(', ')
            
            raise Exception(f'Invalid "as_type" parameter: {as_type}.\n\n')
            
        if cls.initialized:
            return
        
        # Create an instance of the ConfigParser class
        config = configparser.ConfigParser()  

        # Read the contents of the `config.ini` file:
        config.read(os.path.join("config", "config.ini"))

        # Access parameters from Versioning
        cls.version: str = read_param('Versioning', 'version', str)

        # Access parameters from DataProcessing
        cls.raw_data_path: str = read_param('DataProcessing', 'raw_data_path', str)
        cls.cleaned_data_path: str = read_param('DataProcessing', 'cleaned_data_path', str)
        cls.ml_data_path: str = read_param('DataProcessing', 'ml_data_path', str)
        
        cls.target_column: str = read_param('DataProcessing', 'target_column', str)
        cls.datetime_columns: list = read_param('DataProcessing', 'datetime_columns', list)

        # Access parameters from Modeling
        cls.model_attr_path: str = read_param('Modeling', 'model_attr_path', str)

        cls.algorithms: str = read_param('Modeling', 'algorithms', str)
        cls.eval_metric: str = read_param('Modeling', 'eval_metric', str)
        cls.val_splits: int = read_param('Modeling', 'val_splits', int)
        cls.train_test_ratio: float = read_param('Modeling', 'train_test_ratio', float)
        cls.n_candidates: int = read_param('Modeling', 'n_candidates', int)
        cls.max_evals: int = read_param('Modeling', 'max_evals', int)
        cls.timeout_mins: int = read_param('Modeling', 'timeout_mins', int)
        cls.loss_threshold: float = read_param('Modeling', 'loss_threshold', float)
        cls.min_performance: float = read_param('Modeling', 'min_performance', float)

        # Access parameters from Mlflow
        cls.tracking_url: str = read_param('Mlflow', 'tracking_url', str)
        cls.local_registry: bool = read_param('Mlflow', 'local_registry', bool)
        cls.local_registry_path: str = read_param('Mlflow', 'local_registry_path', str)
        cls.artifacts_path: str = read_param('Mlflow', 'artifacts_path', str)
        cls.experiment_name: str = read_param('Mlflow', 'experiment_name', str)

        # Define client
        cls.ml_client = mlflow.tracking.MlflowClient(tracking_uri=f"file://{os.getcwd()}/mlruns")

        # Define Experiment
        experiment = cls.ml_client.get_experiment_by_name(name=cls.experiment_name)
        if experiment is None:
            cls.ml_client.create_experiment(name=cls.experiment_name)
        
        # Get experiment_id
        cls.experiment_id = int(cls.ml_client.get_experiment_by_name(name=cls.experiment_name).experiment_id)        
        
        # Access parameters from Inference
        cls.request_url: str = read_param('Inference', 'request_url', str)
        cls.inference_path: str = read_param('Inference', 'inference_path', str)
        cls.course_name: str = read_param('Inference', 'course_name', str)
        cls.user_uuids: list = read_param('Inference', 'user_uuids', list)
        cls.course_uuids: list = read_param('Inference', 'course_uuids', list)
        cls.particion: int = read_param('Inference', 'particion', int)
        cls.pick_random: bool = read_param('Inference', 'pick_random', bool)

        # Access parameters from Updating
        cls.refit_model: bool = read_param('Updating', 'refit_model', bool)
        cls.find_new_shap_values: bool = read_param('Updating', 'find_new_shap_values', bool)

        # Access parameters from Default
        cls.raw_df: str = read_param('Default', 'raw_df', str)
        cls.save: bool = read_param('Default', 'save', bool)


# Initialize Params
if not Params.initialized:
    Params.load()

