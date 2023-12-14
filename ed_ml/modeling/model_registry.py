from config.params import Params
from ed_ml.modeling.model import Model
from ed_ml.pipeline.pipeline import MLPipeline
import mlflow
import pandas as pd
import os
import json
import subprocess
import shutil
import time
from pprint import pprint
from typing import List, Dict, Tuple
import warnings

warnings.filterwarnings("ignore")


class ModelRegistry:
    
    def __init__(
        self,
        n_candidates: int,
        local_registry: bool = False
    ) -> None:
        # Define attributes
        self.n_candidates: int = n_candidates
        self.local: bool = local_registry

        # Load registry
        self.local_registry: Dict[str, List[str]] = json.load(
            open(Params.local_registry_path)
        )

    def load_models(
        self,
        model_ids: str = None,
        stage: str = 'development'
    ) -> List[Model]:
        models: List[Model] = []
        if self.local:
            # Load from file system
            for model_id in model_ids:
                # Instanciate & load Model
                model = Model(model_id=model_id)
                model.load_from_file_system()

                # Append model
                models.append(model)
        else:
            # Query algorithm runs & sort by validation performance
            filter_string = f"tag.stage = '{stage}'"

            # Find metric
            if stage == 'development':
                metric = 'val_score'
            else:
                metric = 'test_score'

            runs_df: pd.DataFrame = mlflow.search_runs(
                experiment_names=[Params.experiment_name],
                filter_string=filter_string,
                order_by=[f'metrics.{metric}']
            )

            # Retrieve logged models and metadata
            for run_id in runs_df['run_id']:
                # Retrieve run
                run = Params.ml_client.get_run(run_id=run_id)

                # Instanciate Model
                model = Model(
                    artifact_uri=run.info.artifact_uri,
                    algorithm=run.data.tags.get('algorithm')
                )

                # Load model from registry
                if stage == 'development':
                    use_tracking_server = True
                else:
                    use_tracking_server = False

                model.load_from_registry(load_model_from_tracking_server=use_tracking_server)

                # Append model to dev_models
                models.append(model)
        return models

    @property
    def dev_models(self) -> List[Model]:
        # Load Dev Models
        dev_models: List[Model] = self.load_models(
            model_ids=self.local_registry['development'],
            stage='development'
        )

        # Sort dev models based on validation score
        dev_models.sort(key=lambda model: model.val_score, reverse=True)
        
        return dev_models
    
    @property
    def staging_models(self) -> List[Model]:
        # Load Staging Models
        stage_models: List[Model] = self.load_models(
            model_ids=self.local_registry['staging'],
            stage='staging'
        )

        # Sort dev models based on validation score
        stage_models.sort(key=lambda model: model.test_score, reverse=True)
        
        return stage_models
    
    @property
    def prod_model(self) -> Model:
        # Load Production Model (i.e.: champion Model)
        loaded_models = self.load_models(
            model_ids=self.local_registry['production'],
            stage='production'
        )

        if len(loaded_models) == 0:
            return None
        if len(loaded_models) > 1:
            print(f'[WARNING] There is more than one production model registered!')
            loaded_models.sort(key=lambda model: model.test_score, reverse=True)
            
        return loaded_models[0]
            
    def set_up_tracking_server(self):
        print('Setting up local tracking server.')
        # command = ['mlflow', 'ui', '--port', '5050']
        command = ['mlflow', 'server', '--host', '0.0.0.0', '--port', '5050']
        subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        
        # Whait 5 sec for the tracking server to set up
        time.sleep(5)

        # Set up tracking URI to local_gost
        mlflow.set_tracking_uri(Params.tracking_url)        
        print('Finished!')

    def register_models(self):
        # Register Champion
        self.prod_model.register_model()

        # Register Staging Models
        for model in self.staging_models:
            model.register_model()

    def update_model_stages(
        self,
        update_champion: bool = True
    ):
        # Degrade all models to development (except for champion)
        dev_models: List[Model] = self.dev_models + self.staging_models
        staging_models: List[Model] = []
        champion: Model = self.prod_model

        for model in dev_models:
            # Re-asign stage
            model.stage = 'development'

        # Sort Dev Models (based on validation performance)
        dev_models.sort(key=lambda model: model.val_score, reverse=True)

        # Find top n candidates
        staging_candidates = dev_models[: self.n_candidates]

        for model in staging_candidates:
            # Test model
            diagnostics_dict = model.diagnose_model()
            if diagnostics_dict['passed']:
                # Promote Model
                model.stage = 'staging'

                # Add model to staging_models
                staging_models.append(model)

                # Remove model from dev_models
                dev_models.remove(model)
            else:
                print(f'[WARNING] {model.model_id} was NOT pushed to Staging.\n'
                      f'diagnostics_dict:')
                pprint(diagnostics_dict)
                print('\n\n')

        # Sort Dev Models (based on test performance)
        staging_models.sort(key=lambda model: model.test_score, reverse=True)
        
        #  Update Champion
        if champion is None:
            print(f'[WARNING] There was no previous champion.\n'
                  f'Therefore, a new provisory champion will be chosen.\n\n')
            
            # Promote New Champion
            new_champion = staging_models[0]
            new_champion.stage = 'production'

            # Remove model from staging_models
            staging_models.remove(new_champion)

            # Save new_champion
            new_champion.save()

            # Define New Champion
            champion = new_champion

            print(f'New champion model:')
            print(champion)
            print('\n\n')

        elif update_champion:
            # Pick Challenger
            challenger = staging_models[0]

            if challenger.test_score > champion.test_score:
                print(f'New Champion mas found (test performance: {challenger.test_score}):')
                print(challenger)
                print(f'Previous Champion (test performance: {champion.test_score}):')
                print(champion)

                # Promote Challenger
                challenger.stage = 'production'

                # Demote Champion
                champion.stage = 'staging'

                # Remove Challenger from staging_models & append Champion
                staging_models.remove(challenger)
                staging_models.append(champion)

                # Save challenger
                challenger.save()

                # Define New Champion
                champion = challenger

                print(f'New champion model:')
                print(champion)
                print('\n\n')

        # Save Dev Models
        self.local_registry['development'] = [m.model_id for m in dev_models]
        for model in dev_models:
            # Assert model stage
            assert model.stage == 'development'

            # Save model to file system
            model.save_to_file_system()

            # Log model to tracking server
            model.log_model()

        # Save Staging Models
        self.local_registry['staging'] = [m.model_id for m in staging_models]
        for model in staging_models:
            # Assert model stage
            assert model.stage == 'staging'

            # Save model to file system
            model.save_to_file_system()

            # Register model
            model.register_model()

        # Save Champion
        self.local_registry['production'] = [champion.model_id]
        # Assert champion stage
        assert champion.stage == 'production'

        # Save champion
        champion.save()
        
        # Save local_registry
        self.save_local_registry()
    
    def clean_registry(self) -> None:
        # Clean file_system
        self.clean_file_system()

        # Clean tracking server
        self.clean_tracking_server()

        # Clean mlflow registry
        self.clean_mlflow_registry()

    def clean_file_system(self) -> None:
        # Retrieve Models
        models = (
            self.dev_models +
            self.staging_models + 
            [self.prod_model]
        )

        # Clean Models
        keep_ids = [model.model_id for model in models]

        for root, directories, files in os.walk(Params.model_attr_path):
            for file in files:
                model_id = file.split('_')[0]
                if model_id not in keep_ids:
                    delete_path = os.path.join(Params.model_attr_path, file)
                    print(f"Deleting {delete_path}.")
                    os.remove(delete_path)

        # Clean artifacts
        keep_run_ids = [m.run_id for m in models]

        for root, directories, files in os.walk(Params.artifacts_path):
            for file in files:
                artifact_run_id = file.split('_')[0]
                if artifact_run_id not in keep_run_ids:
                    delete_path = os.path.join(Params.artifacts_path, file)
                    print(f"Deleting {delete_path}.")
                    os.remove(delete_path)

        # Clean mlruns
        keep_run_ids = [m.run_id for m in models]

        subdirs = []
        for root, directories, files in os.walk(os.path.join('mlruns', str(Params.experiment_id))):
            splits = root.split('/')
            if len(splits) >= 3:
                subdirs.append(os.path.join(*splits[:3]))
        subdirs = list(set(subdirs))

        for subdir in subdirs:
            if subdir.split('/')[-1] not in keep_run_ids:
                print(f'Deleting {subdir}.')
                shutil.rmtree(subdir)

    def clean_tracking_server(self) -> None:
        # Retrieve Models
        models = (
            self.dev_models +
            self.staging_models + 
            [self.prod_model]
        )

        # Find keep runs
        keep_run_ids = [m.run_id for m in models]

        # Clean runs
        runs_df: pd.DataFrame = mlflow.search_runs(
            experiment_names=[Params.experiment_name]
        )

        # Retrieve logged models and metadata
        for run_id in runs_df['run_id']:
            if run_id not in keep_run_ids:
                print(f'deleating {run_id} run.')
                mlflow.delete_run(run_id=run_id)

    def clean_mlflow_registry(self) -> None:
        # Retrieve Models
        models = (
            self.dev_models +
            self.staging_models + 
            [self.prod_model]
        )

        # Find keep model_names
        keep_names = [m.model_name for m in models]

        # Delete registerd model
        reg_model_names = [reg.name for reg in mlflow.search_registered_models()]
        for model_name in reg_model_names:
            if model_name not in keep_names:
                print(f'deleting {model_name} from mlflow registry.')
                Params.ml_client.delete_registered_model(name=model_name)

    def save_local_registry(self) -> None:
        with open(Params.local_registry_path, "w") as f:
            json.dump(self.local_registry, f, indent=4)

    def __repr__(self) -> str:
        print('\nModel Registry:')
        
        # Prod Model
        champion = self.prod_model
        print(f'Production Model: {champion.model_id} [Test: {round(champion.test_score, 4)}, Validation: {round(champion.val_score, 4)}]')

        # Staging Models
        for model in self.staging_models:
            print(f'Staging Model: {model.model_id} [Test: {round(model.test_score, 4)}, Validation: {round(model.val_score, 4)}]')
        
        # Dev Models
        for model in self.dev_models:
            print(f'Dev Model: {model.model_id} [Test: {round(model.test_score, 4)}, Validation: {round(model.val_score, 4)}]')

        return '\n\n'
