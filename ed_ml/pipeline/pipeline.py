from config.params import Params
from ed_ml.modeling.model import Model
from ed_ml.data_processing.data_cleaning import DataCleaner
from ed_ml.data_processing.feature_engineering import FeatureEngineer

from imblearn.over_sampling import RandomOverSampler
from sklearn.model_selection import train_test_split

import pandas as pd
import numpy as np
import warnings

warnings.filterwarnings("ignore")


class MLPipeline:

    def __init__(self) -> None:
        # Datasets
        self.X_train: pd.DataFrame = None
        self.X_test: pd.DataFrame = None

        self.y_train: pd.DataFrame = None
        self.y_test: pd.DataFrame = None

    def prepare_datasets(
        self,
        ml_df: pd.DataFrame,
        train_test_ratio: float,
        debug: bool = False
    ) -> None:
        # Keep numerical features
        ml_df = ml_df.select_dtypes(include=['number'])

        # Divide ml_df into y & X datasets
        X = ml_df.drop([Params.target_column], axis=1)
        y = ml_df.filter(items=[Params.target_column])

        if debug:
            print(f"overall balance: \n{y.groupby(Params.target_column)[Params.target_column].count() / y.shape[0]}\n")

        # Divide un-balanced datasets into train & test
        #   - Note: stratify=y ensures that the splitting process maintains the same class 
        #     distribution in both the original dataset and the resulting training and testing sets.
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=train_test_ratio, random_state=23111997, stratify=y
        )

        # Balance Train Dataset
        RO = RandomOverSampler(random_state=0)
        self.X_train, self.y_train = RO.fit_resample(self.X_train, self.y_train)

        if debug:
            print(f"train balance: \n{self.y_train.groupby(Params.target_column)[Params.target_column].count() / self.y_train.shape[0]}\n"
                  f"test balance: \n{self.y_test.groupby(Params.target_column)[Params.target_column].count() / self.y_test.shape[0]}\n\n")        

    def build_pipeline(
        self,
        ml_params: dict,
        eval_metric: str,
        splits: int,
        debug: bool = False
    ) -> Model:
        # Instanciate Model
        model = Model(**ml_params)

        # Build Model
        model.build(debug=debug)

        # Evaluate Model utilizing cross validation
        model.evaluate_val(
            X_train=self.X_train,
            y_train=self.y_train,
            eval_metric=eval_metric,
            splits=splits,
            debug=debug
        )

        # Fit Model on entire train datasets
        model.fit(
            y_train=self.y_train,
            X_train=self.X_train
        )

        return model

    def inference_pipeline(
        self,
        model: Model,
        raw_df: pd.DataFrame
    ):
        # Prepare raw_df
        if Params.target_column not in raw_df.columns:
            raw_df[Params.target_column] = np.nan

        # Clean Raw Data
        DC = DataCleaner(df=raw_df)
        clean_df = DC.cleaner_pipeline()
        
        # Prepare ML Datasets
        FE = FeatureEngineer(df=clean_df)
        ml_df = FE.data_enricher_pipeline()

        ml_df: pd.DataFrame = (
            ml_df
            .groupby(['user_uuid', 'course_uuid'])
            .apply(lambda df: df.loc[df['particion'].idxmax()])
        )

        # Keep numerical features
        ml_df = ml_df.select_dtypes(include=['number'])
        
        # Extract X
        X = ml_df.drop([Params.target_column], axis=1)

        # Predict y_proba
        y_proba = model.predict_proba(X=X)[:, 1]
        
        # Create inference output
        ml_df['predicted_probability'] = y_proba

        ml_df = (
            ml_df
            .filter(items=['predicted_probability'])
            .droplevel('course_uuid')
            .to_dict()
        )

        return ml_df

    def evaluate_pipeline(
        self,
        model: Model
    ):
        pass
