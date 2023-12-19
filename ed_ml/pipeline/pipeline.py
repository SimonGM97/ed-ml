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
    """
    Class designed to standardize the main modelling processes:
        - Preparing machine learning datasets.
        - Running a model building pipeline.
        - Running an inference pipeline.
        - Running an model updating pipeline.
    """

    def __init__(self) -> None:
        """
        Instanciate MLPipeline
        """
        # Train features
        self.X_train: pd.DataFrame = None
        
        # Test features
        self.X_test: pd.DataFrame = None

        # Train target (balanced)
        self.y_train: pd.DataFrame = None

        # Test target (unbalanced)
        self.y_test: pd.DataFrame = None

    def prepare_datasets(
        self,
        ml_df: pd.DataFrame,
        train_test_ratio: float,
        debug: bool = False
    ) -> None:
        """
        Method that will prepare machine learning datasets, by:
            - Randomly selecting train & test datasets.
            - Balancing the train datasets (X_train & y_train) with an oversampling technique

        Note that the test datasets (X_test & y_test) are purposely kept unbalanced to more accurately
        depict the real life group proportions; thus achieving a better estimate of the model performance
        in a production environment. 

        :param `ml_df`: (pd.DataFrame) Engineered DataFrame outputted by the FeatureEngineer class.
        :param `train_test_ratio`: (float) Proportion of data to keep as the test set; relative to the complete
         dataset.
        :param `debug`: (bool) Wether or not to show dataset balances for debugging purposes.
        """
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
        """
        Method that will run the model building pipeline, by:
            - Instanciateing the model.
            - Evaluating the cross validation score over the train datasets.
            - Re-fit the model with the complete train datasets.

        :param `ml_params`: (dict) Parameters required when instanciating the Model class.
        :param `eval_metric`: (str) Name of the metric utilized to evaluate ML model over the validation set.
        :param `splits`: (int) Number of splits utilized in for cross validation of ML model.
        :param `debug`: (bool) Wether or not to show intermediate results, for debugging purposes.

        :return: (Model) Fitted Model instance.
        """
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
    ) -> dict:
        """
        Method that will run the inference pipeline, by:
            - Cleaning new raw datasets.
            - Calculating engineered datasets.
            - Extracting a new X dataset.
            - Performing new probabilistic predictions.

        :param `model`: (Model) Instance from class Model utilized to infer new predictions.
        :param `raw_df`: (pd.DataFrame) New raw observations to make inferences on.
        """
        # Prepare raw_df
        if Params.target_column not in raw_df.columns:
            raw_df[Params.target_column] = np.nan

        # Clean Raw Data
        DC = DataCleaner(df=raw_df)
        clean_df = DC.run_cleaner_pipeline()
        
        # Prepare ML Datasets
        FE = FeatureEngineer(df=clean_df)
        ml_df = FE.run_feature_engineering_pipeline()

        # Keep only the latest partition for each new inference
        ml_df: pd.DataFrame = (
            ml_df
            .groupby(['user_uuid', 'course_uuid'])
            .apply(lambda df: df.loc[df['particion'].idxmax()])
            .select_dtypes(include=['number'])
        )
        
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

    def updating_pipeline(
        self,
        model: Model,
        eval_metric: str,
        refit_model: bool = False,
        find_new_shap_values: bool = False
    ) -> Model:
        """
        Method that will run the model updating pipeline, by:
            - (Optionally) Re-fitting model on new train datasets.
            - Evaluating model on test set & update test performance scores.
            - Calculating feature importance.
        
        :param `model`: (Model) Instance from class Model that will be updated.
        :param `eval_metric`: (str) Name of the metric utilized to evaluate ML model over the test set.
        :param `refit_model`: (bool) Wether or not to re-fit the inputed model with train datasets.
        :param `find_new_shap_values`: (bool) Wether or not to calculate new shaply values.
        """
        # Refit ml model (if specified)
        if refit_model: 
            model.fit(
                y_train=self.y_train,
                X_train=self.X_train
            )
        
        # Evaluate test performance
        model.evaluate_test(
            X_test=self.X_test,
            y_test=self.y_test,
            eval_metric=eval_metric
        )

        # Find Model feature importance
        model.find_feature_importance(
            X_test=self.X_test,
            find_new_shap_values=find_new_shap_values
        )

        return model
