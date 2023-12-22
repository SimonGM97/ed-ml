from config.params import Params
from ed_ml.pipeline.pipeline import MLPipeline
from ed_ml.modeling.model import Model
from ed_ml.modeling.model_registry import ModelRegistry

from scipy.stats import chisquare
from unittest import TestCase
from unittest.mock import patch
import pytest
import pandas as pd
import json
import os
from pprint import pprint
import warnings

warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)


class TestMLPipeline(TestCase):

    def test__ml_pipeline__success_datasets_are_correctly_balanced(
        self,
        debug: bool = False
    ):
        """
        Testing method that will verify that:
            - Train datasets created by the MLPipeline are balanced (i.e.: has approximately 50% observations of 
              each group).
            - Test datasets contain the same unbalanced group distribution as the overall (train & test) datasets.
        """
        # Load mock input data
        mock_input_path = os.path.join(
            'data_lake', 'datasets', 'ml_data', 'ml_data.csv'
        )
        ml_df = pd.read_csv(mock_input_path, index_col=0)

        # Instanciate MLPipeline with mock input
        pipeline = MLPipeline()

        # Prepare ML datasets
        pipeline.prepare_datasets(
            ml_df=ml_df,
            train_test_ratio=Params.train_test_ratio,
            balance_train=Params.balance_train,
            balance_method=Params.balance_method,
            debug=False
        )

        # Train balance
        train_balance = (
            pipeline.y_train
            .groupby(Params.target_column)
            [Params.target_column]
            .count()
            / pipeline.y_train.shape[0]
        ).values.tolist()

        if debug:
            print(f'train_balance: {train_balance}\n')

        # Assert train balance is roughly 50% of each category
        self.assertAlmostEqual(train_balance[0], 0.5, delta=0.001)

        # Overall Balance
        y = ml_df.filter(items=[Params.target_column])

        expected_test_balance = (
            y.groupby(Params.target_column)
            [Params.target_column]
            .count()
            / y.shape[0]
        ).values.tolist()

        if debug:
            print(f'expected_test_balance: {expected_test_balance}\n')

        # Test balance
        observed_test_balance = (
            pipeline.y_test
            .groupby(Params.target_column)
            [Params.target_column]
            .count()
            / pipeline.y_test.shape[0]
        ).values.tolist()

        if debug:
            print(f'observed_test_balance: {observed_test_balance}\n')

        # Assert test balance is roughly the same as the overall balance
        self.assertAlmostEqual(observed_test_balance[0], expected_test_balance[0], delta=0.001)

    def test__ml_pipeline__inferences_are_consistent(
        self,
        debug: bool = False
    ):
        """
        Method that will validate that new inferences being made to a mocked raw df are consistent with 
        the expected inference.
        """
        # Load mock input data
        mock_input_path = os.path.join(
            'data_lake', 'datasets', 'mock_data', 'data_processing_input.csv'
        )
        mock_raw_df = pd.read_csv(mock_input_path, index_col=0)

        # Load champion Model
        ml_registry = ModelRegistry(
            load_from_local_registry=Params.local_registry
        )
        
        champion = ml_registry.prod_model

        # Define Pipeline
        ml_pipeline = MLPipeline()

        # Run new prediction
        prediction = ml_pipeline.inference_pipeline(
            model=champion,
            raw_df=mock_raw_df
        )
        
        if debug:
            print('prediction:')
            pprint(prediction)
            print('\n\n')

        # Load expected output
        mock_output_path = os.path.join(
            'data_lake', 'datasets', 'mock_data', 'expected_inference.json'
        )

        expected_prediction = json.load(open(mock_output_path))

        # Assert dicts are equal
        self.assertDictEqual(
            d1=prediction,
            d2=expected_prediction
        )


# python3 -m unittest test/test_ml_pipeline.py
if __name__ == '__main__':
    TMLP = TestMLPipeline()
    TMLP.test__ml_pipeline__success_datasets_are_correctly_balanced(debug=True)
    TMLP.test__ml_pipeline__inferences_are_consistent(debug=True)