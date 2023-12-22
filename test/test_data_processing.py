from ed_ml.data_processing.data_cleaning import DataCleaner
from ed_ml.data_processing.feature_engineering import FeatureEngineer

from unittest import TestCase
from unittest.mock import patch
import pytest
from pandas.testing import assert_frame_equal
import pandas as pd
import os


class TestDataProcessing(TestCase):

    def test__data_processor__success_raw_data_transformed_as_expected(self):
        """
        Testing method that will validate that a mocked raw dataset is transformed as expected.
        """
        # Load mock input data
        mock_input_path = os.path.join(
            'data_lake', 'datasets', 'mock_data', 'data_processing_input.csv'
        )
        mock_raw_df = pd.read_csv(mock_input_path, index_col=0)

        # Instanciate DataCleaner with mock input
        DC = DataCleaner(df=mock_raw_df)
        
        # Run cleaner_pipeline process with mocked input
        mock_clean_df = DC.run_cleaner_pipeline()

        # Instanciate FeatureEngineer with mock_cleaned_df
        FE = FeatureEngineer(df=mock_clean_df)

        # Run cleaner_pipeline process with mocked input
        mock_ml_df = FE.run_feature_engineering_pipeline()

        # Load expected output
        mock_output_path = os.path.join(
            'data_lake', 'datasets', 'mock_data', 'data_processing_expected_output.csv'
        )
        expected_ml_df = pd.read_csv(mock_output_path, index_col=0)

        # Assert outputs are equivalent
        assert_frame_equal(
            left=mock_ml_df.reset_index(drop=True),
            right=expected_ml_df.reset_index(drop=True)
        )


# python3 -m unittest test/test_data_processing.py
if __name__ == '__main__':
    TDP = TestDataProcessing()
    TDP.test__data_processor__success_raw_data_transformed_as_expected()