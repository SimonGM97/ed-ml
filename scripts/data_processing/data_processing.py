#!/usr/bin/env python3
from ed_ml.utils.load_data import load_raw_data
from ed_ml.data_processing.data_cleaning import DataCleaner
from ed_ml.data_processing.feature_engineering import FeatureEngineer
from ed_ml.utils.add_parser_arguments import add_parser_arguments
import argparse
import pandas as pd


def main(
    raw_df: pd.DataFrame = None,
    save: bool = True
):
    print(f'Running data_processing.main with parameters:\n'
          f'    - raw_df: {raw_df} ({type(raw_df)})\n'
          f'    - save: {save} ({type(save)})\n\n')
    
    if raw_df is None:
        # Load Raw Data
        raw_df = load_raw_data(
            user_uuids=None,
            course_uuids=None
        )

    # Clean Raw Data
    DC = DataCleaner(df=raw_df.copy())
    clean_df = DC.run_cleaner_pipeline(save=save)

    # clean_df.loc[
    #     (clean_df['user_uuid'] == '13df535e-065c-4593-98ea-5b1e29015b7d') &
    #     (clean_df['course_uuid'] == '09614210-fce2-48bc-93e3-bc4bd441fe00')
    # ].to_excel('clean_df_test.xlsx')
    
    # Prepare ML Datasets
    FE = FeatureEngineer(df=clean_df.copy())
    ml_df = FE.run_feature_engineering_pipeline(save=save)

    # ml_df.loc[
    #     (ml_df['user_uuid'] == '13df535e-065c-4593-98ea-5b1e29015b7d') &
    #     (ml_df['course_uuid'] == '09614210-fce2-48bc-93e3-bc4bd441fe00')
    # ].to_excel('ml_df_test.xlsx')

    print(f'Finished data_processing.main.\n\n')

    return ml_df


# .venv/bin/python scripts/data_processing/data_processing.py --raw_df None --save True
if __name__ == '__main__':
    # Define parser
    parser = argparse.ArgumentParser(description='Data processing script.')

    # Add expected arguments
    parser = add_parser_arguments(parser=parser)

    # Extract arguments
    args = parser.parse_args()

    # Run main with parsed parguments
    main(raw_df=args.raw_df, save=args.save)

