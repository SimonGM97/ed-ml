#!/usr/bin/env python3
from config.params import Params
from ed_ml.utils.load_data import load_raw_data
from ed_ml.utils.add_parser_arguments import add_parser_arguments
import pandas as pd
import numpy as np
import requests
import json
import argparse
import os
from pprint import pprint


def main(raw_df: pd.DataFrame):
    # Replace nan values with dummy values
    raw_df.replace(np.nan, 'nan', inplace=True)

    # Send request
    prediction = requests.post(
        Params.request_url, 
        json=raw_df.to_dict()
    ).json()

    print(f'New inference:')
    pprint(prediction)
    print('\n\n')

    # Save prediction
    user_uuid = raw_df['user_uuid'].unique()[0]
    course_uuid = raw_df['course_uuid'].unique()[0]
    particion = raw_df['particion'].unique()[0]

    with open(os.path.join(Params.inference_path, f"{user_uuid}_{course_uuid}_{particion}_inference.json"), "w") as f:
        json.dump(prediction, f, indent=4)


# .venv/bin/python scripts/inference/inference.py --course_name User-friendly intangible flexibility
if __name__ == '__main__':
    # Define parser
    parser = argparse.ArgumentParser(description='Data processing script.')

    # Add expected arguments
    parser = add_parser_arguments(parser=parser)

    # Extract arguments
    args = parser.parse_args()

    # Load raw data
    raw_df = load_raw_data(
        course_name=args.course_name, # 'User-friendly intangible flexibility'
        user_uuids=args.user_uuids, # ['bc281b7f-8c99-40c8-96ba-458b2140953c']
        course_uuids=args.course_uuids, # ['14100057-7f38-4776-a037-279e4f58b729']
        particion=args.particion, # 44
        pick_random=args.pick_random
    )

    # Run new inference
    main(raw_df=raw_df)