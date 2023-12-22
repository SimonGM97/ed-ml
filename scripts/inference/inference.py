#!/usr/bin/env python3
from config.params import Params
from ed_ml.utils.load_data import load_raw_data
from ed_ml.utils.add_parser_arguments import add_parser_arguments
import pandas as pd
import numpy as np
from datetime import datetime
import requests
import json
import argparse
import os
from pprint import pprint


# def main(raw_df: pd.DataFrame):
def main(
    course_name: str = None,
    user_uuids: list = None,
    particion: int = None,
    pick_random: bool = False
):
    # Load raw data
    raw_df = load_raw_data(
        course_name=course_name,
        user_uuids=user_uuids,
        particion=particion,
        pick_random=pick_random
    )

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

    # Validate course_name
    if course_name is None:
        if raw_df['course_name'].nunique() > 1:
            course_name = 'all'
        else:
            course_name = raw_df['course_name'].unique()[0]
    
    # Validate particion
    if particion is None:
        particion = str(raw_df['particion'].max())
    
    # Find save time
    now = str(datetime.now()).replace(' ', '_').replace(':', '-')

    # Create subdir
    subdir = os.path.join(Params.inference_path, course_name, particion)
    if not os.path.exists(subdir):
        os.makedirs(subdir)
    
    # Save prediction
    with open(os.path.join(Params.inference_path, course_name, particion, f"{now}_inference.json"), "w") as f:
        json.dump(prediction, f, indent=4)


# .venv/bin/python scripts/inference/inference.py --course_name "User-friendly intangible flexibility"
if __name__ == '__main__':
    # Define parser
    parser = argparse.ArgumentParser(description='Data processing script.')

    # Add expected arguments
    parser = add_parser_arguments(parser=parser)

    # Extract arguments
    args = parser.parse_args()

    # Run new inference
    main(
        course_name=args.course_name, # 'User-friendly intangible flexibility'
        user_uuids=args.user_uuids, # ['bc281b7f-8c99-40c8-96ba-458b2140953c']
        particion=args.particion, # 44
        pick_random=args.pick_random
    )
