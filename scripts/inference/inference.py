from config.params import Params
import pandas as pd
import numpy as np
import requests
import json
import random
import os
from pprint import pprint


def main(new_request_df: pd.DataFrame):
    # Replace nan values with dummy values
    new_request_df.replace(np.nan, 'nan', inplace=True)

    # Send request
    prediction = requests.post(
        Params.request_url, 
        json=new_request_df.to_dict()
    ).json()

    print(f'New inference:')
    pprint(prediction)
    print('\n\n')

    # Save prediction
    user_uuid = new_request_df['user_uuid'].unique()[0]
    course_uuid = new_request_df['course_uuid'].unique()[0]
    particion = new_request_df['particion'].unique()[0]

    with open(os.path.join(Params.inference_path, f"{user_uuid}_{course_uuid}_{particion}_inference.json"), "w") as f:
        json.dump(prediction, f, indent=4)

def pick_random_sample(
    raw_df: pd.DataFrame,
    user_uuid: str = None,
    course_uuid: str = None,
    particion: int = None
):
    # Filter random user_uuid
    if user_uuid is None:
        user_uuid = random.choice(raw_df['user_uuid'].unique())
    print(f'Chosen user_uuid: {user_uuid}')
    raw_df = raw_df.loc[raw_df['user_uuid'] == user_uuid]

    # Filter random course_uuid
    if course_uuid is None:
        course_uuid = random.choice(raw_df['course_uuid'].unique())
    print(f'Chosen course_uuid: {course_uuid}')
    raw_df = raw_df.loc[raw_df['course_uuid'] == course_uuid]

    # Filter random partition
    if particion is None:
        particion = random.choice([p for p in raw_df['particion'].unique() if p > 10])
    print(f'Chosen particion: {particion}')
    raw_df = raw_df.loc[raw_df['particion'] <= particion]

    return raw_df


# .venv/bin/python scripts/inference/inference.py
if __name__ == '__main__':
    # Load raw data
    raw_df = pd.read_csv(Params.raw_data_path, sep=';')

    # Pick random user_uuid, course_uuid & partition
    new_request_df = pick_random_sample(
        raw_df=raw_df,
        user_uuid=None, # 'bc281b7f-8c99-40c8-96ba-458b2140953c',
        course_uuid=None, # '14100057-7f38-4776-a037-279e4f58b729',
        particion=None # 44
    )

    # Run new inference
    main(new_request_df=new_request_df)