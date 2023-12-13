from config.params import Params
import pandas as pd
import os


def load_raw_data(
    user_uuids: list = None,
    course_uuids: list = None
):
    # Load Dataset
    df = pd.read_csv(Params.raw_data_path, sep=';')

    # Filter Dataset
    if user_uuids is not None:
        df = df.loc[df['user_uuid'].isin(user_uuids)]

    if course_uuids is not None:
        df = df.loc[df['course_uuid'].isin(course_uuids)]

    return df