from config.params import Params
import pandas as pd
import random


def filter_data(
    df: pd.DataFrame,
    course_name: str = None,
    user_uuids: list = None,
    course_uuids: list = None,
    particion: int = None,
    pick_random: bool = False
):
    # Filter based on course_name
    if pick_random:
        course_name = list(random.choice(df['course_name'].unique()))

    if course_name is not None:
        df = df.loc[df['course_name'] == course_name]

    # Filter based on user_uuid
    if pick_random:
        user_uuids = list(random.choice(df['user_uuid'].unique()))

    if user_uuids is not None:
        df = df.loc[df['user_uuid'].isin(user_uuids)]

    # Filter based on course_uuids
    if pick_random:
        course_uuids = list(random.choice(df['course_uuid'].unique()))

    if course_uuids is not None:
        df = df.loc[df['course_uuid'].isin(course_uuids)]

    # Filter based on partition
    if pick_random:
        particion = random.choice([p for p in df['particion'].unique() if p > 10])

    if particion is not None:
        df = df.loc[df['particion'] <= particion]
    
    return df


def load_raw_data(
    course_name: str = None,
    user_uuids: list = None,
    course_uuids: list = None,
    particion: int = None,
    pick_random: bool = False
):
    # Load Dataset
    df = pd.read_csv(Params.raw_data_path, sep=';')

    # Filter DataFrame
    df = filter_data(
        df=df,
        course_name=course_name,
        user_uuids=user_uuids,
        course_uuids=course_uuids,
        particion=particion,
        pick_random=pick_random
    )

    return df


def load_clean_data(
    course_name: str = None,
    user_uuids: list = None,
    course_uuids: list = None,
    particion: int = None,
    pick_random: bool = False
):
    # Load Dataset
    df = pd.read_csv(Params.cleaned_data_path, sep=';')

    # Filter DataFrame
    df = filter_data(
        df=df,
        course_name=course_name,
        user_uuids=user_uuids,
        course_uuids=course_uuids,
        particion=particion,
        pick_random=pick_random
    )

    return df