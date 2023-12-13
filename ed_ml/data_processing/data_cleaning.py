from config.params import Params
import pandas as pd
import numpy as np
import os


class DataCleaner:

    def __init__(
        self, 
        df: pd.DataFrame = None,
        load_dataset: bool = False
    ) -> None:
        # Define self.df
        self.df: pd.DataFrame = df

        # Load Dataset
        if load_dataset:
            self.load_dataset()

    def cleaner_pipeline(
        self,
        save: bool = False
    ) -> pd.DataFrame:
        # Define Target
        self.df['target'] = np.where(
            self.df['nota_final_materia'] >= 4, 1, 0
        )

        # Drop unrequired features
        self.df.drop(columns=Params.ignore_features, inplace=True)

        # Transform datetime columns
        self.df[Params.datetime_columns] = (
            self.df[Params.datetime_columns]
            .fillna(0)
            .astype(int)
            .apply(pd.to_datetime, unit='s')
        )

        # Se asume que:
        #   - Si para un mismo alumno, curso y assignment hay más de una observación, entonces se trata de un 
        #     trabajo con distintas entregas parciales, en donde la nota resultante está dado por el promedio 
        #     de las entregas parciales.

        def find_agg_fun(col: str):
            if col in list(self.df.select_dtypes(include=['object', 'datetime']).columns):
                return 'first'
            return 'mean'
        
        subset_cols = ['user_uuid', 'course_uuid', 'ass_name_sub']
        mask = (
            (self.df['ass_name_sub'].notna()) &
            (self.df.duplicated(subset=subset_cols, keep='first') | self.df.duplicated(subset=subset_cols, keep='last'))
        )
        
        corrected_obs = (
            self.df.loc[mask]
            .groupby(subset_cols)
            .agg({col: find_agg_fun(col) for col in self.df.columns})
            .reset_index(drop=True)
        )
        self.df = self.df.loc[~mask]
        self.df = pd.concat([self.df, corrected_obs])

        # Sort by partition
        self.df.sort_values(by=['user_uuid', 'course_uuid', 'particion'], inplace=True)

        # Save Dataset
        if save:
            self.save_dataset()
        
        return self.df
    
    def save_dataset(self):
        self.df.to_csv(Params.cleaned_data_path)

    def load_dataset(self):
        self.df = pd.read_csv(Params.cleaned_data_path)
    
