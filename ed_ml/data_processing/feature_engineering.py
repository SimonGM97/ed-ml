from config.params import Params
import pandas as pd
import numpy as np
import os

pd.options.display.max_rows = 100
pd.set_option('display.max_columns', None)


class FeatureEngineer:

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
    
    @staticmethod
    def first(col: pd.Series):
        return col.values[0]

    def prepare_intermediate_features(self) -> None:
        # Prepare parciales features
        self.df['parciales_n'] = np.where(
            np.logical_or(
                self.df['nombre_examen'] == 'PRIMER PARCIAL(20)',
                self.df['nombre_examen'] == 'SEGUNDO PARCIAL(20)'
            ), 1, 0
        )

        self.df['nota_parciales'] = np.where(
            np.logical_or(
                self.df['nombre_examen'] == 'PRIMER PARCIAL(20)',
                self.df['nombre_examen'] == 'SEGUNDO PARCIAL(20)'
            ),
            self.df['nota_parcial'],
            np.nan
        )

        # Prepare integradores features
        self.df['integradores_n'] = np.where(
            self.df['nombre_examen'] == 'INTEGRADOR(30)',
            1, 0
        )

        self.df['nota_integradores'] = np.where(
            self.df['nombre_examen'] == 'INTEGRADOR(30)',
            self.df['nota_parcial'],
            np.nan
        )

        # Prepare recuperatorios features
        self.df['recuperatorios_n'] = np.where(
            np.logical_or(
                self.df['nombre_examen'] == 'RECUPERATORIO PRIMER PARCIAL(20)',
                self.df['nombre_examen'] == 'RECUPERATORIO SEGUNDO PARCIAL(20)'
            ), 1, 0
        )

        self.df['nota_recuperatorios'] = np.where(
            np.logical_or(
                self.df['nombre_examen'] == 'RECUPERATORIO PRIMER PARCIAL(20)',
                self.df['nombre_examen'] == 'RECUPERATORIO SEGUNDO PARCIAL(20)'
            ),
            self.df['nota_parcial'],
            np.nan
        )

        # Prepare overall parciales features
        self.df['overall_parciales_n'] = np.where(
            self.df['nombre_examen'].notna(),
            1, 0
        )

        self.df.rename(columns={'nota_parcial': 'nota_overall'}, inplace=True)

        # Prepare assignment features
        self.df['assignment_n'] = np.where(
            self.df['ass_name_sub'].notna(),
            1,0
        )

    @staticmethod
    def rename_cols(c):
        ignore_rename = ['user_uuid', 'course_uuid', 'particion', 'target']
        if c[0] in ignore_rename:
            return c[0]
        return '_'.join(c)

    def calculate_expanding_features(self) -> None:
        # Apply Transformations
        agg_funs = {
            'target': self.first,
            'parciales_n': 'sum',
            'nota_parciales': ['min', 'max', 'mean', 'std'],
            'integradores_n': 'sum',
            'nota_integradores': ['min', 'max', 'mean', 'std'],
            'recuperatorios_n': 'sum',
            'nota_recuperatorios': ['min', 'max', 'mean', 'std'],
            'overall_parciales_n': 'sum',
            'nota_overall': ['min', 'max', 'mean', 'std'],
            'assignment_n': 'sum',
            'score': ['min', 'max', 'mean', 'std'],
            'particion': 'max'
        }
        
        self.df: pd.DataFrame = (
            self.df
            .groupby(['user_uuid', 'course_uuid'])
            .expanding()
            .agg(agg_funs)
            .reset_index()
            .drop(columns='level_2')
        )

        self.df.columns = [self.rename_cols(c) for c in self.df.columns]

        self.df['parcial_sobre_ass'] = np.where(
            self.df['score_mean'] > 0,
            100 * self.df['nota_overall_mean'] / self.df['score_mean'],
            0
        )

    def data_enricher_pipeline(
        self,
        save: bool = False
    ) -> pd.DataFrame:
        # Prepare intermediate features
        self.prepare_intermediate_features()

        # print(self.df.select_dtypes(include=['datetime']).iloc[10:20])

        # Calculate expanding features
        self.calculate_expanding_features()

        # TODO: agregar features:
        #   - tiempo promedio que tardó en entregar una tarea (los más aplicados entregan rápido)

        # Remove unuseful observations
        self.df.fillna(value={
            'assignment_n_sum': 0, 
            'overall_parciales_n_sum': 0
        }, inplace=True)

        self.df = self.df.loc[
            (self.df['assignment_n_sum'] > 0) |
            (self.df['overall_parciales_n_sum'] > 0)
        ]

        # Fill null values
        self.df.fillna(0, inplace=True)

        # Save Dataset
        if save:
            self.save_dataset()

        return self.df
    
    def save_dataset(self):
        self.df.to_csv(Params.ml_data_path)

    def load_dataset(self):
        self.df = pd.read_csv(Params.ml_data_path)
        self.df.drop(columns=[c for c in self.df.columns if c.startswith('Unnamed:')], inplace=True)
