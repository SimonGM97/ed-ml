from config.params import Params
import pandas as pd
import numpy as np
import warnings

warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)

pd.options.display.max_rows = 100
pd.set_option('display.max_columns', None)


class FeatureEngineer:
    """
    Class used to perform feature engineering processes & transformations required
    to generate ML datasets that will later be consumed by ML models.
    """

    def __init__(
        self,
        df: pd.DataFrame = None,
        load_dataset: bool = False
    ) -> None:
        """
        Initialize FeatureEngineer.

        :param df: (pd.DataFrame) Input DataFrame.
        :param load_dataset: (bool) Whether to load the dataset during initialization.
        """
        # Define self.df
        self.df: pd.DataFrame = df

        # Load Dataset
        if load_dataset:
            self.load_dataset()
    
    @staticmethod
    def add_target_column(
        df: pd.DataFrame
    ) -> pd.DataFrame:
        # Define target column
        df[Params.target_column] = np.where(
            df['nota_final_materia'] >= 4, 1, 0
        )

        return df

    @staticmethod
    def add_timing_based_features(
        df: pd.DataFrame
    ) -> pd.DataFrame:
        # Add total hours to do
        df['hrs_to_do_assignment'] = np.where(
            df['ass_name'].notna(),
            (df['ass_due_at'] - df['ass_unlock_at']).dt.total_seconds() / 3600,
            np.nan
        )

        df['hrs_to_do_assignment'].fillna(0, inplace=True)

        # Add time taken to do the assignment
        df['hrs_taken_to_do_assignment'] = np.where(
            df['ass_name'].notna(),
            (df['s_submitted_at'] - df['ass_unlock_at']).dt.total_seconds() / 3600,
            np.nan
        )

        df['hrs_taken_to_do_assignment'].fillna(0, inplace=True)

        # Add relative tima taken to do assignment
        df['rel_time_taken_to_do_assignment'] = np.where(
            np.logical_and(
                df['ass_name'].notna(),
                df['hrs_to_do_assignment'] > 0
            ),
            df['hrs_taken_to_do_assignment'] / df['hrs_to_do_assignment'],
            np.nan
        )

        df['rel_time_taken_to_do_assignment'].fillna(0, inplace=True)

        # Add time submitted before due date
        df['hrs_before_due'] = np.where(
            df['ass_name'].notna(),
            (df['ass_due_at'] - df['s_submitted_at']).dt.total_seconds() / 3600,
            np.nan
        )

        df['hrs_before_due'].fillna(0, inplace=True)

        return df

    @staticmethod
    def prepare_intermediate_features(
        df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Add features required for the expanding DataFrame.
        """
        # Add column that tells wether or not an examen parcial was taken
        df['parciales_n'] = np.where(
            np.logical_or(
                df['nombre_examen'] == 'PRIMER PARCIAL(20)',
                df['nombre_examen'] == 'SEGUNDO PARCIAL(20)'
            ), 1, 0
        )

        # Add column to reflect the scores for the examen parciales taken
        df['nota_parciales'] = np.where(
            np.logical_or(
                df['nombre_examen'] == 'PRIMER PARCIAL(20)',
                df['nombre_examen'] == 'SEGUNDO PARCIAL(20)'
            ),
            df['nota_parcial'],
            np.nan
        )

        # Add column that tells wether or not an examen integrador was taken
        df['integradores_n'] = np.where(
            df['nombre_examen'] == 'INTEGRADOR(30)',
            1, 0
        )

        # Add column to reflect the scores for the examen integradores taken
        df['nota_integradores'] = np.where(
            df['nombre_examen'] == 'INTEGRADOR(30)',
            df['nota_parcial'],
            np.nan
        )

        # Add column that tells wether or not an recuperatorio was taken
        df['recuperatorios_n'] = np.where(
            np.logical_or(
                df['nombre_examen'] == 'RECUPERATORIO PRIMER PARCIAL(20)',
                df['nombre_examen'] == 'RECUPERATORIO SEGUNDO PARCIAL(20)'
            ), 1, 0
        )

        # Add column to reflect the scores for the recuperatorios taken
        df['nota_recuperatorios'] = np.where(
            np.logical_or(
                df['nombre_examen'] == 'RECUPERATORIO PRIMER PARCIAL(20)',
                df['nombre_examen'] == 'RECUPERATORIO SEGUNDO PARCIAL(20)'
            ),
            df['nota_parcial'],
            np.nan
        )

        # Add column that tells wether or not a parcial was taken
        df['overall_parciales_n'] = np.where(
            df['nombre_examen'].notna(),
            1, 0
        )

        df.rename(columns={'nota_parcial': 'nota_overall'}, inplace=True)

        # Add column that tells wether or not an assignment was taken
        df['assignment_n'] = np.where(
            df['ass_name_sub'].notna(),
            1,0
        )

        # Add a column that reflect if an assignment got a 0
        df['assignment_zero'] = np.where(
            np.logical_and(
                df['ass_name_sub'].notna(),
                df['score'] <= 0
            ), 1,0
        )

        return df
    
    @staticmethod
    def calculate_expanding_features(
        df: pd.DataFrame
    ) -> None:
        def first(col: pd.Series):
            """
            Return the first value of a Series.

            :param col: (pd.Series) Input Series.

            :return: The first value of the Series.
            """
            return col.values[0]
        
        def rename_cols(col):
            ignore_rename = ['user_uuid', 'course_uuid', 'particion', 'target']
            if col[0] in ignore_rename:
                return col[0]
            return '_'.join(col)

        agg_funs = {
            'target': first,
            'particion': 'max',

            # Parciales related engineered features
            'parciales_n': 'sum',
            'nota_parciales': ['min', 'max', 'mean', 'std'],
            'integradores_n': 'sum',
            'nota_integradores': ['min', 'max', 'mean', 'std'],
            'recuperatorios_n': 'sum',
            'nota_recuperatorios': ['min', 'max', 'mean', 'std'],
            'overall_parciales_n': 'sum',
            'nota_overall': ['min', 'max', 'mean', 'std'],

            # Assignment related engineered features
            'assignment_n': 'sum',
            'score': ['min', 'max', 'mean', 'std'],
            'assignment_zero': 'sum',

            # Time related engineered features
            'hrs_to_do_assignment': ['min', 'max', 'mean', 'std'],
            'hrs_taken_to_do_assignment': ['min', 'max', 'mean', 'std'],
            'rel_time_taken_to_do_assignment': ['min', 'max', 'mean', 'std'],
            'hrs_before_due': ['min', 'max', 'mean', 'std']
        }
        
        # Run agg functions on expanding DataFrame
        df: pd.DataFrame = (
            df
            .groupby(['user_uuid', 'course_uuid'])
            .expanding()
            .agg(agg_funs)
            .reset_index()
            .drop(columns='level_2')
        )

        # Rename composed columns
        df.columns = [rename_cols(c) for c in df.columns]

        # Add column to show relationship between parciales and assignments
        df['parcial_sobre_ass'] = np.where(
            df['score_mean'] > 0,
            100 * df['nota_overall_mean'] / df['score_mean'],
            0
        )

        return df

    @staticmethod
    def remove_unuseful_obs(
        df: pd.DataFrame
    ) -> pd.DataFrame:
        # Fill values required to remove unnecessary rows
        df.fillna(value={
            'assignment_n_sum': 0, 
            'overall_parciales_n_sum': 0
        }, inplace=True)

        # Remove observations
        df = df.loc[
            (df['assignment_n_sum'] > 0) |
            (df['overall_parciales_n_sum'] > 0)
        ]

        return df

    def data_enricher_pipeline(
        self,
        save: bool = False
    ) -> pd.DataFrame:
        # Add target column
        self.df = self.add_target_column(df=self.df)

        # Add timing based features
        self.df = self.add_timing_based_features(df=self.df)

        # Prepare intermediate features
        self.df = self.prepare_intermediate_features(df=self.df)

        # Calculate expanding features
        self.df = self.calculate_expanding_features(df=self.df)

        # Remove unuseful observations
        self.df = self.remove_unuseful_obs(df=self.df)

        # Fill null values
        self.df.fillna(0, inplace=True)

        # Save Dataset
        if save:
            self.save_dataset()

        return self.df
    
    def save_dataset(self):
        print('\nSaving new engineered DataFrame.\n\n')
        self.df.to_csv(Params.ml_data_path)

    def load_dataset(self):
        self.df = pd.read_csv(Params.ml_data_path, index_col=0)
