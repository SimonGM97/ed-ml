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

        :param `df`: (pd.DataFrame) Input DataFrame.
        :param `load_dataset`: (bool) Whether to load the dataset during initialization.
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
        """
        Method that will produce the target variable by assigning a 1 if the final grade is greater or equal to 4,
        and a 0 if that is not the case.

        :param `df`: (pd.DataFrame) DataFrame without target column.

        :return: (pd.DataFrame) DataFrame with target column.
        """
        # Define target column
        df[Params.target_column] = np.where(
            df['nota_final_materia'] >= 4, 1, 0
        )

        return df

    @staticmethod
    def add_timing_based_features(
        df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Method that will create features realted to the time at which the assignments were unlocked, submitted
        and due at. The intent is to model & distinguish lazy students that usually prepare and submit assignments
        near the deadline; compared to dedicated students that usually submit assignments with time to spare.

        Features being created:
            - hrs_to_do_assignment: total hours assigned to do an assignment (from unlock date until due date)
            - hrs_taken_to_do_assignment: number of hours spent by the student to do the assignment (from unlock 
              date until submittion date)
            - rel_time_taken_to_do_assignment: the number of hours spent working on the exam, relative to the total
              hours available to do the exam.
            - hrs_before_due: number of spare hours that the student had when submitting the assignment (from 
              submission date until due date).

        Note that this method requires the transformation DataCleaner.allign_assignments() to have been ran.

        :param `df`: (pd.DataFrame) DataFrame without timing based features.

        :return: (pd.DataFrame) DataFrame with timing based features.
        """
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
        Method that will calculate features required for a more seamingless calculation of cumulative features
        that will be generated in the self.calculate_expanding_features() method.
        
        Features created:
            - parciales_n: column containing a 1 if the student has taken a 'PRIMER PARCIAL(20)' or 
              'SEGUNDO PARCIAL(20)' in that partition.
            - nota_parciales: column containing the grade of the 'PRIMER PARCIAL(20)' or 'SEGUNDO PARCIAL(20)'
              that was obtained in that partition.
            - integradores_n: column containing a 1 if the student has taken a 'INTEGRADOR(30))' exam in that 
              partition.
            - nota_integradores: column containing the grade of the 'INTEGRADOR(30)' that was obtained in that 
              partition.
            - recuperatorios_n: column containing a 1 if the student has taken a 'RECUPERATORIO PRIMER PARCIAL(20)' 
              or 'RECUPERATORIO SEGUNDO PARCIAL(20)' in that partition.
            - nota_recuperatiorios: column containing the grade of the 'RECUPERATORIO PRIMER PARCIAL(20)' or 
              'RECUPERATORIO SEGUNDO PARCIAL(20)' that was obtained in that partition.
            - overall_parciales_n: column containing a 1 if the student has taken any exam in that partition.
            - assignment_n: column containing a 1 if the student has submitted any assignment in that partition.
            - assignment_zero: column that reflect if an assignment was graded with a 0.

        :param `df`: (pd.DataFrame) DataFrame without intermediate features.

        :return: (pd.DataFrame) DataFrame with intermediate features.
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
    ) -> pd.DataFrame:
        """
        Method that will calculate aggregated features on an expanding DataFrame. This is designed so that the 
        value of a feature calculated at a particular partition, is considering all information upto that 
        partition.

        Applied Transformations:
            Exam related engineered features
            - parciales_n:
                - sum: count of exams taken, until that partition
            - nota_parciales: 
                - min: minimum grade obtained in an exam, until that partition.
                - max: maximum grade obtained in an exam, until that partition.
                - mean: mean grade obtained in an exam, until that partition.
                - std: standard deviation of grades obtained in an exam, until that partition.
            - integradores_n: 
                - sum: count of integradores taken, until that partition
            - nota_integradores:
                - min: minimum grade obtained in an integrador exam, until that partition.
                - max: maximum grade obtained in an integrador exam, until that partition.
                - mean: mean grade obtained in an integrador exam, until that partition.
                - std: standard deviation of grades obtained in an integrador exam, until that partition.
            - recuperatorios_n:
                - sum: count of make-up exams taken, until that partition.
            - nota_recuperatorios:
                - min: minimum grade obtained in a make-up exam, until that partition.
                - max: maximum grade obtained in a make-up exam, until that partition.
                - mean: mean grade obtained in a make-up exam, until that partition.
                - std: standard deviation of grades obtained in a make-up exam, until that partition.
            - overall_parciales_n:
                - sum: count of all kinds of exams taken, until that partition.
            - nota_overall:
                - min: minimum grade obtained on all exams taken, until that partition.
                - max: maximum grade obtained on all exams taken, until that partition.
                - mean: mean grade obtained on all exams taken, until that partition.
                - std: standard deviation of grades obtained on all exams taken, until that partition.

            Assignment related engineered features
            - assignment_n:
                - sum: count of all assignments submitted, until that partition.
            - score:
                - min: minimum grade obtained on all assignments submitted, until that partition.
                - max: maximum grade obtained on all assignments submitted, until that partition.
                - mean: mean grade obtained on all assignments submitted, until that partition.
                - std: standard deviation of grades obtained on all assignments submitted, until that 
                  partition.
            - assignment_zero:
                - sum: count of all assignments submitted which were graded with a 0, until that partition.

            Time related engineered features
            - hrs_to_do_assignment:
                - min: minimum available hours to complete an assignment, until that partition.
                - max: maximum available hours to complete an assignment, until that partition.
                - mean: mean available hours to complete an assignment, until that partition.
                - std: standard deviation of available hours to complete an assignment, until that partition.
            - hrs_taken_to_do_assignment:
                - min: minimum hours taken to complete an assignment, until that partition.
                - max: maximum hours taken to complete an assignment, until that partition.
                - mean: mean hours taken to complete an assignment, until that partition.
                - std: standard deviation of hours taken to complete an assignment, until that partition.
            - rel_time_taken_to_do_assignment:
                - min: minimum relative time taken to complete an assignment, until that partition.
                - max: maximum relative time taken to complete an assignment, until that partition.
                - mean: mean relative time taken to complete an assignment, until that partition.
                - std: standard deviation of the relative time taken to complete an assignment, until that partition.
            - hrs_before_due:
                - min: minimum number of spare hours that the student had when submitting an assignment, until that partition.
                - max: maximum number of spare hours that the student had when submitting an assignment, until that partition.
                - mean: mean number of spare hours that the student had when submitting an assignment, until that partition.
                - std: standard deviation of the number of spare hours that the student had when submitting an assignment, until that partition.

        :param `df`: (pd.DataFrame) DataFrame without expanding features.

        :return: (pd.DataFrame) DataFrame with expanding features.
        """
        def first(col: pd.Series):
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
        """
        Method that will remove partitions until the first exam was taken or assignment was submitted.
        The reason for removing observations until this point is so that ML models will have at least some 
        useful information to make inferences & understand patterns.

        :param `df`: (pd.DataFrame) DataFrame containing "unuseful" observations.

        :return: (pd.DataFrame) DataFrame without "unuseful" observations.
        """
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

    def run_feature_engineering_pipeline(
        self,
        save: bool = False
    ) -> pd.DataFrame:
        """
        Method that executes the feature engineering pipeline, which will:
            - Add target column
            - Calculate timing based features
            - Calculate intermediate features
            - Calculate aggregated features on expanding DataFrame
            - Remove unusefull observations
            - Fill null observations
            - Save engineered dataset

        :param save: (bool) Whether to save the cleaned dataset.

        :return: (pd.DataFrame) Cleaned DataFrame.
        """
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
        """
        Save the engineered dataset to a CSV file.
        """
        print('\nSaving new engineered DataFrame.\n\n')
        self.df.to_csv(Params.ml_data_path)

    def load_dataset(self):
        """
        Load the engineered dataset to a CSV file.
        """
        self.df = pd.read_csv(Params.ml_data_path, index_col=0)
