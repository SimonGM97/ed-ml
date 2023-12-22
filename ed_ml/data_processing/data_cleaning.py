from config.params import Params
from ed_ml.utils.timing import timing
import pandas as pd
import numpy as np


class DataCleaner:
    """
    Class for cleaning and preprocessing raw datasets.
    """

    def __init__(
        self, 
        df: pd.DataFrame = None,
        load_dataset: bool = False
    ) -> None:
        """
        Initialize DataCleaner.

        :param df: (pd.DataFrame) Input DataFrame.
        :param load_dataset: (bool) Whether to load the dataset from data_lake during initialization.
        """
        # Define self.df
        self.df: pd.DataFrame = df

        # Load Dataset
        if load_dataset:
            self.load_dataset()

    @staticmethod
    def correct_periodo(
        df: pd.DataFrame
    ):
        """
        Method that corrects column "periodo" so that it only contains valid values.

        :param save: (pd.DataFrame) DataFrame with raw "periodo" column.

        :return: (pd.DataFrame) DataFrame with corrected "periodo" column values.
        """
        def correct_date(date: str):
            day, year = date.split('-')
            day = ('0' + day)[-2:]
            return day + '-' + year

        # Correct periodo
        df['periodo'] = df['periodo'].replace({
            date: correct_date(date) for date in df['periodo'].unique()
        })

        # Fill particion
        df['particion'].fillna(0, inplace=True)

        return df

    @staticmethod
    def format_datetime_columns(
        df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Method that casts datetime columns into datetime objects.

        :param save: (pd.DataFrame) DataFrame with raw datetime columns.

        :return: (pd.DataFrame) DataFrame with corrected datetime column types.
        """
        # Format datetime columns
        df[Params.datetime_columns] = (
            df[Params.datetime_columns]
            .fillna(0)
            .astype(int)
            .apply(pd.to_datetime, unit='s')
            .replace('01/01/1970  00:00:00', np.nan)
        )

        return df

    @staticmethod
    def fill_null_test_and_ass(
        df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Method that will fill null values for tests & assignments, based on the following assumptions:
            - If a user_uuid has a null nota_parcial, then it will be interpreted as the student being 
              absent for the exam; thus the student will be assigned a 0 score for that exam.
            - If a user_uuid has a null assignment score, then it will be interpreted as the student not 
              submitting any material; thus the student will be assigned a 0 score for that assignment.

        :param save: (pd.DataFrame) DataFrame with unexpected null values in "nota_parcial" or "score".

        :return: (pd.DataFrame) DataFrame without unexpected null values in "nota_parcial" or "score".
        """
        # Fill null nota_parcial
        df.loc[
            (df['nombre_examen'].notna()) &
            (df['nota_parcial'].isna()), 
            'nota_parcial'
        ] = 0

        # Fill null score
        df.loc[
            (df['ass_name_sub'].notna()) &
            (df['score'].isna()), 
            'score'
        ] = 0

        return df

    @staticmethod
    def allign_assignments(
        df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Method that alligns, on a same row, the assignment creations with their respective assignment 
        submissions, by applying the following steps:
            1) Find assignment submissions & assignment creations
            2) Corect assignment submissions with it's corresponding assignment creation
            3) Concatenate corrected assignments with initial DataFrame
            4) Fill assignment creations with no assignment submissions, assuming a 0 score for that 
               assignment.
            5) Clean unnecessary assignment creations (repeated observations that have already been matched)

        :param save: (pd.DataFrame) DataFrame with missaligned assignment creations & assignment submissions.

        :return: (pd.DataFrame) DataFrame with alligned assignment creations & assignment submissions.
        """
        def complete_assignment_creation(gb_df: pd.DataFrame):
            # Find assignment submissions
            ass_sub = gb_df.loc[gb_df['ass_name_sub'].notna()]

            # Find assignment creations
            ass_creat = gb_df.loc[gb_df['ass_name'].notna()]

            # Define assignment creation columns
            ass_creation_cols = [
                'assignment_id', 'ass_name', 'ass_created_at', 'ass_due_at', 
                'ass_unlock_at', 'ass_lock_at', 'points_possible'
            ]

            # Corect assignment submissions with it's corresponding assignment creation
            corrected_ass_sub = pd.merge(
                left=ass_sub.drop(columns=ass_creation_cols), 
                right=ass_creat.filter(items=ass_creation_cols), 
                left_on='ass_name_sub', 
                right_on='ass_name', 
                how='left'
            )
            
            # Concatenate corrected assignments
            gb_df = (
                pd.concat([gb_df.loc[gb_df['ass_name_sub'].isna()], corrected_ass_sub])
                .sort_values(by=['particion'], ascending=True)
            )

            # Correct assignment creations with no assignment submissions
            ass_creat_names = set(ass_creat['ass_name'].unique())
            ass_sub_names = set(ass_sub['ass_name_sub'].unique())
            ass_created_but_not_sub = ass_creat_names - ass_sub_names

            if len(ass_created_but_not_sub) > 0:
                # Complete scores
                gb_df.loc[
                    gb_df['ass_name'].isin(list(ass_created_but_not_sub)), 
                    'score'
                ] = 0

                # Complete Assignment Submission name
                gb_df.loc[
                    gb_df['ass_name'].isin(list(ass_created_but_not_sub)), 
                    'ass_name_sub'
                ] = gb_df.loc[
                    gb_df['ass_name'].isin(list(ass_created_but_not_sub)), 
                    'ass_name'
                ]

            # Clean unnecessary assignment creations
            condition = (gb_df['ass_name'].notna()) & (gb_df['ass_name_sub'].isna())
            gb_df.loc[condition, ass_creation_cols] = np.nan

            return gb_df
        
        # Correct assignment rows for each user_uuid & course_uuid
        df = (
            df
            .groupby(['user_uuid', 'course_uuid'])
            .apply(complete_assignment_creation)
            .reset_index(drop=True)
        )

        return df

    @staticmethod
    def clean_duplicates(
        df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Function that will correct unexpected duplicate observations, based on the following assumptions:
            - If there is more than one observation for the same user_uuid, course_uuid & nombre_examen, 
              this will be interpreted as a multiple part exam, where the final note (for that exam) is 
              given by the average of the partial submissions.
            - If theere is more thatn one observation for the same user_uuid, course_uuid & ass_name_sum, 
              this will be interpreted as a multiple part assignment, where the final note (for that 
              assignment exam) is given by the average of the partial submissions.

        :param save: (pd.DataFrame) DataFrame containing unintended duplicate observations.

        :return: (pd.DataFrame) DataFrame without unintended duplicate observations.
        """
        def find_agg_fun(col: str):
            if col in list(df.select_dtypes(include=['object', 'datetime']).columns):
                return 'first'
            return 'mean'
        
        def clean_duplicates_in(df: pd.DataFrame, col: str):
            # Find observations to correct
            subset_cols = ['user_uuid', 'course_uuid', col]
            mask = (
                (df[col].notna()) &
                (df.duplicated(subset=subset_cols, keep='first') | df.duplicated(subset=subset_cols, keep='last'))
            )
            
            # Correct found observations
            corrected_obs = (
                df.loc[mask]
                .groupby(subset_cols)
                .agg({col: find_agg_fun(col) for col in df.columns})
                .reset_index(drop=True)
            )

            # Concatenate corrected observations
            df = df.loc[~mask]
            df = pd.concat([df, corrected_obs])

            return df

        # Correct parcial duplicates
        df = clean_duplicates_in(df=df, col='nombre_examen')

        # Correct assignment submissions duplicates
        df = clean_duplicates_in(df=df, col='ass_name_sub')

        return df
    
    @timing
    def run_cleaner_pipeline(
        self,
        save: bool = False
    ) -> pd.DataFrame:
        """
        Method that executes the data cleaning pipeline, which will:
            - Correct period values
            - Format datetime columns
            - Fill values for tests & assignments
            - Clean duplicate observations
            - Alligh assignments, so that assignment creation and assignment submissions 
              are found in the same row
            - Sorts results y user_uuid, course_uuid & particion
            - (optionally) saves intermediate results.

        :param save: (bool) Whether to save the cleaned dataset.

        :return: (pd.DataFrame) Cleaned DataFrame.
        """
        # Correct Periodo
        self.df = self.correct_periodo(df=self.df)

        # Format datetime columns
        self.df = self.format_datetime_columns(df=self.df)

        # Clean unexpected null tests & assignments
        self.df = self.fill_null_test_and_ass(df=self.df)

        # Allign assignments
        self.df = self.allign_assignments(df=self.df)

        # Clean duplicate assignments
        self.df = self.clean_duplicates(df=self.df)

        # Sort DataFrame
        self.df.sort_values(
            by=['user_uuid', 'course_uuid', 'particion'], 
            inplace=True
        )

        # Save Dataset
        if save:
            self.save_dataset()
        
        return self.df
    
    def save_dataset(self):
        """
        Save the cleaned dataset to a CSV file.
        """
        print('\nSaving new cleaned DataFrame.\n\n')
        self.df.to_csv(Params.cleaned_data_path)

    def load_dataset(self):
        """
        Load the cleaned dataset from a CSV file.
        """
        self.df = pd.read_csv(Params.cleaned_data_path, index_col=0)
    
