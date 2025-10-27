import pandas as pd


class ValueFiller:
    '''
        desc: Class for filling missing values in datasets
    ''' 
    @staticmethod
    def constant_imputation(df: pd.DataFrame, value: int, columns: list[str] = None) -> pd.DataFrame:
        '''
            desc: 
                Replace missing values with a constant value
            inpt:
                df [pd.DataFrame]: DataFrame with missing values
                value [int]: Constant value to replace missing values
                columns [list[str]]: List of columns to apply imputation
            oupt:
                df [pd.DataFrame]: DataFrame with missing values filled
        '''
        if columns is None:
            columns = df.columns

        df[columns] = df[columns].fillna(value)
        return df

    @staticmethod
    def mean_imputation(df: pd.DataFrame, columns: list[str] = None) -> pd.DataFrame:
        '''
            desc: 
                Replace missing values with the mean of the column
            inpt:
                df [pd.DataFrame]: DataFrame with missing values
                columns [list[str]]: List of columns to apply imputation
            oupt:
                df [pd.DataFrame]: DataFrame with missing values filled
        '''
        if columns is None:
            columns = df.columns

        df[columns] = df[columns].fillna(df[columns].mean())
        return df

    @staticmethod
    def median_imputation(df: pd.DataFrame, columns: list[str] = None) -> pd.DataFrame:
        '''
            desc: 
                Replace missing values with the median of the column
            inpt:
                df [pd.DataFrame]: DataFrame with missing values
                columns [list[str]]: List of columns to apply imputation
            oupt:
                df [pd.DataFrame]: DataFrame with missing values filled
        '''
        if columns is None:
            columns = df.columns

        df[columns] = df[columns].fillna(df[columns].median())
        return df

    @staticmethod
    def mode_imputation(df: pd.DataFrame, columns: list[str] = None) -> pd.DataFrame:
        '''
            desc: 
                Replace missing values with the mode of the column
            inpt:
                df [pd.DataFrame]: DataFrame with missing values
                columns [list[str]]: List of columns to apply imputation
            oupt:
                df [pd.DataFrame]: DataFrame with missing values filled
        '''
        if columns is None:
            columns = df.columns

        df[columns] = df[columns].fillna(df[columns].mode().iloc[0]) #iloc needed to select the first mode from the resulting df
        return df

    @staticmethod
    def KNN_imputation(df: pd.DataFrame, n_neighbors: int) -> pd.DataFrame:
        pass

    @staticmethod
    def regression_imputation(df: pd.DataFrame, target_column: str) -> pd.DataFrame:
        pass