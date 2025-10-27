import pandas as pd
from sklearn.impute import KNNImputer
from sklearn.linear_model import LinearRegression

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
    def KNN_imputation(df: pd.DataFrame, n_neighbors: int = 2, columns: list[str] = None) -> pd.DataFrame:
        '''
            desc: 
                Replace missing values using K-Nearest Neighbors imputation
            inpt:
                df [pd.DataFrame]: DataFrame with missing values
                n_neighbors [int]: Number of neighboring samples to use for imputation
                columns [list[str]]: List of columns to apply imputation
            oupt:
                df [pd.DataFrame]: DataFrame with missing values filled
        '''
        if columns is None:
            columns = df.columns
        imputer = KNNImputer(n_neighbors=n_neighbors) # create imputer object
        df[columns] = imputer.fit_transform(df[columns]) # using that object, fit and apply the changes to the columns
        return df

    @staticmethod
    def regression_imputation(df: pd.DataFrame, target_column: str) -> pd.DataFrame:
        pass