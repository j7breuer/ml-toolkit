import pandas as pd
from sklearn.impute import KNNImputer
from sklearn.linear_model import LinearRegression

class ValueFiller:
    '''
        desc: Class for filling missing values in datasets
    ''' 
    @staticmethod
    def add_flags(df: pd.DataFrame, columns: list[str], mask: pd.DataFrame):
        '''
            desc: 
                Add flag columns indicating where imputations were made
            inpt:
                df [pd.DataFrame]: DataFrame with missing values
                columns [list[str]]: List of columns to add flags for
                mask [pd.DataFrame]: Boolean DataFrame indicating where imputations were made
            oupt:
                df [pd.DataFrame]: DataFrame with added flag columns
        '''
        for col in columns:
            flag_col = f"{col}_imputed"
            df[flag_col] = mask[col]
        return df
    
    @staticmethod
    def add_flags(df: pd.DataFrame, column: str, mask: pd.Series):
        '''
            desc: 
                Add flag columns indicating where imputations were made
            inpt:
                df [pd.DataFrame]: DataFrame with missing values
                columns [list[str]]: List of columns to add flags for
                mask [pd.DataFrame]: Boolean DataFrame indicating where imputations were made
            oupt:
                df [pd.DataFrame]: DataFrame with added flag columns
        '''
        flag_col = f"{column}_imputed"
        df[flag_col] = mask
        return df
    
    @staticmethod
    def constant_imputation(df: pd.DataFrame, value: int, columns: list[str] = None, flag_imputations: bool = False) -> pd.DataFrame:
        '''
            desc: 
                Replace missing values with a constant value
            inpt:
                df [pd.DataFrame]: DataFrame with missing values
                value [int]: Constant value to replace missing values
                columns [list[str]]: List of columns to apply imputation
                flag_imputations [bool]: Whether to add flags for imputed values
            oupt:
                df [pd.DataFrame]: DataFrame with missing values filled
        '''
        mask = df[columns].isna() if flag_imputations else None
        if columns is None:
            columns = df.columns

        df[columns] = df[columns].fillna(value)
        if flag_imputations:
            df = ValueFiller.add_flags(df, columns, mask)
        return df

    @staticmethod
    def mean_imputation(df: pd.DataFrame, columns: list[str] = None, flag_imputations: bool = False) -> pd.DataFrame:
        '''
            desc: 
                Replace missing values with the mean of the column
            inpt:
                df [pd.DataFrame]: DataFrame with missing values
                columns [list[str]]: List of columns to apply imputation
                flag_imputations [bool]: Whether to add flags for imputed values
            oupt:
                df [pd.DataFrame]: DataFrame with missing values filled
        '''
        mask = df[columns].isna() if flag_imputations else None
        if columns is None:
            columns = df.columns

        df[columns] = df[columns].fillna(df[columns].mean())
        if flag_imputations:
            df = ValueFiller.add_flags(df, columns, mask)
        return df

    @staticmethod
    def median_imputation(df: pd.DataFrame, columns: list[str] = None, flag_imputations: bool = False) -> pd.DataFrame:
        '''
            desc: 
                Replace missing values with the median of the column
            inpt:
                df [pd.DataFrame]: DataFrame with missing values
                columns [list[str]]: List of columns to apply imputation
                flag_imputations [bool]: Whether to add flags for imputed values
            oupt:
                df [pd.DataFrame]: DataFrame with missing values filled
        '''
        mask = df[columns].isna() if flag_imputations else None
        if columns is None:
            columns = df.columns

        df[columns] = df[columns].fillna(df[columns].median())
        if flag_imputations:
            df = ValueFiller.add_flags(df, columns, mask)
        return df

    @staticmethod
    def mode_imputation(df: pd.DataFrame, columns: list[str] = None, flag_imputations: bool = False) -> pd.DataFrame:
        '''
            desc: 
                Replace missing values with the mode of the column
            inpt:
                df [pd.DataFrame]: DataFrame with missing values
                columns [list[str]]: List of columns to apply imputation
                flag_imputations [bool]: Whether to add flags for imputed values
            oupt:
                df [pd.DataFrame]: DataFrame with missing values filled
        '''
        if columns is None:
            columns = df.columns

        df[columns] = df[columns].fillna(df[columns].mode().iloc[0]) #iloc needed to select the first mode from the resulting df
        return df

    @staticmethod
    def KNN_imputation(df: pd.DataFrame, n_neighbors: int = 2, columns: list[str] = None, flag_imputations: bool = False) -> pd.DataFrame:
        '''
            desc: 
                Replace missing values using K-Nearest Neighbors imputation
            inpt:
                df [pd.DataFrame]: DataFrame with missing values
                n_neighbors [int]: Number of neighboring samples to use for imputation
                columns [list[str]]: List of columns to apply imputation
                flag_imputations [bool]: Whether to add flags for imputed values
            oupt:
                df [pd.DataFrame]: DataFrame with missing values filled
        '''
        mask = df[columns].isna() if flag_imputations else None
        if columns is None:
            columns = df.columns
        imputer = KNNImputer(n_neighbors=n_neighbors) # create imputer object
        df[columns] = imputer.fit_transform(df[columns]) # using that object, fit and apply the changes to the columns
        if flag_imputations:
            df = ValueFiller.add_flags(df, columns, mask)
        return df

    @staticmethod
    def regression_imputation(df: pd.DataFrame, target_column: str, predictor_columns: list[str], flag_imputations: bool = False) -> pd.DataFrame:
        '''
            desc: 
                Replace missing values using regression imputation
            inpt:
                df [pd.DataFrame]: DataFrame with missing values
                target_column [str]: Column to apply imputation
                predictor_columns [list[str]]: Columns to use as predictors
                flag_imputations [bool]: Whether to add flags for imputed values
            oupt:
                df [pd.DataFrame]: DataFrame with missing values filled
        '''
        mask = df[target_column].isna() if flag_imputations else None
        train_data = df[df[target_column].notna()] # train on the data that isn't null
        predict_data = df[df[target_column].isna()] # predicting null values 
        feature_columns = [col for col in predictor_columns if col != target_column]
        model = LinearRegression()
        model.fit(train_data[feature_columns], train_data[target_column]) # "create" the best fit line
        predicted_values = model.predict(predict_data[feature_columns]) # given the "test" data's features, predict the values 
        df.loc[df[target_column].isna(), target_column] = predicted_values # only fill in values that are null
        if flag_imputations:
            df = ValueFiller.add_flags(df, target_column, mask)
        return df