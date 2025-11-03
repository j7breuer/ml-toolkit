import pandas as pd
from sklearn.model_selection import KFold, LeaveOneOut, RepeatedKFold, StratifiedKFold, cross_val_score

class ModelValidator:
    '''
        desc: Class for validating machine learning models
    ''' 

    @staticmethod
    def k_fold_validation(model, X: pd.DataFrame, y: pd.Series, n_splits: int = 5, shuffle: bool = True, random_state: int = 42) -> float:
        '''
            desc: 
                Perform k-fold cross-validation on the given model
            input:
                model: Machine learning model to validate
                X [pd.DataFrame]: Feature dataset
                y [pd.Series]: Target variable
                n_splits [int]: Number of folds for cross-validation
                shuffle [bool]: Whether to shuffle data before splitting
                random_state [int]: Random seed for reproducibility
            output:
                float: Average cross-validation score
        '''
        kf = KFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)
        scores = cross_val_score(model, X, y, cv=kf)
        return scores.mean()
    
    @staticmethod
    def stratified_k_fold_validation(model, X: pd.DataFrame, y: pd.Series, n_splits: int = 5, shuffle: bool = True, random_state: int = 42) -> float:
        '''
            desc:
                Perform Stratified K-Fold Cross-Validation.
                Similar to K-Fold but maintains the class distribution across folds (important for classification tasks).
            input:
                model: Machine learning model to evaluate
                X [pd.DataFrame]: Features
                y [pd.Series]: Target (categorical or discrete)
                n_splits [int]: Number of folds (k)
                shuffle [bool]: Whether to shuffle data before splitting
                random_state [int]: Random seed for reproducibility
            output:
                float: Average stratified K Fold score        
        '''
        skf = StratifiedKFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)
        scores = cross_val_score(model, X, y, cv=skf)
        return scores.mean()


    @staticmethod
    def leave_one_out_validation(model, X: pd.DataFrame, y: pd.Series) -> float:
        '''
            desc:
                Perform Leave-One-Out Cross-Validation (LOOCV)
                Each sample is used once as a test set, and the remaining n-1 samples are used for training
                Note: Computationally expensive for large datasets
            input:
                model: Machine learning model to evaluate
                X [pd.DataFrame]: Features
                y [pd.Series]: Target
            output:
                float: Average cross-validation score
        '''
        loo = LeaveOneOut()
        scores = cross_val_score(model, X, y, cv=loo)
        return scores.mean()


    @staticmethod
    def repeated_k_fold_validation(model, X: pd.DataFrame, y: pd.Series, n_splits: int = 5, n_repeats: int = 3, random_state: int = 42) -> float:
        '''
            desc:
                Perform Repeated K-Fold Cross-Validation.
                K-Fold is repeated multiple times with different random splits.
                Results are averaged to get a more robust performance estimate.
            input:
                model: Machine learning model to evaluate
                X [pd.DataFrame]: Features
                y [pd.Series]: Target
                n_splits [int]: Number of folds in each repetition
                n_repeats [int]: Number of times to repeat the K-Fold process
                random_state [int]: Random seed for reproducibility
            output:
                float: Average cross-validation score
        '''
        rkf = RepeatedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=random_state)
        scores = cross_val_score(model, X, y, cv=rkf)
        return scores.mean()


    @staticmethod
    def nested_cross_validation(model, X: pd.DataFrame, y: pd.Series, outer_splits: int = 5, inner_splits: int = 3, random_state: int = 42) -> float:
        pass