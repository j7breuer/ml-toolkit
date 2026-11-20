import pandas as pd
import numpy as np
from sklearn import clone
from sklearn.model_selection import GridSearchCV, KFold, LeaveOneOut, RepeatedKFold, StratifiedKFold, cross_val_score
from sklearn.metrics import get_scorer

class Validations:
    """
        desc: Class for validating machine learning models
    """ 

    @staticmethod
    def k_fold_validation(model, X: pd.DataFrame, y: pd.Series, n_splits: int = 5, shuffle: bool = True, scoring: str = None, random_state: int = 111) -> float:
        """
            desc: 
                Perform k-fold cross-validation on the given model
            input:
                model: Machine learning model to validate
                X [pd.DataFrame]: Feature dataset
                y [pd.Series]: Target variable
                n_splits [int]: Number of folds for cross-validation
                shuffle [bool]: Whether to shuffle data before splitting
                scoring [str]: Scoring metric for evaluation
                random_state [int]: Random seed for reproducibility
            output:
                float: Average cross-validation score
        """
        kf = KFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)
        scores = cross_val_score(model, X, y, cv=kf, scoring=scoring)
        return scores.mean()
    
    @staticmethod
    def stratified_k_fold_validation(model, X: pd.DataFrame, y: pd.Series, n_splits: int = 5, shuffle: bool = True, scoring: str = None, random_state: int = 111) -> float:
        """
            desc:
                Perform Stratified K-Fold Cross-Validation.
                Similar to K-Fold but maintains the class distribution across folds (important for classification tasks).
            input:
                model: Machine learning model to evaluate
                X [pd.DataFrame]: Features
                y [pd.Series]: Target (categorical or discrete)
                n_splits [int]: Number of folds (k)
                shuffle [bool]: Whether to shuffle data before splitting
                scoring [str]: Scoring metric for evaluation
                random_state [int]: Random seed for reproducibility
            output:
                float: Average stratified K Fold score        
        """
        skf = StratifiedKFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)
        scores = cross_val_score(model, X, y, cv=skf, scoring=scoring)
        return scores.mean()


    @staticmethod
    def leave_one_out_validation(model, X: pd.DataFrame, y: pd.Series, scoring: str = None) -> float:
        """
        desc:
            Perform Leave-One-Out Cross-Validation (LOOCV).
            Each sample is used once as the validation set.
        input:
            model: Machine learning model to evaluate
            X [pd.DataFrame]: Features
            y [pd.Series]: Target, 
            scoring [str]: Scoring metric for evaluation
        output:
            float: Average validation score
        """
        loo = LeaveOneOut()
        scores = cross_val_score(model, X, y, cv=loo, scoring=scoring)
        return scores.mean()


    @staticmethod
    def repeated_k_fold_validation(model, X: pd.DataFrame, y: pd.Series, n_splits: int = 5, n_repeats: int = 3, scoring: str = None, random_state: int = 111) -> float:
        """
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
                scoring [str]: Scoring metric for evaluation
                random_state [int]: Random seed for reproducibility
            output:
                float: Average cross-validation score
        """
        rkf = RepeatedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=random_state)
        scores = cross_val_score(model, X, y, cv=rkf, scoring=scoring)
        return scores.mean()


    @staticmethod
    def nested_cross_validation(model, X: pd.DataFrame, y: pd.Series, param_grid: dict, outer_splits: int = 5, inner_splits: int = 3, scoring: str = None, random_state: int = 111
    ) -> float:
        """
            desc:
                Perform Nested Cross-Validation.
                Outer loop splits the data for model evaluation.
                Inner loop performs hyperparameter tuning using GridSearchCV.
            input:
                model: Machine learning model to evaluate
                X [pd.DataFrame]: Features
                y [pd.Series]: Target
                param_grid [dict]: Grid of hyperparameters to search
                outer_splits [int]: Number of folds in the outer loop
                inner_splits [int]: Number of folds in the inner loop
                scoring [str]: Scoring metric for evaluation
                random_state [int]: Random seed for reproducibility
            output:
                float: Average cross-validation score
        """
        rkf = RepeatedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=random_state)
        scores = cross_val_score(model, X, y, cv=rkf, scoring=scoring)
        return scores.mean()


    @staticmethod
    def nested_cross_validation(model, X: pd.DataFrame, y: pd.Series, param_grid: dict, outer_splits: int = 5, inner_splits: int = 3, scoring: str = None, random_state: int = 111
    ) -> float:
        """
        desc:
            Perform Nested Cross-Validation.
            Outer loop splits the data for model evaluation.
            Inner loop performs hyperparameter tuning using GridSearchCV.
        input:
            model: Machine learning model to evaluate
            X [pd.DataFrame]: Features
            y [pd.Series]: Target
            param_grid [dict]: Grid of hyperparameters to search
            outer_splits [int]: Number of folds in the outer loop
            inner_splits [int]: Number of folds in the inner loop
            scoring [str]: Scoring metric for evaluation
            random_state [int]: Random seed for reproducibility
        output:
            float: Average cross-validation score
        """
        outer_cv = KFold(n_splits=outer_splits, shuffle=True, random_state=random_state)
        inner_cv = KFold( n_splits=inner_splits, shuffle=True, random_state=random_state)
        scores = []

        for train_idx, test_idx in outer_cv.split(X, y):

            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

            grid_search = GridSearchCV(
                estimator=clone(model),
                param_grid=param_grid,
                cv=inner_cv,
                scoring=scoring
            )

            grid_search.fit(X_train, y_train)

            best_model = grid_search.best_estimator_

            if scoring is None:
                score = best_model.score(X_test, y_test)
            else:
                score = get_scorer(scoring)(best_model, X_test, y_test)

            scores.append(score)

        return float(np.mean(scores))