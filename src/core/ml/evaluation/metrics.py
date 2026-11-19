
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    log_loss,
    mean_absolute_error,
    mean_squared_error,
    root_mean_squared_error,
    r2_score,
    mean_absolute_percentage_error,
    median_absolute_error
)


class Metrics:
    """
    Class for computing machine learning model metrics.
    """

    @staticmethod
    def accuracy(y_true, y_pred) -> float:
        """Proportion of correct predictions."""
        return accuracy_score(y_true, y_pred)

    @staticmethod
    def precision(y_true, y_pred, average='binary') -> float:
        """Proportion of positive predictions that are correct."""
        return precision_score(y_true, y_pred, average=average)

    @staticmethod
    def recall(y_true, y_pred, average='binary') -> float:
        """Proportion of actual positives correctly identified."""
        return recall_score(y_true, y_pred, average=average)

    @staticmethod
    def f1(y_true, y_pred, average='binary') -> float:
        """Harmonic mean of precision and recall."""
        return f1_score(y_true, y_pred, average=average)

    @staticmethod
    def roc_auc(y_true, y_prob) -> float:
        """Ability of the model to distinguish between classes."""
        return roc_auc_score(y_true, y_prob)

    @staticmethod
    def pr_auc(y_true, y_prob) -> float:
        """Area under the Precision-Recall curve."""
        return average_precision_score(y_true, y_prob)

    @staticmethod
    def log_loss(y_true, y_prob) -> float:
        """Penalizes confident incorrect predictions."""
        return log_loss(y_true, y_prob)

    @staticmethod
    def cross_entropy(y_true, y_prob) -> float:
        """Cross Entropy loss (equivalent to Log Loss for classification)."""
        return log_loss(y_true, y_prob)

    # Regression Metrics

    @staticmethod
    def mae(y_true, y_pred) -> float:
        """Mean Absolute Error."""
        return mean_absolute_error(y_true, y_pred)

    @staticmethod
    def mse(y_true, y_pred) -> float:
        """Mean Squared Error."""
        return mean_squared_error(y_true, y_pred)

    @staticmethod
    def rmse(y_true, y_pred) -> float:
        """Root Mean Squared Error."""
        return root_mean_squared_error(y_true, y_pred)

    @staticmethod
    def r_squared(y_true, y_pred) -> float:
        """R-squared: proportion of variance explained by the model."""
        return r2_score(y_true, y_pred)

    @staticmethod
    def mape(y_true, y_pred) -> float:
        """Mean Absolute Percentage Error."""
        return mean_absolute_percentage_error(y_true, y_pred)

    @staticmethod
    def median_ae(y_true, y_pred) -> float:
        """Median Absolute Error."""
        return median_absolute_error(y_true, y_pred)
