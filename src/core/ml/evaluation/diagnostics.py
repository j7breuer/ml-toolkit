
import pandas as pd
import numpy as np

from sklearn.metrics import confusion_matrix as sk_confusion_matrix
from sklearn.metrics import precision_recall_fscore_support


class Diagnostics:
    """
    Class for diagnosing machine learning model performance.
    """

    @staticmethod
    def confusion_matrix(y_true, y_pred, labels=None, normalize=None) -> pd.DataFrame:
        """
            desc: 
                Generate a confusion matrix as a pandas DataFrame.
            input:
                y_true [array-like]: True labels
                y_pred [array-like]: Predicted labels
                labels [list]: List of labels to index the matrix
                normalize [str]: Normalization mode ('true', 'pred', 'all', or None
            output:
                pd.DataFrame: Confusion matrix with actual vs predicted labels
        """
        cm = sk_confusion_matrix(y_true, y_pred, labels=labels, normalize=normalize)

        if labels is None:
            labels = np.unique(np.concatenate([y_true, y_pred]))

        return pd.DataFrame(
            cm,
            index=[f"Actual {label}" for label in labels],
            columns=[f"Predicted {label}" for label in labels]
        )

    @staticmethod
    def residual_analysis(y_true, y_pred) -> pd.DataFrame:
        """
            desc: 
                Generate a residual analysis DataFrame.
            input:
                y_true [array-like]: True labels
                y_pred [array-like]: Predicted labels
            output:
                pd.DataFrame: DataFrame containing residual analysis
        """
        residuals = np.asarray(y_true) - np.asarray(y_pred)

        return pd.DataFrame({
            "Actual": np.asarray(y_true),
            "Predicted": np.asarray(y_pred),
            "Residual": residuals,
            "Absolute Residual": np.abs(residuals),
            "Squared Residual": residuals ** 2
        })

    @staticmethod
    def per_class_metrics(y_true, y_pred, labels=None, zero_division=0) -> pd.DataFrame:
        """
            desc: 
                Calculate precision, recall, F1-score, and support for each class.
            input:
                y_true [array-like]: True labels
                y_pred [array-like]: Predicted labels
                labels [list]: List of labels to index the matrix
                zero_division [int]: Value to return when there is a zero division
            output:
                pd.DataFrame: Per-class metrics
        """
        precision, recall, f1, support = precision_recall_fscore_support(
            y_true,
            y_pred,
            labels=labels,
            zero_division=zero_division
        )

        if labels is None:
            labels = np.unique(np.concatenate([y_true, y_pred]))

        return pd.DataFrame({
            "Precision": precision,
            "Recall": recall,
            "F1-Score": f1,
            "Support": support
        }, index=labels)

    @staticmethod
    def stability_analysis(scores) -> dict:
        """
            desc: 
                Calculate stability metrics (mean, std, variance, min, max, range) for a list of scores.
            input:
                scores [array-like]: List of scores
            output:
                dict: Dictionary containing stability metrics
        """
        scores = np.asarray(scores, dtype=float)

        return {
            "mean": np.mean(scores),
            "std": np.std(scores, ddof=1) if len(scores) > 1 else 0.0,
            "variance": np.var(scores, ddof=1) if len(scores) > 1 else 0.0,
            "min": np.min(scores),
            "max": np.max(scores),
            "range": np.ptp(scores)
        }
