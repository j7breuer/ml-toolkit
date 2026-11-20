
import pandas as pd
import numpy as np
import pytest

from core.ml.evaluation.diagnostics import Diagnostics

from sklearn.metrics import confusion_matrix as sk_confusion_matrix
from sklearn.metrics import precision_recall_fscore_support


# --------------------------------------------------
# Test datasets
# --------------------------------------------------

# Classification dataset
y_class_true = np.array([0, 0, 0, 0, 1, 1, 1, 1])
y_class_pred = np.array([0, 0, 1, 0, 1, 1, 0, 1])


# Regression dataset
y_reg_true = np.array([100, 150, 200, 250, 300])
y_reg_pred = np.array([110, 140, 190, 260, 280])


# Stability scores
scores = np.array([0.91, 0.89, 0.92, 0.90, 0.88])


# --------------------------------------------------
# Confusion Matrix
# --------------------------------------------------

def test_confusion_matrix_default():
    expected = sk_confusion_matrix(y_class_true, y_class_pred)
    actual = Diagnostics.confusion_matrix(y_class_true, y_class_pred)

    assert np.array_equal(actual.to_numpy(), expected)


def test_confusion_matrix_custom_labels():
    labels = [0, 1]
    expected = sk_confusion_matrix(y_class_true, y_class_pred, labels=labels)
    actual = Diagnostics.confusion_matrix(y_class_true, y_class_pred, labels=labels)

    assert np.array_equal(actual.to_numpy(), expected)


def test_confusion_matrix_normalized():
    expected = sk_confusion_matrix(y_class_true, y_class_pred, normalize="true")
    actual = Diagnostics.confusion_matrix(y_class_true, y_class_pred, normalize="true")

    assert np.allclose(actual.to_numpy(), expected)


def test_confusion_matrix_returns_dataframe():
    actual = Diagnostics.confusion_matrix(y_class_true, y_class_pred)

    assert isinstance(actual, pd.DataFrame)


def test_confusion_matrix_labels():
    actual = Diagnostics.confusion_matrix(y_class_true, y_class_pred)

    assert list(actual.index) == ["Actual 0", "Actual 1"]
    assert list(actual.columns) == ["Predicted 0", "Predicted 1"]


# --------------------------------------------------
# Residual Analysis
# --------------------------------------------------

def test_residual_analysis_values():
    expected_residuals = y_reg_true - y_reg_pred
    actual = Diagnostics.residual_analysis(y_reg_true, y_reg_pred)

    assert np.array_equal(actual["Residual"].to_numpy(), expected_residuals)


def test_residual_analysis_absolute_residuals():
    expected = np.abs(y_reg_true - y_reg_pred)
    actual = Diagnostics.residual_analysis(y_reg_true, y_reg_pred)

    assert np.array_equal(actual["Absolute Residual"].to_numpy(), expected)


def test_residual_analysis_squared_residuals():
    expected = (y_reg_true - y_reg_pred) ** 2
    actual = Diagnostics.residual_analysis(y_reg_true, y_reg_pred)

    assert np.array_equal(actual["Squared Residual"].to_numpy(), expected)


def test_residual_analysis_actual_and_predicted():
    actual = Diagnostics.residual_analysis(y_reg_true, y_reg_pred)

    assert np.array_equal(actual["Actual"].to_numpy(), y_reg_true)
    assert np.array_equal(actual["Predicted"].to_numpy(), y_reg_pred)


def test_residual_analysis_returns_dataframe():
    actual = Diagnostics.residual_analysis(y_reg_true, y_reg_pred)

    assert isinstance(actual, pd.DataFrame)


def test_residual_analysis_columns():
    actual = Diagnostics.residual_analysis(y_reg_true, y_reg_pred)

    assert list(actual.columns) == ["Actual", "Predicted", "Residual", "Absolute Residual", "Squared Residual"]


# --------------------------------------------------
# Per-Class Metrics
# --------------------------------------------------

def test_per_class_metrics_default():
    expected = precision_recall_fscore_support(y_class_true, y_class_pred, zero_division=0)
    actual = Diagnostics.per_class_metrics(y_class_true, y_class_pred)

    assert actual["Precision"].to_numpy() == pytest.approx(expected[0])
    assert actual["Recall"].to_numpy() == pytest.approx(expected[1])
    assert actual["F1-Score"].to_numpy() == pytest.approx(expected[2])
    assert actual["Support"].to_numpy() == pytest.approx(expected[3])


def test_per_class_metrics_custom_labels():
    labels = [0, 1]
    expected = precision_recall_fscore_support(y_class_true, y_class_pred, labels=labels, zero_division=0)
    actual = Diagnostics.per_class_metrics(y_class_true, y_class_pred, labels=labels)

    assert actual["Precision"].to_numpy() == pytest.approx(expected[0])
    assert actual["Recall"].to_numpy() == pytest.approx(expected[1])
    assert actual["F1-Score"].to_numpy() == pytest.approx(expected[2])
    assert actual["Support"].to_numpy() == pytest.approx(expected[3])


def test_per_class_metrics_returns_dataframe():
    actual = Diagnostics.per_class_metrics(y_class_true, y_class_pred)

    assert isinstance(actual, pd.DataFrame)


def test_per_class_metrics_columns():
    actual = Diagnostics.per_class_metrics(y_class_true, y_class_pred)

    assert list(actual.columns) == ["Precision", "Recall", "F1-Score", "Support"]


# --------------------------------------------------
# Stability Analysis
# --------------------------------------------------

def test_stability_analysis_statistics():
    expected_mean = np.mean(scores)
    expected_std = np.std(scores, ddof=1)
    expected_variance = np.var(scores, ddof=1)
    expected_min = np.min(scores)
    expected_max = np.max(scores)
    expected_range = np.ptp(scores)

    actual = Diagnostics.stability_analysis(scores)

    assert actual["mean"] == pytest.approx(expected_mean)
    assert actual["std"] == pytest.approx(expected_std)
    assert actual["variance"] == pytest.approx(expected_variance)
    assert actual["min"] == pytest.approx(expected_min)
    assert actual["max"] == pytest.approx(expected_max)
    assert actual["range"] == pytest.approx(expected_range)


def test_stability_analysis_single_score():
    single_score = np.array([0.90])
    actual = Diagnostics.stability_analysis(single_score)

    assert actual["mean"] == pytest.approx(0.90)
    assert actual["std"] == pytest.approx(0.0)
    assert actual["variance"] == pytest.approx(0.0)
    assert actual["min"] == pytest.approx(0.90)
    assert actual["max"] == pytest.approx(0.90)
    assert actual["range"] == pytest.approx(0.0)


def test_stability_analysis_returns_dict():
    actual = Diagnostics.stability_analysis(scores)

    assert isinstance(actual, dict)


def test_stability_analysis_keys():
    actual = Diagnostics.stability_analysis(scores)

    assert list(actual.keys()) == ["mean", "std", "variance", "min", "max", "range"]
