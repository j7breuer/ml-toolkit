from core.ml.evaluation.metrics import Metrics
import numpy as np
import pytest
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
# ==========================================
# Classification Test Data
# ==========================================

y_true_class = np.array([0, 1, 1, 0, 1, 0, 1, 0])
y_pred_class = np.array([0, 1, 0, 0, 1, 1, 1, 0])

y_prob_class = np.array([
    0.10, 0.90, 0.40, 0.20,
    0.85, 0.60, 0.95, 0.15
])


# ==========================================
# Regression Test Data
# ==========================================

y_true_reg = np.array([3.0, -0.5, 2.0, 7.0])
y_pred_reg = np.array([2.5, 0.0, 2.0, 8.0])


# ==========================================
# Classification Tests
# ==========================================

def test_accuracy():
    expected = accuracy_score(y_true_class, y_pred_class)
    actual = Metrics.accuracy(y_true_class, y_pred_class)

    assert actual == pytest.approx(expected)


def test_precision():
    expected = precision_score(y_true_class, y_pred_class)
    actual = Metrics.precision(y_true_class, y_pred_class)

    assert actual == pytest.approx(expected)


def test_recall():
    expected = recall_score(y_true_class, y_pred_class)
    actual = Metrics.recall(y_true_class, y_pred_class)

    assert actual == pytest.approx(expected)


def test_f1():
    expected = f1_score(y_true_class, y_pred_class)
    actual = Metrics.f1(y_true_class, y_pred_class)

    assert actual == pytest.approx(expected)


def test_roc_auc():
    expected = roc_auc_score(y_true_class, y_prob_class)
    actual = Metrics.roc_auc(y_true_class, y_prob_class)

    assert actual == pytest.approx(expected)


def test_pr_auc():
    expected = average_precision_score(y_true_class, y_prob_class)
    actual = Metrics.pr_auc(y_true_class, y_prob_class)

    assert actual == pytest.approx(expected)


def test_log_loss():
    expected = log_loss(y_true_class, y_prob_class)
    actual = Metrics.log_loss(y_true_class, y_prob_class)

    assert actual == pytest.approx(expected)


def test_cross_entropy():
    expected = log_loss(y_true_class, y_prob_class)
    actual = Metrics.cross_entropy(y_true_class, y_prob_class)

    assert actual == pytest.approx(expected)


# ==========================================
# Regression Tests
# ==========================================

def test_mae():
    expected = mean_absolute_error(y_true_reg, y_pred_reg)
    actual = Metrics.mae(y_true_reg, y_pred_reg)

    assert actual == pytest.approx(expected)


def test_mse():
    expected = mean_squared_error(y_true_reg, y_pred_reg)
    actual = Metrics.mse(y_true_reg, y_pred_reg)

    assert actual == pytest.approx(expected)


def test_rmse():
    expected = root_mean_squared_error(y_true_reg, y_pred_reg)
    actual = Metrics.rmse(y_true_reg, y_pred_reg)

    assert actual == pytest.approx(expected)


def test_r_squared():
    expected = r2_score(y_true_reg, y_pred_reg)
    actual = Metrics.r_squared(y_true_reg, y_pred_reg)

    assert actual == pytest.approx(expected)


def test_mape():
    expected = mean_absolute_percentage_error(y_true_reg, y_pred_reg)
    actual = Metrics.mape(y_true_reg, y_pred_reg)

    assert actual == pytest.approx(expected)


def test_median_ae():
    expected = median_absolute_error(y_true_reg, y_pred_reg)
    actual = Metrics.median_ae(y_true_reg, y_pred_reg)

    assert actual == pytest.approx(expected)


# ==========================================
# Additional Classification Tests
# ==========================================

def test_precision_multiclass():
    y_true = [0, 1, 2, 0, 1, 2]
    y_pred = [0, 2, 2, 0, 1, 1]

    expected = precision_score(
        y_true, y_pred, average='weighted'
    )

    actual = Metrics.precision(
        y_true, y_pred, average='weighted'
    )

    assert actual == pytest.approx(expected)


def test_recall_multiclass():
    y_true = [0, 1, 2, 0, 1, 2]
    y_pred = [0, 2, 2, 0, 1, 1]

    expected = recall_score(
        y_true, y_pred, average='weighted'
    )

    actual = Metrics.recall(
        y_true, y_pred, average='weighted'
    )

    assert actual == pytest.approx(expected)


def test_f1_multiclass():
    y_true = [0, 1, 2, 0, 1, 2]
    y_pred = [0, 2, 2, 0, 1, 1]

    expected = f1_score(
        y_true, y_pred, average='weighted'
    )

    actual = Metrics.f1(
        y_true, y_pred, average='weighted'
    )

    assert actual == pytest.approx(expected)


# ==========================================
# Perfect Prediction Tests
# ==========================================

def test_perfect_classification():
    y_true = np.array([0, 1, 1, 0])
    y_pred = np.array([0, 1, 1, 0])
    y_prob = np.array([0.01, 0.99, 0.99, 0.01])

    assert Metrics.accuracy(y_true, y_pred) == 1.0
    assert Metrics.precision(y_true, y_pred) == 1.0
    assert Metrics.recall(y_true, y_pred) == 1.0
    assert Metrics.f1(y_true, y_pred) == 1.0
    assert Metrics.roc_auc(y_true, y_prob) == 1.0
    assert Metrics.pr_auc(y_true, y_prob) == 1.0


def test_perfect_regression():
    y_true = np.array([1.0, 2.0, 3.0, 4.0])
    y_pred = np.array([1.0, 2.0, 3.0, 4.0])

    assert Metrics.mae(y_true, y_pred) == 0.0
    assert Metrics.mse(y_true, y_pred) == 0.0
    assert Metrics.rmse(y_true, y_pred) == 0.0
    assert Metrics.r_squared(y_true, y_pred) == 1.0
    assert Metrics.mape(y_true, y_pred) == 0.0
    assert Metrics.median_ae(y_true, y_pred) == 0.0


# ==========================================
# Metric Consistency Tests
# ==========================================

def test_rmse_is_sqrt_mse():
    mse = Metrics.mse(y_true_reg, y_pred_reg)
    rmse = Metrics.rmse(y_true_reg, y_pred_reg)

    assert rmse == pytest.approx(np.sqrt(mse))


def test_log_loss_equals_cross_entropy():
    logloss = Metrics.log_loss(y_true_class, y_prob_class)
    cross_entropy = Metrics.cross_entropy(y_true_class, y_prob_class)

    assert logloss == pytest.approx(cross_entropy)