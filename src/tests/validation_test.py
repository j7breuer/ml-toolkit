
import pandas as pd
import numpy as np
import pytest

from core.ml.evaluation.validation import Validations

from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import (
    KFold,
    StratifiedKFold,
    LeaveOneOut,
    RepeatedKFold,
    GridSearchCV,
    cross_val_score
)
from sklearn.base import clone
from sklearn.metrics import get_scorer


# ==========================================
# Test Data
# ==========================================

# Regression dataset
X_reg = pd.DataFrame({
    "feature_1": np.arange(1, 31),
    "feature_2": np.arange(31, 61)
})

y_reg = pd.Series(
    2 * X_reg["feature_1"] + 3 * X_reg["feature_2"] + 5
)

# Classification dataset
X_class = pd.DataFrame({
    "feature_1": np.arange(1, 41),
    "feature_2": np.arange(41, 81)
})

y_class = pd.Series([0] * 20 + [1] * 20)

# --------------------------------------------------
# K-Fold Validation
# --------------------------------------------------

def test_k_fold_validation_default_scoring():
    model = LinearRegression()
    expected = cross_val_score(model, X_reg, y_reg, cv=KFold(n_splits=5, shuffle=True, random_state=111)).mean()
    actual = Validations.k_fold_validation(model, X_reg, y_reg)
    assert actual == pytest.approx(expected)


def test_k_fold_validation_custom_scoring():
    model = LinearRegression()
    expected = cross_val_score(model, X_reg, y_reg, cv=KFold(n_splits=5, shuffle=True, random_state=111), scoring="neg_mean_squared_error").mean()
    actual = Validations.k_fold_validation(model, X_reg, y_reg, scoring="neg_mean_squared_error")
    assert actual == pytest.approx(expected)


def test_k_fold_validation_custom_splits():
    model = LinearRegression()
    expected = cross_val_score(model, X_reg, y_reg, cv=KFold(n_splits=3, shuffle=True, random_state=111)).mean()
    actual = Validations.k_fold_validation(model, X_reg, y_reg, n_splits=3)
    assert actual == pytest.approx(expected)


# --------------------------------------------------
# Stratified K-Fold Validation
# --------------------------------------------------

def test_stratified_k_fold_validation_default_scoring():
    model = LogisticRegression()
    expected = cross_val_score(model, X_class, y_class, cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=111)).mean()
    actual = Validations.stratified_k_fold_validation(model, X_class, y_class)
    assert actual == pytest.approx(expected)


def test_stratified_k_fold_validation_custom_scoring():
    model = LogisticRegression()
    expected = cross_val_score(model, X_class, y_class, cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=111), scoring="f1").mean()
    actual = Validations.stratified_k_fold_validation(model, X_class, y_class, scoring="f1")
    assert actual == pytest.approx(expected)


def test_stratified_k_fold_validation_custom_splits():
    model = LogisticRegression()
    expected = cross_val_score(model, X_class, y_class, cv=StratifiedKFold(n_splits=4, shuffle=True, random_state=111)).mean()
    actual = Validations.stratified_k_fold_validation(model, X_class, y_class, n_splits=4)
    assert actual == pytest.approx(expected)


# --------------------------------------------------
# Leave-One-Out Validation
# --------------------------------------------------

def test_leave_one_out_validation_custom_scoring():
    model = LinearRegression()
    expected = cross_val_score(model, X_reg, y_reg, cv=LeaveOneOut(), scoring="neg_mean_absolute_error").mean()
    actual = Validations.leave_one_out_validation(model, X_reg, y_reg, scoring="neg_mean_absolute_error")
    assert actual == pytest.approx(expected)


# --------------------------------------------------
# Repeated K-Fold Validation
# --------------------------------------------------

def test_repeated_k_fold_validation_default_scoring():
    model = LinearRegression()
    expected = cross_val_score(model, X_reg, y_reg, cv=RepeatedKFold(n_splits=5, n_repeats=3, random_state=111)).mean()
    actual = Validations.repeated_k_fold_validation(model, X_reg, y_reg)
    assert actual == pytest.approx(expected)


def test_repeated_k_fold_validation_custom_scoring():
    model = LinearRegression()
    expected = cross_val_score(model, X_reg, y_reg, cv=RepeatedKFold(n_splits=5, n_repeats=3, random_state=111), scoring="neg_mean_squared_error").mean()
    actual = Validations.repeated_k_fold_validation(model, X_reg, y_reg, scoring="neg_mean_squared_error")
    assert actual == pytest.approx(expected)


def test_repeated_k_fold_validation_custom_splits_and_repeats():
    model = LinearRegression()
    expected = cross_val_score(model, X_reg, y_reg, cv=RepeatedKFold(n_splits=3, n_repeats=2, random_state=111)).mean()
    actual = Validations.repeated_k_fold_validation(model, X_reg, y_reg, n_splits=3, n_repeats=2)
    assert actual == pytest.approx(expected)


# --------------------------------------------------
# Nested Cross-Validation
# --------------------------------------------------

def test_nested_cross_validation_default_scoring():
    model = Ridge()
    param_grid = {"alpha": [0.1, 1.0, 10.0]}
    outer_cv = KFold(n_splits=5, shuffle=True, random_state=111)
    inner_cv = KFold(n_splits=3, shuffle=True, random_state=111)
    expected_scores = []

    for train_idx, test_idx in outer_cv.split(X_reg, y_reg):
        X_train, X_test = X_reg.iloc[train_idx], X_reg.iloc[test_idx]
        y_train, y_test = y_reg.iloc[train_idx], y_reg.iloc[test_idx]
        grid_search = GridSearchCV(estimator=clone(model), param_grid=param_grid, cv=inner_cv)
        grid_search.fit(X_train, y_train)
        expected_scores.append(grid_search.best_estimator_.score(X_test, y_test))

    expected = np.mean(expected_scores)
    actual = Validations.nested_cross_validation(model, X_reg, y_reg, param_grid)
    assert actual == pytest.approx(expected)


def test_nested_cross_validation_custom_scoring():
    model = Ridge()
    param_grid = {"alpha": [0.1, 1.0, 10.0]}
    outer_cv = KFold(n_splits=5, shuffle=True, random_state=111)
    inner_cv = KFold(n_splits=3, shuffle=True, random_state=111)
    expected_scores = []

    for train_idx, test_idx in outer_cv.split(X_reg, y_reg):
        X_train, X_test = X_reg.iloc[train_idx], X_reg.iloc[test_idx]
        y_train, y_test = y_reg.iloc[train_idx], y_reg.iloc[test_idx]
        grid_search = GridSearchCV(estimator=clone(model), param_grid=param_grid, cv=inner_cv, scoring="neg_mean_squared_error")
        grid_search.fit(X_train, y_train)
        expected_scores.append(get_scorer("neg_mean_squared_error")(grid_search.best_estimator_, X_test, y_test))

    expected = np.mean(expected_scores)
    actual = Validations.nested_cross_validation(model, X_reg, y_reg, param_grid, scoring="neg_mean_squared_error")
    assert actual == pytest.approx(expected)


def test_nested_cross_validation_custom_splits():
    model = Ridge()
    param_grid = {"alpha": [0.1, 1.0, 10.0]}
    outer_cv = KFold(n_splits=3, shuffle=True, random_state=111)
    inner_cv = KFold(n_splits=2, shuffle=True, random_state=111)
    expected_scores = []

    for train_idx, test_idx in outer_cv.split(X_reg, y_reg):
        X_train, X_test = X_reg.iloc[train_idx], X_reg.iloc[test_idx]
        y_train, y_test = y_reg.iloc[train_idx], y_reg.iloc[test_idx]
        grid_search = GridSearchCV(estimator=clone(model), param_grid=param_grid, cv=inner_cv)
        grid_search.fit(X_train, y_train)
        expected_scores.append(grid_search.best_estimator_.score(X_test, y_test))

    expected = np.mean(expected_scores)
    actual = Validations.nested_cross_validation(model, X_reg, y_reg, param_grid, outer_splits=3, inner_splits=2)
    assert actual == pytest.approx(expected)


# --------------------------------------------------
# Return Type Tests
# --------------------------------------------------

def test_k_fold_validation_returns_float():
    actual = Validations.k_fold_validation(LinearRegression(), X_reg, y_reg)
    assert isinstance(actual, (float, np.floating))


def test_stratified_k_fold_validation_returns_float():
    actual = Validations.stratified_k_fold_validation(LogisticRegression(), X_class, y_class)
    assert isinstance(actual, (float, np.floating))


def test_leave_one_out_validation_returns_float():
    actual = Validations.leave_one_out_validation(LinearRegression(), X_reg, y_reg, scoring="neg_mean_absolute_error")
    assert isinstance(actual, (float, np.floating))


def test_repeated_k_fold_validation_returns_float():
    actual = Validations.repeated_k_fold_validation(LinearRegression(), X_reg, y_reg)
    assert isinstance(actual, (float, np.floating))


def test_nested_cross_validation_returns_float():
    actual = Validations.nested_cross_validation(Ridge(), X_reg, y_reg, {"alpha": [0.1, 1.0, 10.0]})
    assert isinstance(actual, float)