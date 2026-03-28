"""
Tests for regression evaluation and diagnostics (Phase 1).

Tests verify:
- evaluate_model returns correct metric keys and reasonable values
- compare_models produces valid comparison DataFrame
- cross_validate_model works correctly
"""

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression, Ridge

from src.regression import (
    build_linear_regression,
    evaluate_model,
    compare_models,
    cross_validate_model,
)


class TestEvaluateModel:
    """Tests for evaluate_model()."""

    def test_returns_correct_keys(self, simple_regression_data):
        X, y = simple_regression_data
        model = LinearRegression().fit(X[:160], y[:160])
        metrics = evaluate_model(model, X[160:], y[160:])
        expected_keys = {'mse', 'rmse', 'mae', 'r2'}
        assert set(metrics.keys()) == expected_keys

    def test_metrics_are_floats(self, simple_regression_data):
        X, y = simple_regression_data
        model = LinearRegression().fit(X[:160], y[:160])
        metrics = evaluate_model(model, X[160:], y[160:])
        for key, val in metrics.items():
            assert isinstance(val, float), f"{key} should be a float"

    def test_rmse_is_sqrt_of_mse(self, simple_regression_data):
        X, y = simple_regression_data
        model = LinearRegression().fit(X[:160], y[:160])
        metrics = evaluate_model(model, X[160:], y[160:])
        assert np.isclose(metrics['rmse'], np.sqrt(metrics['mse']), atol=1e-6)

    def test_mse_non_negative(self, simple_regression_data):
        X, y = simple_regression_data
        model = LinearRegression().fit(X[:160], y[:160])
        metrics = evaluate_model(model, X[160:], y[160:])
        assert metrics['mse'] >= 0
        assert metrics['mae'] >= 0

    def test_perfect_model(self):
        """A perfect model should have MSE=0 and R²=1."""
        X = np.array([[1], [2], [3], [4], [5]], dtype=float)
        y = np.array([2, 4, 6, 8, 10], dtype=float)
        model = LinearRegression().fit(X, y)
        metrics = evaluate_model(model, X, y)
        assert metrics['mse'] < 1e-10
        assert np.isclose(metrics['r2'], 1.0, atol=1e-6)


class TestCompareModels:
    """Tests for compare_models()."""

    def test_returns_dataframe(self, simple_regression_data):
        X, y = simple_regression_data
        models = {
            'Linear': LinearRegression().fit(X[:160], y[:160]),
            'Ridge': Ridge().fit(X[:160], y[:160]),
        }
        df = compare_models(models, X[160:], y[160:])
        assert isinstance(df, pd.DataFrame)

    def test_correct_number_of_rows(self, simple_regression_data):
        X, y = simple_regression_data
        models = {
            'Linear': LinearRegression().fit(X[:160], y[:160]),
            'Ridge': Ridge().fit(X[:160], y[:160]),
        }
        df = compare_models(models, X[160:], y[160:])
        assert df.shape[0] == 2

    def test_has_metric_columns(self, simple_regression_data):
        X, y = simple_regression_data
        models = {
            'Linear': LinearRegression().fit(X[:160], y[:160]),
        }
        df = compare_models(models, X[160:], y[160:])
        for col in ['mse', 'rmse', 'mae', 'r2']:
            assert col in df.columns, f"Missing column: {col}"

    def test_model_names_present(self, simple_regression_data):
        X, y = simple_regression_data
        models = {
            'MyLinear': LinearRegression().fit(X[:160], y[:160]),
            'MyRidge': Ridge().fit(X[:160], y[:160]),
        }
        df = compare_models(models, X[160:], y[160:])
        # Check model names are either in index or a column
        all_text = ' '.join(df.index.astype(str).tolist()) + ' '.join(
            df.values.astype(str).flatten().tolist()
        )
        # Model names should be present somewhere in the DataFrame
        assert 'MyLinear' in all_text or 'MyLinear' in df.index


class TestCrossValidateModel:
    """Tests for cross_validate_model()."""

    def test_returns_correct_keys(self, simple_regression_data):
        X, y = simple_regression_data
        results = cross_validate_model(LinearRegression(), X, y, cv=5)
        assert 'mean_score' in results
        assert 'std_score' in results
        assert 'scores' in results

    def test_correct_number_of_folds(self, simple_regression_data):
        X, y = simple_regression_data
        results = cross_validate_model(LinearRegression(), X, y, cv=3)
        assert len(results['scores']) == 3

    def test_five_fold_default(self, simple_regression_data):
        X, y = simple_regression_data
        results = cross_validate_model(LinearRegression(), X, y, cv=5)
        assert len(results['scores']) == 5

    def test_mean_is_average_of_scores(self, simple_regression_data):
        X, y = simple_regression_data
        results = cross_validate_model(LinearRegression(), X, y, cv=5)
        assert np.isclose(results['mean_score'], np.mean(results['scores']), atol=1e-10)

    def test_std_is_std_of_scores(self, simple_regression_data):
        X, y = simple_regression_data
        results = cross_validate_model(LinearRegression(), X, y, cv=5)
        assert np.isclose(results['std_score'], np.std(results['scores']), atol=1e-10)
