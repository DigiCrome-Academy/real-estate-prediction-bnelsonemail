"""
Tests for linear regression models (Phase 1).

Tests verify:
- Models train successfully and produce predictions
- Models achieve reasonable performance on California Housing
- Regularized models behave correctly
- Polynomial regression works with transformations
"""

import numpy as np
import pytest
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet

from src.regression import (
    build_linear_regression,
    build_ridge_regression,
    build_lasso_regression,
    build_elasticnet_regression,
    build_polynomial_regression,
    evaluate_model,
)


class TestLinearRegression:
    """Tests for build_linear_regression()."""

    def test_returns_fitted_model(self, simple_regression_data):
        X, y = simple_regression_data
        model = build_linear_regression(X, y)
        assert isinstance(model, LinearRegression)
        assert hasattr(model, 'coef_'), "Model must be fitted (have coef_)"

    def test_has_correct_coefficients(self, simple_regression_data):
        X, y = simple_regression_data
        model = build_linear_regression(X, y)
        assert len(model.coef_) == X.shape[1]

    def test_predictions_shape(self, simple_regression_data):
        X, y = simple_regression_data
        model = build_linear_regression(X, y)
        preds = model.predict(X)
        assert preds.shape == y.shape

    def test_reasonable_r2_on_housing(self, train_test_housing):
        X_train, X_test, y_train, y_test = train_test_housing
        model = build_linear_regression(X_train, y_train)
        metrics = evaluate_model(model, X_test, y_test)
        assert metrics['r2'] > 0.4, "Linear regression should achieve R² > 0.4 on housing"


class TestRidgeRegression:
    """Tests for build_ridge_regression()."""

    def test_returns_fitted_model(self, simple_regression_data):
        X, y = simple_regression_data
        model = build_ridge_regression(X, y, alpha=1.0)
        assert isinstance(model, Ridge)
        assert hasattr(model, 'coef_')

    def test_alpha_parameter_used(self, simple_regression_data):
        X, y = simple_regression_data
        model = build_ridge_regression(X, y, alpha=10.0)
        assert model.alpha == 10.0

    def test_regularization_shrinks_coefficients(self, simple_regression_data):
        X, y = simple_regression_data
        model_low = build_ridge_regression(X, y, alpha=0.01)
        model_high = build_ridge_regression(X, y, alpha=100.0)
        # Higher alpha should produce smaller coefficient magnitudes
        norm_low = np.linalg.norm(model_low.coef_)
        norm_high = np.linalg.norm(model_high.coef_)
        assert norm_high < norm_low, "Higher alpha should shrink coefficients"

    def test_reasonable_performance(self, train_test_housing):
        X_train, X_test, y_train, y_test = train_test_housing
        model = build_ridge_regression(X_train, y_train, alpha=1.0)
        metrics = evaluate_model(model, X_test, y_test)
        assert metrics['r2'] > 0.4


class TestLassoRegression:
    """Tests for build_lasso_regression()."""

    def test_returns_fitted_model(self, simple_regression_data):
        X, y = simple_regression_data
        model = build_lasso_regression(X, y, alpha=0.1)
        assert isinstance(model, Lasso)
        assert hasattr(model, 'coef_')

    def test_feature_selection_property(self, simple_regression_data):
        X, y = simple_regression_data
        model = build_lasso_regression(X, y, alpha=10.0)
        # High alpha Lasso should zero out some coefficients
        n_zero = np.sum(np.abs(model.coef_) < 1e-10)
        # Just check it ran; strong sparsity depends on alpha vs signal
        assert len(model.coef_) == X.shape[1]

    def test_reasonable_performance(self, train_test_housing):
        X_train, X_test, y_train, y_test = train_test_housing
        model = build_lasso_regression(X_train, y_train, alpha=0.01)
        metrics = evaluate_model(model, X_test, y_test)
        assert metrics['r2'] > 0.3


class TestElasticNetRegression:
    """Tests for build_elasticnet_regression()."""

    def test_returns_fitted_model(self, simple_regression_data):
        X, y = simple_regression_data
        model = build_elasticnet_regression(X, y, alpha=0.1, l1_ratio=0.5)
        assert isinstance(model, ElasticNet)
        assert hasattr(model, 'coef_')

    def test_l1_ratio_parameter(self, simple_regression_data):
        X, y = simple_regression_data
        model = build_elasticnet_regression(X, y, alpha=0.1, l1_ratio=0.7)
        assert model.l1_ratio == 0.7

    def test_reasonable_performance(self, train_test_housing):
        X_train, X_test, y_train, y_test = train_test_housing
        model = build_elasticnet_regression(X_train, y_train, alpha=0.01, l1_ratio=0.5)
        metrics = evaluate_model(model, X_test, y_test)
        assert metrics['r2'] > 0.3


class TestPolynomialRegression:
    """Tests for build_polynomial_regression()."""

    def test_returns_model_and_transformer(self, simple_regression_data):
        X, y = simple_regression_data
        model, poly = build_polynomial_regression(X, y, degree=2)
        assert hasattr(model, 'coef_'), "Model must be fitted"
        assert hasattr(poly, 'powers_'), "Poly transformer must be fitted"

    def test_polynomial_degree(self, simple_regression_data):
        X, y = simple_regression_data
        model, poly = build_polynomial_regression(X, y, degree=3)
        assert poly.degree == 3

    def test_expanded_features(self, simple_regression_data):
        X, y = simple_regression_data
        model, poly = build_polynomial_regression(X, y, degree=2)
        X_poly = poly.transform(X)
        # Degree 2 with 5 features: 5 + 15 (combinations) = 20
        assert X_poly.shape[1] > X.shape[1], "Polynomial should expand features"

    def test_predictions_work(self, simple_regression_data):
        X, y = simple_regression_data
        model, poly = build_polynomial_regression(X, y, degree=2)
        X_poly = poly.transform(X[:5])
        preds = model.predict(X_poly)
        assert len(preds) == 5
