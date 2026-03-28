"""
Tests for tree-based and boosting regression models (Phase 1).

Tests verify:
- Decision Tree, Random Forest, Gradient Boosting, XGBoost, LightGBM
- Models train, predict, and achieve reasonable performance
- Feature importances are available
"""

import numpy as np
import pytest
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

from src.regression import (
    build_decision_tree,
    build_random_forest,
    build_gradient_boosting,
    build_xgboost,
    build_lightgbm,
    evaluate_model,
)


class TestDecisionTree:
    """Tests for build_decision_tree()."""

    def test_returns_fitted_model(self, simple_regression_data):
        X, y = simple_regression_data
        model = build_decision_tree(X, y, max_depth=5)
        assert isinstance(model, DecisionTreeRegressor)
        assert hasattr(model, 'tree_'), "Model must be fitted"

    def test_max_depth_respected(self, simple_regression_data):
        X, y = simple_regression_data
        model = build_decision_tree(X, y, max_depth=3)
        assert model.get_depth() <= 3

    def test_has_feature_importances(self, simple_regression_data):
        X, y = simple_regression_data
        model = build_decision_tree(X, y)
        assert hasattr(model, 'feature_importances_')
        assert len(model.feature_importances_) == X.shape[1]

    def test_predictions_shape(self, simple_regression_data):
        X, y = simple_regression_data
        model = build_decision_tree(X, y)
        preds = model.predict(X[:10])
        assert preds.shape == (10,)


class TestRandomForest:
    """Tests for build_random_forest()."""

    def test_returns_fitted_model(self, simple_regression_data):
        X, y = simple_regression_data
        model = build_random_forest(X, y, n_estimators=50)
        assert isinstance(model, RandomForestRegressor)
        assert hasattr(model, 'estimators_'), "Model must be fitted"

    def test_correct_n_estimators(self, simple_regression_data):
        X, y = simple_regression_data
        model = build_random_forest(X, y, n_estimators=30)
        assert len(model.estimators_) == 30

    def test_feature_importances(self, simple_regression_data):
        X, y = simple_regression_data
        model = build_random_forest(X, y, n_estimators=50)
        importances = model.feature_importances_
        assert len(importances) == X.shape[1]
        assert np.isclose(importances.sum(), 1.0, atol=1e-5)

    def test_outperforms_single_tree(self, train_test_housing):
        X_train, X_test, y_train, y_test = train_test_housing
        tree = build_decision_tree(X_train, y_train, max_depth=10)
        forest = build_random_forest(X_train, y_train, n_estimators=100)
        tree_r2 = evaluate_model(tree, X_test, y_test)['r2']
        forest_r2 = evaluate_model(forest, X_test, y_test)['r2']
        assert forest_r2 > tree_r2, "Random Forest should outperform a single tree"

    def test_reasonable_performance(self, train_test_housing):
        X_train, X_test, y_train, y_test = train_test_housing
        model = build_random_forest(X_train, y_train, n_estimators=100)
        metrics = evaluate_model(model, X_test, y_test)
        assert metrics['r2'] > 0.7, "Random Forest should achieve R² > 0.7 on housing"


class TestGradientBoosting:
    """Tests for build_gradient_boosting()."""

    def test_returns_fitted_model(self, simple_regression_data):
        X, y = simple_regression_data
        model = build_gradient_boosting(X, y, n_estimators=50)
        assert isinstance(model, GradientBoostingRegressor)
        assert hasattr(model, 'estimators_')

    def test_feature_importances(self, simple_regression_data):
        X, y = simple_regression_data
        model = build_gradient_boosting(X, y, n_estimators=50)
        assert len(model.feature_importances_) == X.shape[1]

    def test_reasonable_performance(self, train_test_housing):
        X_train, X_test, y_train, y_test = train_test_housing
        model = build_gradient_boosting(X_train, y_train, n_estimators=100)
        metrics = evaluate_model(model, X_test, y_test)
        assert metrics['r2'] > 0.7


class TestXGBoost:
    """Tests for build_xgboost()."""

    def test_returns_fitted_model(self, simple_regression_data):
        X, y = simple_regression_data
        model = build_xgboost(X, y, n_estimators=50)
        assert hasattr(model, 'predict')
        preds = model.predict(X[:5])
        assert len(preds) == 5

    def test_reasonable_performance(self, train_test_housing):
        X_train, X_test, y_train, y_test = train_test_housing
        model = build_xgboost(X_train, y_train, n_estimators=100)
        metrics = evaluate_model(model, X_test, y_test)
        assert metrics['r2'] > 0.7, "XGBoost should achieve R² > 0.7"


class TestLightGBM:
    """Tests for build_lightgbm()."""

    def test_returns_fitted_model(self, simple_regression_data):
        X, y = simple_regression_data
        model = build_lightgbm(X, y, n_estimators=50)
        assert hasattr(model, 'predict')
        preds = model.predict(X[:5])
        assert len(preds) == 5

    def test_reasonable_performance(self, train_test_housing):
        X_train, X_test, y_train, y_test = train_test_housing
        model = build_lightgbm(X_train, y_train, n_estimators=100)
        metrics = evaluate_model(model, X_test, y_test)
        assert metrics['r2'] > 0.7, "LightGBM should achieve R² > 0.7"
