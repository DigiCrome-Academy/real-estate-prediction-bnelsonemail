"""
Phase 1: Regression Modeling Module

This module contains functions for building and evaluating various regression models
for real estate price prediction.

Models to implement:
- Linear Regression (baseline)
- Ridge Regression (L2 regularization)
- Lasso Regression (L1 regularization)
- ElasticNet (combined L1 + L2)
- Polynomial Regression
- Decision Tree Regressor
- Random Forest Regressor
- Gradient Boosting (XGBoost, LightGBM)
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.preprocessing import PolynomialFeatures
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import cross_val_score


# =============================================================================
# Section 1: Model Building Functions
# =============================================================================

def build_linear_regression(X_train, y_train):
    """
    Build and train a basic Linear Regression model.

    Args:
        X_train (np.ndarray): Training features.
        y_train (np.ndarray): Training target values.

    Returns:
        LinearRegression: Fitted model.

    Example:
        >>> from sklearn.datasets import make_regression
        >>> X, y = make_regression(n_samples=100, n_features=5, random_state=42)
        >>> model = build_linear_regression(X, y)
        >>> hasattr(model, 'coef_')
        True
    """
    # TODO: Implement this function
    raise NotImplementedError("Implement build_linear_regression()")


def build_ridge_regression(X_train, y_train, alpha=1.0):
    """
    Build and train a Ridge Regression model.

    Args:
        X_train (np.ndarray): Training features.
        y_train (np.ndarray): Training target values.
        alpha (float): Regularization strength.

    Returns:
        Ridge: Fitted model.
    """
    # TODO: Implement this function
    raise NotImplementedError("Implement build_ridge_regression()")


def build_lasso_regression(X_train, y_train, alpha=1.0):
    """
    Build and train a Lasso Regression model.

    Args:
        X_train (np.ndarray): Training features.
        y_train (np.ndarray): Training target values.
        alpha (float): Regularization strength.

    Returns:
        Lasso: Fitted model.
    """
    # TODO: Implement this function
    raise NotImplementedError("Implement build_lasso_regression()")


def build_elasticnet_regression(X_train, y_train, alpha=1.0, l1_ratio=0.5):
    """
    Build and train an ElasticNet Regression model.

    Args:
        X_train (np.ndarray): Training features.
        y_train (np.ndarray): Training target values.
        alpha (float): Regularization strength.
        l1_ratio (float): Mix ratio between L1 and L2 (0=Ridge, 1=Lasso).

    Returns:
        ElasticNet: Fitted model.
    """
    # TODO: Implement this function
    raise NotImplementedError("Implement build_elasticnet_regression()")


def build_polynomial_regression(X_train, y_train, degree=2):
    """
    Build a Polynomial Regression model.

    This should create polynomial features and fit a LinearRegression on them.

    Args:
        X_train (np.ndarray): Training features.
        y_train (np.ndarray): Training target values.
        degree (int): Polynomial degree.

    Returns:
        tuple: (model, poly_transformer)
            - model (LinearRegression): Fitted linear model on polynomial features
            - poly_transformer (PolynomialFeatures): Fitted polynomial transformer

    Example:
        >>> X = np.random.rand(100, 3)
        >>> y = np.random.rand(100)
        >>> model, poly = build_polynomial_regression(X, y, degree=2)
        >>> poly.degree == 2
        True
        >>> hasattr(model, 'coef_')
        True
    """
    # TODO: Implement this function
    # Hints:
    #   1. Create PolynomialFeatures(degree=degree, include_bias=False)
    #   2. Transform X_train using fit_transform
    #   3. Fit LinearRegression on the transformed features
    #   4. Return both the model and the transformer
    raise NotImplementedError("Implement build_polynomial_regression()")


def build_decision_tree(X_train, y_train, max_depth=10, random_state=42):
    """
    Build and train a Decision Tree Regressor.

    Args:
        X_train (np.ndarray): Training features.
        y_train (np.ndarray): Training target values.
        max_depth (int): Maximum tree depth.
        random_state (int): Random seed.

    Returns:
        DecisionTreeRegressor: Fitted model.
    """
    # TODO: Implement this function
    raise NotImplementedError("Implement build_decision_tree()")


def build_random_forest(X_train, y_train, n_estimators=100, max_depth=None, random_state=42):
    """
    Build and train a Random Forest Regressor.

    Args:
        X_train (np.ndarray): Training features.
        y_train (np.ndarray): Training target values.
        n_estimators (int): Number of trees.
        max_depth (int or None): Maximum tree depth.
        random_state (int): Random seed.

    Returns:
        RandomForestRegressor: Fitted model.
    """
    # TODO: Implement this function
    raise NotImplementedError("Implement build_random_forest()")


def build_gradient_boosting(X_train, y_train, n_estimators=100, learning_rate=0.1, random_state=42):
    """
    Build and train a Gradient Boosting Regressor (sklearn implementation).

    Args:
        X_train (np.ndarray): Training features.
        y_train (np.ndarray): Training target values.
        n_estimators (int): Number of boosting stages.
        learning_rate (float): Shrinkage rate.
        random_state (int): Random seed.

    Returns:
        GradientBoostingRegressor: Fitted model.
    """
    # TODO: Implement this function
    raise NotImplementedError("Implement build_gradient_boosting()")


def build_xgboost(X_train, y_train, n_estimators=100, learning_rate=0.1, random_state=42):
    """
    Build and train an XGBoost Regressor.

    Args:
        X_train (np.ndarray): Training features.
        y_train (np.ndarray): Training target values.
        n_estimators (int): Number of boosting rounds.
        learning_rate (float): Step size shrinkage.
        random_state (int): Random seed.

    Returns:
        xgboost.XGBRegressor: Fitted model.
    """
    # TODO: Implement this function
    # Hint: import xgboost as xgb; use xgb.XGBRegressor
    raise NotImplementedError("Implement build_xgboost()")


def build_lightgbm(X_train, y_train, n_estimators=100, learning_rate=0.1, random_state=42):
    """
    Build and train a LightGBM Regressor.

    Args:
        X_train (np.ndarray): Training features.
        y_train (np.ndarray): Training target values.
        n_estimators (int): Number of boosting rounds.
        learning_rate (float): Step size shrinkage.
        random_state (int): Random seed.

    Returns:
        lightgbm.LGBMRegressor: Fitted model.
    """
    # TODO: Implement this function
    # Hint: import lightgbm as lgb; use lgb.LGBMRegressor(verbose=-1)
    raise NotImplementedError("Implement build_lightgbm()")


# =============================================================================
# Section 2: Evaluation Functions
# =============================================================================

def evaluate_model(model, X_test, y_test):
    """
    Evaluate a regression model and return standard metrics.

    Args:
        model: Fitted sklearn-compatible regression model.
        X_test (np.ndarray): Test features.
        y_test (np.ndarray): Test target values.

    Returns:
        dict: Dictionary with keys 'mse', 'rmse', 'mae', 'r2' and their float values.

    Example:
        >>> from sklearn.linear_model import LinearRegression
        >>> from sklearn.datasets import make_regression
        >>> X, y = make_regression(n_samples=100, n_features=5, random_state=42)
        >>> model = LinearRegression().fit(X[:80], y[:80])
        >>> metrics = evaluate_model(model, X[80:], y[80:])
        >>> set(metrics.keys()) == {'mse', 'rmse', 'mae', 'r2'}
        True
        >>> metrics['r2'] > 0
        True
    """
    # TODO: Implement this function
    # Hints:
    #   1. Generate predictions using model.predict(X_test)
    #   2. Calculate MSE, RMSE (sqrt of MSE), MAE, and R²
    #   3. Return as a dictionary
    raise NotImplementedError("Implement evaluate_model()")


def compare_models(models_dict, X_test, y_test):
    """
    Compare multiple models and return a DataFrame of their metrics.

    Args:
        models_dict (dict): {model_name: fitted_model} dictionary.
        X_test (np.ndarray): Test features.
        y_test (np.ndarray): Test target values.

    Returns:
        pd.DataFrame: DataFrame with model names as index, metric columns.

    Example:
        >>> from sklearn.linear_model import LinearRegression, Ridge
        >>> from sklearn.datasets import make_regression
        >>> X, y = make_regression(n_samples=100, n_features=5, random_state=42)
        >>> models = {
        ...     'Linear': LinearRegression().fit(X[:80], y[:80]),
        ...     'Ridge': Ridge().fit(X[:80], y[:80])
        ... }
        >>> df = compare_models(models, X[80:], y[80:])
        >>> df.shape[0] == 2
        True
        >>> 'rmse' in df.columns
        True
    """
    # TODO: Implement this function
    # Hints:
    #   1. Loop through models_dict
    #   2. Call evaluate_model for each
    #   3. Collect results into a DataFrame
    raise NotImplementedError("Implement compare_models()")


def cross_validate_model(model, X, y, cv=5, scoring='neg_mean_squared_error'):
    """
    Perform k-fold cross-validation on a model.

    Args:
        model: Unfitted sklearn-compatible regression model (or pipeline).
        X (np.ndarray): Full feature matrix.
        y (np.ndarray): Full target vector.
        cv (int): Number of cross-validation folds.
        scoring (str): Scoring metric.

    Returns:
        dict: Dictionary with 'mean_score', 'std_score', 'scores' (all folds).

    Example:
        >>> from sklearn.linear_model import LinearRegression
        >>> from sklearn.datasets import make_regression
        >>> X, y = make_regression(n_samples=200, n_features=5, random_state=42)
        >>> results = cross_validate_model(LinearRegression(), X, y, cv=5)
        >>> len(results['scores']) == 5
        True
        >>> 'mean_score' in results
        True
    """
    # TODO: Implement this function
    # Hints:
    #   1. Use cross_val_score from sklearn
    #   2. Return mean, std, and all individual fold scores
    raise NotImplementedError("Implement cross_validate_model()")
