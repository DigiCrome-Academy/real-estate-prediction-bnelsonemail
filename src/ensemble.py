"""
Phase 4: Model Ensemble Module

This module implements ensemble methods that combine multiple regression models
for improved prediction performance.

Methods to implement:
- Voting Regressor (averaging multiple models)
- Stacking Regressor (meta-learner on top of base models)
"""

import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import (
    VotingRegressor,
    StackingRegressor,
    RandomForestRegressor,
    GradientBoostingRegressor,
)
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score


# =============================================================================
# Section 1: Voting Ensemble
# =============================================================================

def build_voting_ensemble(X_train, y_train, models=None):
    """
    Build and train a Voting Regressor ensemble.

    The VotingRegressor averages predictions from multiple base models.

    Args:
        X_train (np.ndarray): Training features.
        y_train (np.ndarray): Training target values.
        models (list[tuple] or None): List of (name, estimator) tuples.
            If None, use a default set of:
              - ('ridge', Ridge(alpha=1.0))
              - ('rf', RandomForestRegressor(n_estimators=100, random_state=42))
              - ('gb', GradientBoostingRegressor(n_estimators=100, random_state=42))

    Returns:
        VotingRegressor: Fitted voting ensemble.

    Example:
        >>> from sklearn.datasets import make_regression
        >>> X, y = make_regression(n_samples=200, n_features=5, random_state=42)
        >>> ensemble = build_voting_ensemble(X, y)
        >>> hasattr(ensemble, 'predict')
        True
        >>> preds = ensemble.predict(X[:5])
        >>> len(preds) == 5
        True
    """
    if models is None:
        models = [
            ('ridge', Ridge(alpha=1.0)),
            ('rf', RandomForestRegressor(n_estimators=100, random_state=42)),
            ('gb', GradientBoostingRegressor(n_estimators=100, random_state=42)),
        ]
    ensemble = VotingRegressor(estimators=models)
    ensemble.fit(X_train, y_train)
    return ensemble


def evaluate_voting_vs_individual(X_train, y_train, X_test, y_test, models=None):
    """
    Compare the voting ensemble against each individual model.

    Args:
        X_train, y_train: Training data.
        X_test, y_test: Test data.
        models (list[tuple] or None): Same as build_voting_ensemble.

    Returns:
        pd.DataFrame: Comparison table with columns ['model', 'mse', 'rmse', 'r2'].
            Includes a row for each individual model AND the ensemble.

    Example:
        >>> from sklearn.datasets import make_regression
        >>> X, y = make_regression(n_samples=300, n_features=5, random_state=42)
        >>> df = evaluate_voting_vs_individual(X[:240], y[:240], X[240:], y[240:])
        >>> 'VotingEnsemble' in df['model'].values
        True
        >>> df.shape[0] >= 4  # at least 3 individual + 1 ensemble
        True
    """
    if models is None:
        models = [
            ('ridge', Ridge(alpha=1.0)),
            ('rf', RandomForestRegressor(n_estimators=100, random_state=42)),
            ('gb', GradientBoostingRegressor(n_estimators=100, random_state=42)),
        ]

    rows = []
    for name, estimator in models:
        estimator.fit(X_train, y_train)
        preds = estimator.predict(X_test)
        mse = mean_squared_error(y_test, preds)
        rows.append({'model': name, 'mse': mse, 'rmse': np.sqrt(mse), 'r2': r2_score(y_test, preds)})

    ensemble = build_voting_ensemble(X_train, y_train, models=models)
    preds = ensemble.predict(X_test)
    mse = mean_squared_error(y_test, preds)
    rows.append({'model': 'VotingEnsemble', 'mse': mse, 'rmse': np.sqrt(mse), 'r2': r2_score(y_test, preds)})

    return pd.DataFrame(rows, columns=['model', 'mse', 'rmse', 'r2'])


# =============================================================================
# Section 2: Stacking Ensemble
# =============================================================================

def build_stacking_ensemble(X_train, y_train, base_models=None, meta_model=None):
    """
    Build and train a Stacking Regressor ensemble.

    Stacking uses a meta-learner to combine predictions from base models.

    Args:
        X_train (np.ndarray): Training features.
        y_train (np.ndarray): Training target values.
        base_models (list[tuple] or None): List of (name, estimator) tuples for base layer.
            If None, use:
              - ('ridge', Ridge(alpha=1.0))
              - ('rf', RandomForestRegressor(n_estimators=100, random_state=42))
              - ('gb', GradientBoostingRegressor(n_estimators=100, random_state=42))
        meta_model (estimator or None): Meta-learner model.
            If None, use LinearRegression().

    Returns:
        StackingRegressor: Fitted stacking ensemble.

    Example:
        >>> from sklearn.datasets import make_regression
        >>> X, y = make_regression(n_samples=200, n_features=5, random_state=42)
        >>> ensemble = build_stacking_ensemble(X, y)
        >>> hasattr(ensemble, 'predict')
        True
    """
    if base_models is None:
        base_models = [
            ('ridge', Ridge(alpha=1.0)),
            ('rf', RandomForestRegressor(n_estimators=100, random_state=42)),
            ('gb', GradientBoostingRegressor(n_estimators=100, random_state=42)),
        ]
    if meta_model is None:
        meta_model = LinearRegression()
    ensemble = StackingRegressor(estimators=base_models, final_estimator=meta_model, cv=5)
    ensemble.fit(X_train, y_train)
    return ensemble


def evaluate_stacking_vs_voting(X_train, y_train, X_test, y_test):
    """
    Compare stacking ensemble vs voting ensemble vs individual models.

    Args:
        X_train, y_train: Training data.
        X_test, y_test: Test data.

    Returns:
        pd.DataFrame: Comparison table with columns ['model', 'mse', 'rmse', 'r2'].
            Includes rows for individual models, voting ensemble, and stacking ensemble.

    Example:
        >>> from sklearn.datasets import make_regression
        >>> X, y = make_regression(n_samples=300, n_features=5, random_state=42)
        >>> df = evaluate_stacking_vs_voting(X[:240], y[:240], X[240:], y[240:])
        >>> 'StackingEnsemble' in df['model'].values
        True
        >>> 'VotingEnsemble' in df['model'].values
        True
    """
    df = evaluate_voting_vs_individual(X_train, y_train, X_test, y_test)

    ensemble = build_stacking_ensemble(X_train, y_train)
    preds = ensemble.predict(X_test)
    mse = mean_squared_error(y_test, preds)
    stacking_row = pd.DataFrame([{
        'model': 'StackingEnsemble',
        'mse': mse,
        'rmse': np.sqrt(mse),
        'r2': r2_score(y_test, preds),
    }])

    return pd.concat([df, stacking_row], ignore_index=True)


# =============================================================================
# Section 3: Model Persistence
# =============================================================================

def save_model(model, filepath):
    """
    Save a trained model to disk using joblib.

    Args:
        model: Fitted sklearn-compatible model.
        filepath (str): Path to save the model (.joblib extension recommended).

    Returns:
        str: The filepath where the model was saved.

    Example:
        >>> from sklearn.linear_model import LinearRegression
        >>> import tempfile, os
        >>> model = LinearRegression().fit([[1],[2],[3]], [1,2,3])
        >>> path = save_model(model, tempfile.mktemp(suffix='.joblib'))
        >>> os.path.exists(path)
        True
    """
    joblib.dump(model, filepath)
    return filepath


def load_model(filepath):
    """
    Load a trained model from disk.

    Args:
        filepath (str): Path to the saved model file.

    Returns:
        Loaded model object.

    Example:
        >>> from sklearn.linear_model import LinearRegression
        >>> import tempfile
        >>> model = LinearRegression().fit([[1],[2],[3]], [1,2,3])
        >>> path = save_model(model, tempfile.mktemp(suffix='.joblib'))
        >>> loaded = load_model(path)
        >>> loaded.predict([[4]])
        array([4.])
    """
    return joblib.load(filepath)
