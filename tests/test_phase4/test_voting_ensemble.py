"""
Tests for Voting Ensemble (Phase 4).

Tests verify:
- Voting ensemble builds and predicts correctly
- Comparison against individual models is valid
"""

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import VotingRegressor

from src.ensemble import build_voting_ensemble, evaluate_voting_vs_individual


class TestBuildVotingEnsemble:
    """Tests for build_voting_ensemble()."""

    def test_returns_voting_regressor(self, simple_regression_data):
        X, y = simple_regression_data
        ensemble = build_voting_ensemble(X, y)
        assert isinstance(ensemble, VotingRegressor)

    def test_is_fitted(self, simple_regression_data):
        X, y = simple_regression_data
        ensemble = build_voting_ensemble(X, y)
        # A fitted VotingRegressor has estimators_
        assert hasattr(ensemble, 'estimators_')

    def test_can_predict(self, simple_regression_data):
        X, y = simple_regression_data
        ensemble = build_voting_ensemble(X, y)
        preds = ensemble.predict(X[:10])
        assert len(preds) == 10

    def test_predictions_are_finite(self, simple_regression_data):
        X, y = simple_regression_data
        ensemble = build_voting_ensemble(X, y)
        preds = ensemble.predict(X[:10])
        assert np.all(np.isfinite(preds))

    def test_custom_models(self, simple_regression_data):
        from sklearn.linear_model import LinearRegression, Ridge
        X, y = simple_regression_data
        custom_models = [
            ('lr', LinearRegression()),
            ('ridge', Ridge(alpha=0.5)),
        ]
        ensemble = build_voting_ensemble(X, y, models=custom_models)
        assert isinstance(ensemble, VotingRegressor)
        preds = ensemble.predict(X[:5])
        assert len(preds) == 5

    def test_default_has_three_models(self, simple_regression_data):
        X, y = simple_regression_data
        ensemble = build_voting_ensemble(X, y)
        assert len(ensemble.estimators_) >= 3


class TestEvaluateVotingVsIndividual:
    """Tests for evaluate_voting_vs_individual()."""

    def test_returns_dataframe(self, simple_regression_data):
        X, y = simple_regression_data
        df = evaluate_voting_vs_individual(X[:160], y[:160], X[160:], y[160:])
        assert isinstance(df, pd.DataFrame)

    def test_has_ensemble_row(self, simple_regression_data):
        X, y = simple_regression_data
        df = evaluate_voting_vs_individual(X[:160], y[:160], X[160:], y[160:])
        model_names = df['model'].values if 'model' in df.columns else df.index.values
        model_str = ' '.join(str(m) for m in model_names)
        assert 'Voting' in model_str or 'voting' in model_str or 'ensemble' in model_str.lower()

    def test_has_metric_columns(self, simple_regression_data):
        X, y = simple_regression_data
        df = evaluate_voting_vs_individual(X[:160], y[:160], X[160:], y[160:])
        for col in ['mse', 'rmse', 'r2']:
            assert col in df.columns, f"Missing column: {col}"

    def test_at_least_four_rows(self, simple_regression_data):
        X, y = simple_regression_data
        df = evaluate_voting_vs_individual(X[:160], y[:160], X[160:], y[160:])
        assert len(df) >= 4, "Need at least 3 individual + 1 ensemble"
