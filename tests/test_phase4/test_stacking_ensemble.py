"""
Tests for Stacking Ensemble and model persistence (Phase 4).

Tests verify:
- Stacking ensemble builds and predicts correctly
- Comparison of stacking vs voting is valid
- Models can be saved and loaded
"""

import os
import tempfile
import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import StackingRegressor
from sklearn.linear_model import LinearRegression

from src.ensemble import (
    build_stacking_ensemble,
    evaluate_stacking_vs_voting,
    save_model,
    load_model,
)


class TestBuildStackingEnsemble:
    """Tests for build_stacking_ensemble()."""

    def test_returns_stacking_regressor(self, simple_regression_data):
        X, y = simple_regression_data
        ensemble = build_stacking_ensemble(X, y)
        assert isinstance(ensemble, StackingRegressor)

    def test_is_fitted(self, simple_regression_data):
        X, y = simple_regression_data
        ensemble = build_stacking_ensemble(X, y)
        assert hasattr(ensemble, 'estimators_')

    def test_can_predict(self, simple_regression_data):
        X, y = simple_regression_data
        ensemble = build_stacking_ensemble(X, y)
        preds = ensemble.predict(X[:10])
        assert len(preds) == 10

    def test_predictions_finite(self, simple_regression_data):
        X, y = simple_regression_data
        ensemble = build_stacking_ensemble(X, y)
        preds = ensemble.predict(X[:10])
        assert np.all(np.isfinite(preds))

    def test_custom_meta_model(self, simple_regression_data):
        from sklearn.linear_model import Ridge
        X, y = simple_regression_data
        ensemble = build_stacking_ensemble(X, y, meta_model=Ridge(alpha=0.1))
        preds = ensemble.predict(X[:5])
        assert len(preds) == 5


class TestEvaluateStackingVsVoting:
    """Tests for evaluate_stacking_vs_voting()."""

    def test_returns_dataframe(self, simple_regression_data):
        X, y = simple_regression_data
        df = evaluate_stacking_vs_voting(X[:160], y[:160], X[160:], y[160:])
        assert isinstance(df, pd.DataFrame)

    def test_has_both_ensembles(self, simple_regression_data):
        X, y = simple_regression_data
        df = evaluate_stacking_vs_voting(X[:160], y[:160], X[160:], y[160:])
        model_names = df['model'].values if 'model' in df.columns else df.index.values
        model_str = ' '.join(str(m).lower() for m in model_names)
        assert 'stacking' in model_str, "Should include stacking ensemble"
        assert 'voting' in model_str, "Should include voting ensemble"

    def test_has_metric_columns(self, simple_regression_data):
        X, y = simple_regression_data
        df = evaluate_stacking_vs_voting(X[:160], y[:160], X[160:], y[160:])
        for col in ['mse', 'rmse', 'r2']:
            assert col in df.columns


class TestSaveLoadModel:
    """Tests for save_model() and load_model()."""

    def test_save_creates_file(self):
        model = LinearRegression().fit([[1], [2], [3]], [1, 2, 3])
        with tempfile.NamedTemporaryFile(suffix='.joblib', delete=False) as f:
            path = f.name
        try:
            result_path = save_model(model, path)
            assert os.path.exists(result_path)
        finally:
            os.unlink(path)

    def test_save_returns_path(self):
        model = LinearRegression().fit([[1], [2], [3]], [1, 2, 3])
        with tempfile.NamedTemporaryFile(suffix='.joblib', delete=False) as f:
            path = f.name
        try:
            result_path = save_model(model, path)
            assert result_path == path
        finally:
            os.unlink(path)

    def test_load_recovers_model(self):
        model = LinearRegression().fit([[1], [2], [3]], [1, 2, 3])
        with tempfile.NamedTemporaryFile(suffix='.joblib', delete=False) as f:
            path = f.name
        try:
            save_model(model, path)
            loaded = load_model(path)
            assert hasattr(loaded, 'predict')
        finally:
            os.unlink(path)

    def test_loaded_model_produces_same_predictions(self):
        X = np.array([[1], [2], [3], [4], [5]], dtype=float)
        y = np.array([2, 4, 6, 8, 10], dtype=float)
        model = LinearRegression().fit(X, y)
        original_preds = model.predict(X)

        with tempfile.NamedTemporaryFile(suffix='.joblib', delete=False) as f:
            path = f.name
        try:
            save_model(model, path)
            loaded = load_model(path)
            loaded_preds = loaded.predict(X)
            assert np.allclose(original_preds, loaded_preds)
        finally:
            os.unlink(path)
