"""
Tests for data loading and preprocessing (Phase 1).

Tests verify:
- Dataset loads correctly with expected shape and columns
- Feature scaling works properly
- Train/test split produces correct proportions
- Feature engineering creates expected new columns
"""

import numpy as np
import pandas as pd
import pytest

from src.data_loader import (
    load_housing_data,
    preprocess_features,
    split_data,
    create_feature_engineering,
)


class TestLoadHousingData:
    """Tests for load_housing_data()."""

    def test_returns_dataframe(self):
        df = load_housing_data()
        assert isinstance(df, pd.DataFrame), "Should return a pandas DataFrame"

    def test_has_target_column(self):
        df = load_housing_data()
        assert 'MedHouseVal' in df.columns, "DataFrame must contain 'MedHouseVal' column"

    def test_correct_number_of_columns(self):
        df = load_housing_data()
        assert df.shape[1] == 9, "Expected 8 features + 1 target = 9 columns"

    def test_no_missing_values(self):
        df = load_housing_data()
        assert df.isnull().sum().sum() == 0, "California Housing should have no missing values"

    def test_reasonable_row_count(self):
        df = load_housing_data()
        assert df.shape[0] > 20000, "California Housing has ~20,640 rows"


class TestPreprocessFeatures:
    """Tests for preprocess_features()."""

    def test_returns_correct_types(self):
        df = load_housing_data()
        X_scaled, y, feature_names, scaler = preprocess_features(df)
        assert isinstance(X_scaled, np.ndarray), "X_scaled should be np.ndarray"
        assert isinstance(y, np.ndarray), "y should be np.ndarray"
        assert isinstance(feature_names, list), "feature_names should be a list"

    def test_features_are_scaled(self):
        df = load_housing_data()
        X_scaled, y, names, scaler = preprocess_features(df)
        # After standard scaling, means should be approximately 0
        means = np.abs(X_scaled.mean(axis=0))
        assert means.max() < 1e-6, "Scaled feature means should be ~0"

    def test_features_have_unit_variance(self):
        df = load_housing_data()
        X_scaled, y, names, scaler = preprocess_features(df)
        stds = X_scaled.std(axis=0)
        assert np.allclose(stds, 1.0, atol=0.05), "Scaled feature stds should be ~1"

    def test_target_not_in_features(self):
        df = load_housing_data()
        X_scaled, y, names, scaler = preprocess_features(df)
        assert 'MedHouseVal' not in names, "Target should not be in feature names"
        assert X_scaled.shape[1] == len(names)

    def test_target_values_reasonable(self):
        df = load_housing_data()
        _, y, _, _ = preprocess_features(df)
        assert y.min() >= 0, "House values should be non-negative"
        assert y.max() <= 10, "California Housing values are in 100k units, max ~5"


class TestSplitData:
    """Tests for split_data()."""

    def test_default_split_proportions(self):
        X = np.random.rand(100, 5)
        y = np.random.rand(100)
        X_train, X_test, y_train, y_test = split_data(X, y)
        assert len(X_train) == 80, "Default 80% train"
        assert len(X_test) == 20, "Default 20% test"

    def test_custom_split_proportions(self):
        X = np.random.rand(100, 5)
        y = np.random.rand(100)
        X_train, X_test, y_train, y_test = split_data(X, y, test_size=0.3)
        assert len(X_test) == 30

    def test_reproducibility(self):
        X = np.random.rand(100, 5)
        y = np.random.rand(100)
        split1 = split_data(X, y, random_state=42)
        split2 = split_data(X, y, random_state=42)
        assert np.array_equal(split1[0], split2[0]), "Same seed should give same split"

    def test_shapes_consistent(self):
        X = np.random.rand(100, 5)
        y = np.random.rand(100)
        X_train, X_test, y_train, y_test = split_data(X, y)
        assert X_train.shape[1] == 5
        assert X_test.shape[1] == 5
        assert len(y_train) == len(X_train)
        assert len(y_test) == len(X_test)


class TestFeatureEngineering:
    """Tests for create_feature_engineering()."""

    def test_new_columns_created(self):
        df = load_housing_data()
        df_eng = create_feature_engineering(df)
        assert 'rooms_per_household' in df_eng.columns
        assert 'bedrooms_ratio' in df_eng.columns
        assert 'population_density' in df_eng.columns

    def test_more_columns_than_original(self):
        df = load_housing_data()
        df_eng = create_feature_engineering(df)
        assert df_eng.shape[1] > df.shape[1], "Engineered df should have more columns"

    def test_original_columns_preserved(self):
        df = load_housing_data()
        df_eng = create_feature_engineering(df)
        for col in df.columns:
            assert col in df_eng.columns, f"Original column '{col}' should be preserved"

    def test_no_infinite_values(self):
        df = load_housing_data()
        df_eng = create_feature_engineering(df)
        assert not np.isinf(df_eng.values).any(), "No infinite values allowed"

    def test_original_unchanged(self):
        df = load_housing_data()
        original_shape = df.shape
        _ = create_feature_engineering(df)
        assert df.shape == original_shape, "Original DataFrame should not be modified"
