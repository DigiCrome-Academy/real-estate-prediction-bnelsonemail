"""
Shared test fixtures for all phases.

These fixtures provide consistent test data across all test modules.
"""

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import fetch_california_housing, make_regression, make_blobs


@pytest.fixture(scope="session")
def california_housing_df():
    """Load California Housing as a DataFrame (cached across test session)."""
    data = fetch_california_housing(as_frame=True)
    df = data.frame
    return df


@pytest.fixture(scope="session")
def housing_features_target(california_housing_df):
    """Return scaled features and target from California Housing."""
    from sklearn.preprocessing import StandardScaler
    df = california_housing_df
    target_col = 'MedHouseVal'
    X = df.drop(columns=[target_col]).values
    y = df[target_col].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, y


@pytest.fixture(scope="session")
def train_test_housing(housing_features_target):
    """Provide train/test split of California Housing."""
    from sklearn.model_selection import train_test_split
    X, y = housing_features_target
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    return X_train, X_test, y_train, y_test


@pytest.fixture
def simple_regression_data():
    """Small synthetic regression dataset for fast tests."""
    X, y = make_regression(n_samples=200, n_features=5, noise=10, random_state=42)
    return X, y


@pytest.fixture
def blob_data():
    """Synthetic blob data for clustering tests."""
    X, true_labels = make_blobs(n_samples=300, centers=3, n_features=5, random_state=42)
    return X, true_labels


@pytest.fixture
def small_blob_data():
    """Smaller blob data for faster DBSCAN/hierarchical tests."""
    X, true_labels = make_blobs(n_samples=150, centers=3, n_features=4, random_state=42)
    return X, true_labels
