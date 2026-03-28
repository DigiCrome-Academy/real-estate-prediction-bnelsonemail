"""
Tests for DBSCAN clustering (Phase 2).

Tests verify:
- DBSCAN produces valid cluster assignments
- Noise points are correctly identified
- Hyperparameter tuning returns valid results
"""

import numpy as np
import pandas as pd
import pytest

from src.clustering import perform_dbscan, tune_dbscan


class TestDBSCAN:
    """Tests for perform_dbscan()."""

    def test_returns_correct_keys(self, blob_data):
        X, _ = blob_data
        result = perform_dbscan(X, eps=1.0, min_samples=5)
        assert 'model' in result
        assert 'labels' in result
        assert 'n_clusters' in result
        assert 'n_noise' in result
        assert 'silhouette' in result

    def test_labels_length(self, blob_data):
        X, _ = blob_data
        result = perform_dbscan(X, eps=1.0, min_samples=5)
        assert len(result['labels']) == len(X)

    def test_noise_count_non_negative(self, blob_data):
        X, _ = blob_data
        result = perform_dbscan(X, eps=1.0, min_samples=5)
        assert result['n_noise'] >= 0

    def test_n_clusters_non_negative(self, blob_data):
        X, _ = blob_data
        result = perform_dbscan(X, eps=1.0, min_samples=5)
        assert result['n_clusters'] >= 0

    def test_noise_labeled_minus_one(self, blob_data):
        X, _ = blob_data
        result = perform_dbscan(X, eps=0.1, min_samples=50)  # aggressive params -> lots of noise
        labels = result['labels']
        noise_count = np.sum(labels == -1)
        assert noise_count == result['n_noise']

    def test_silhouette_none_when_less_than_2_clusters(self):
        # If eps is huge, everything is one cluster
        X = np.random.rand(50, 3)
        result = perform_dbscan(X, eps=100.0, min_samples=2)
        if result['n_clusters'] < 2:
            assert result['silhouette'] is None

    def test_finds_clusters_in_blobs(self, blob_data):
        X, _ = blob_data
        result = perform_dbscan(X, eps=2.0, min_samples=5)
        assert result['n_clusters'] >= 2, "Should find clusters in well-separated blobs"


class TestTuneDBSCAN:
    """Tests for tune_dbscan()."""

    def test_returns_dataframe(self, small_blob_data):
        X, _ = small_blob_data
        results = tune_dbscan(X, eps_range=[0.5, 1.0], min_samples_range=[3, 5])
        assert isinstance(results, pd.DataFrame)

    def test_correct_number_of_rows(self, small_blob_data):
        X, _ = small_blob_data
        eps_vals = [0.5, 1.0, 1.5]
        ms_vals = [3, 5]
        results = tune_dbscan(X, eps_range=eps_vals, min_samples_range=ms_vals)
        assert len(results) == len(eps_vals) * len(ms_vals)

    def test_has_required_columns(self, small_blob_data):
        X, _ = small_blob_data
        results = tune_dbscan(X, eps_range=[0.5, 1.0], min_samples_range=[3, 5])
        for col in ['eps', 'min_samples', 'n_clusters', 'n_noise', 'silhouette']:
            assert col in results.columns, f"Missing column: {col}"

    def test_uses_default_ranges(self, small_blob_data):
        X, _ = small_blob_data
        results = tune_dbscan(X)
        assert len(results) > 0, "Should produce results with default parameters"
