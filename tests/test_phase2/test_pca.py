"""
Tests for PCA and dimensionality reduction (Phase 2).

Tests verify:
- PCA reduces dimensions correctly
- Explained variance ratios are valid
- Optimal component finder works
- Combined PCA + clustering works
"""

import numpy as np
import pytest

from src.clustering import perform_pca, find_optimal_components, cluster_with_pca


class TestPerformPCA:
    """Tests for perform_pca()."""

    def test_returns_correct_keys(self, blob_data):
        X, _ = blob_data
        result = perform_pca(X, n_components=2)
        assert 'model' in result
        assert 'transformed' in result
        assert 'explained_variance_ratio' in result
        assert 'cumulative_variance' in result
        assert 'n_components' in result

    def test_correct_output_dimensions(self, blob_data):
        X, _ = blob_data
        result = perform_pca(X, n_components=2)
        assert result['transformed'].shape == (len(X), 2)

    def test_explained_variance_sums_to_one_with_all_components(self):
        X = np.random.rand(100, 5)
        result = perform_pca(X, n_components=None)
        assert np.isclose(result['explained_variance_ratio'].sum(), 1.0, atol=1e-6)

    def test_cumulative_variance_increasing(self, blob_data):
        X, _ = blob_data
        result = perform_pca(X, n_components=3)
        cv = result['cumulative_variance']
        assert all(cv[i] <= cv[i + 1] for i in range(len(cv) - 1))

    def test_cumulative_variance_bounded(self, blob_data):
        X, _ = blob_data
        result = perform_pca(X)
        assert result['cumulative_variance'][-1] <= 1.0 + 1e-6

    def test_variance_ratio_length(self, blob_data):
        X, _ = blob_data
        result = perform_pca(X, n_components=3)
        assert len(result['explained_variance_ratio']) == 3

    def test_n_components_stored(self, blob_data):
        X, _ = blob_data
        result = perform_pca(X, n_components=2)
        assert result['n_components'] == 2


class TestFindOptimalComponents:
    """Tests for find_optimal_components()."""

    def test_returns_integer(self):
        X = np.random.rand(200, 10)
        n = find_optimal_components(X, variance_threshold=0.90)
        assert isinstance(n, (int, np.integer))

    def test_result_in_valid_range(self):
        X = np.random.rand(200, 10)
        n = find_optimal_components(X, variance_threshold=0.90)
        assert 1 <= n <= 10

    def test_higher_threshold_needs_more_components(self):
        X = np.random.rand(200, 10)
        n_low = find_optimal_components(X, variance_threshold=0.80)
        n_high = find_optimal_components(X, variance_threshold=0.99)
        assert n_high >= n_low

    def test_threshold_one_returns_all(self):
        X = np.random.rand(100, 5)
        n = find_optimal_components(X, variance_threshold=1.0)
        assert n == 5


class TestClusterWithPCA:
    """Tests for cluster_with_pca()."""

    def test_returns_correct_keys(self, blob_data):
        X, _ = blob_data
        result = cluster_with_pca(X, n_clusters=3, n_components=2)
        assert 'pca_model' in result
        assert 'kmeans_model' in result
        assert 'pca_data' in result
        assert 'labels' in result
        assert 'silhouette' in result

    def test_pca_data_shape(self, blob_data):
        X, _ = blob_data
        result = cluster_with_pca(X, n_clusters=3, n_components=2)
        assert result['pca_data'].shape == (len(X), 2)

    def test_correct_clusters(self, blob_data):
        X, _ = blob_data
        result = cluster_with_pca(X, n_clusters=3, n_components=2)
        assert len(np.unique(result['labels'])) == 3

    def test_labels_length(self, blob_data):
        X, _ = blob_data
        result = cluster_with_pca(X, n_clusters=3, n_components=2)
        assert len(result['labels']) == len(X)

    def test_silhouette_positive(self, blob_data):
        X, _ = blob_data
        result = cluster_with_pca(X, n_clusters=3, n_components=2)
        assert result['silhouette'] > 0
