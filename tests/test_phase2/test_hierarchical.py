"""
Tests for Hierarchical clustering (Phase 2).

Tests verify:
- Agglomerative clustering produces valid clusters
- Linkage matrix has correct structure for dendrograms
"""

import numpy as np
import pytest

from src.clustering import perform_hierarchical_clustering, compute_linkage_matrix


class TestHierarchicalClustering:
    """Tests for perform_hierarchical_clustering()."""

    def test_returns_correct_keys(self, small_blob_data):
        X, _ = small_blob_data
        result = perform_hierarchical_clustering(X, n_clusters=3)
        assert 'model' in result
        assert 'labels' in result
        assert 'silhouette' in result
        assert 'n_clusters' in result

    def test_correct_number_of_clusters(self, small_blob_data):
        X, _ = small_blob_data
        result = perform_hierarchical_clustering(X, n_clusters=3)
        assert len(np.unique(result['labels'])) == 3

    def test_labels_length(self, small_blob_data):
        X, _ = small_blob_data
        result = perform_hierarchical_clustering(X, n_clusters=3)
        assert len(result['labels']) == len(X)

    def test_different_linkage_methods(self, small_blob_data):
        X, _ = small_blob_data
        for method in ['ward', 'complete', 'average']:
            result = perform_hierarchical_clustering(X, n_clusters=3, linkage_method=method)
            assert len(np.unique(result['labels'])) == 3

    def test_silhouette_valid(self, small_blob_data):
        X, _ = small_blob_data
        result = perform_hierarchical_clustering(X, n_clusters=3)
        assert -1 <= result['silhouette'] <= 1

    def test_good_silhouette_for_blobs(self, small_blob_data):
        X, _ = small_blob_data
        result = perform_hierarchical_clustering(X, n_clusters=3)
        assert result['silhouette'] > 0.4


class TestLinkageMatrix:
    """Tests for compute_linkage_matrix()."""

    def test_returns_ndarray(self, small_blob_data):
        X, _ = small_blob_data
        Z = compute_linkage_matrix(X)
        assert isinstance(Z, np.ndarray)

    def test_correct_shape(self, small_blob_data):
        X, _ = small_blob_data
        Z = compute_linkage_matrix(X)
        # Linkage matrix shape: (n_samples - 1, 4)
        assert Z.shape == (len(X) - 1, 4)

    def test_different_methods(self, small_blob_data):
        X, _ = small_blob_data
        for method in ['ward', 'complete', 'average', 'single']:
            Z = compute_linkage_matrix(X, method=method)
            assert Z.shape[1] == 4

    def test_distances_non_negative(self, small_blob_data):
        X, _ = small_blob_data
        Z = compute_linkage_matrix(X)
        # Third column is the distance; should be non-negative
        assert (Z[:, 2] >= 0).all()
