"""
Tests for K-Means clustering (Phase 2).

Tests verify:
- Optimal K finder returns correct structure
- K-Means produces valid cluster assignments
- Silhouette scores and inertias are reasonable
"""

import numpy as np
import pytest

from src.clustering import find_optimal_k, perform_kmeans


class TestFindOptimalK:
    """Tests for find_optimal_k()."""

    def test_returns_correct_keys(self, blob_data):
        X, _ = blob_data
        results = find_optimal_k(X, k_range=range(2, 6))
        assert 'inertias' in results
        assert 'silhouette_scores' in results
        assert 'k_range' in results
        assert 'best_k_silhouette' in results

    def test_correct_number_of_results(self, blob_data):
        X, _ = blob_data
        k_range = range(2, 6)
        results = find_optimal_k(X, k_range=k_range)
        assert len(results['inertias']) == len(k_range)
        assert len(results['silhouette_scores']) == len(k_range)

    def test_inertia_decreases_with_k(self, blob_data):
        X, _ = blob_data
        results = find_optimal_k(X, k_range=range(2, 8))
        inertias = results['inertias']
        # Inertia should generally decrease as k increases
        assert inertias[0] > inertias[-1]

    def test_best_k_in_range(self, blob_data):
        X, _ = blob_data
        k_range = range(2, 8)
        results = find_optimal_k(X, k_range=k_range)
        assert results['best_k_silhouette'] in list(k_range)

    def test_finds_correct_k_for_blobs(self, blob_data):
        X, _ = blob_data  # 3 blobs
        results = find_optimal_k(X, k_range=range(2, 7))
        # Best k should be 3 for well-separated 3-blob data
        assert results['best_k_silhouette'] == 3

    def test_silhouette_scores_valid(self, blob_data):
        X, _ = blob_data
        results = find_optimal_k(X, k_range=range(2, 6))
        for s in results['silhouette_scores']:
            assert -1 <= s <= 1, "Silhouette score must be in [-1, 1]"


class TestPerformKMeans:
    """Tests for perform_kmeans()."""

    def test_returns_correct_keys(self, blob_data):
        X, _ = blob_data
        result = perform_kmeans(X, n_clusters=3)
        assert 'model' in result
        assert 'labels' in result
        assert 'centroids' in result
        assert 'inertia' in result
        assert 'silhouette' in result

    def test_correct_number_of_clusters(self, blob_data):
        X, _ = blob_data
        result = perform_kmeans(X, n_clusters=3)
        assert len(np.unique(result['labels'])) == 3

    def test_labels_length(self, blob_data):
        X, _ = blob_data
        result = perform_kmeans(X, n_clusters=3)
        assert len(result['labels']) == len(X)

    def test_centroids_shape(self, blob_data):
        X, _ = blob_data
        result = perform_kmeans(X, n_clusters=4)
        assert result['centroids'].shape == (4, X.shape[1])

    def test_silhouette_positive_for_good_clusters(self, blob_data):
        X, _ = blob_data
        result = perform_kmeans(X, n_clusters=3)
        assert result['silhouette'] > 0.5, "Well-separated blobs should have high silhouette"

    def test_inertia_is_positive(self, blob_data):
        X, _ = blob_data
        result = perform_kmeans(X, n_clusters=3)
        assert result['inertia'] > 0

    def test_reproducibility(self, blob_data):
        X, _ = blob_data
        r1 = perform_kmeans(X, n_clusters=3, random_state=42)
        r2 = perform_kmeans(X, n_clusters=3, random_state=42)
        assert np.array_equal(r1['labels'], r2['labels'])
