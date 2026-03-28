"""
Tests for content-based recommendation filtering (Phase 3).

Tests verify:
- Similarity matrix is correctly computed
- Recommendations are valid and properly ranked
- KNN-based recommendations work correctly
"""

import numpy as np
import pytest

from src.recommendation import (
    compute_property_similarity,
    content_based_recommend,
    knn_recommend,
)


class TestComputePropertySimilarity:
    """Tests for compute_property_similarity()."""

    def test_correct_shape(self):
        X = np.random.rand(50, 5)
        sim = compute_property_similarity(X, metric='cosine')
        assert sim.shape == (50, 50)

    def test_self_similarity_is_one(self):
        X = np.random.rand(30, 5)
        sim = compute_property_similarity(X, metric='cosine')
        assert np.allclose(np.diag(sim), 1.0, atol=1e-5)

    def test_symmetric_matrix(self):
        X = np.random.rand(30, 5)
        sim = compute_property_similarity(X, metric='cosine')
        assert np.allclose(sim, sim.T, atol=1e-6)

    def test_values_between_zero_and_one(self):
        X = np.random.rand(30, 5)
        sim = compute_property_similarity(X, metric='euclidean')
        assert sim.min() >= -0.01  # allow tiny floating point
        assert sim.max() <= 1.01

    def test_cosine_metric(self):
        X = np.array([[1, 0], [1, 0], [0, 1]])
        sim = compute_property_similarity(X, metric='cosine')
        assert np.isclose(sim[0, 1], 1.0, atol=1e-5)  # identical vectors
        assert sim[0, 2] < sim[0, 1]  # orthogonal is less similar

    def test_euclidean_metric(self):
        X = np.array([[0, 0], [1, 0], [10, 10]])
        sim = compute_property_similarity(X, metric='euclidean')
        assert sim[0, 1] > sim[0, 2]  # closer -> more similar


class TestContentBasedRecommend:
    """Tests for content_based_recommend()."""

    def test_correct_number_of_recommendations(self):
        sim = np.random.rand(100, 100)
        np.fill_diagonal(sim, 1.0)
        recs = content_based_recommend(0, sim, n_recommendations=5)
        assert len(recs) == 5

    def test_query_not_in_results(self):
        sim = np.random.rand(50, 50)
        np.fill_diagonal(sim, 1.0)
        recs = content_based_recommend(10, sim, n_recommendations=5)
        indices = [r['property_index'] for r in recs]
        assert 10 not in indices

    def test_sorted_by_similarity(self):
        sim = np.random.rand(50, 50)
        np.fill_diagonal(sim, 1.0)
        recs = content_based_recommend(0, sim, n_recommendations=5)
        scores = [r['similarity_score'] for r in recs]
        assert scores == sorted(scores, reverse=True)

    def test_result_structure(self):
        sim = np.random.rand(20, 20)
        np.fill_diagonal(sim, 1.0)
        recs = content_based_recommend(0, sim, n_recommendations=3)
        for rec in recs:
            assert 'property_index' in rec
            assert 'similarity_score' in rec

    def test_most_similar_property(self):
        sim = np.array([
            [1.0, 0.95, 0.3, 0.1],
            [0.95, 1.0, 0.4, 0.2],
            [0.3, 0.4, 1.0, 0.8],
            [0.1, 0.2, 0.8, 1.0],
        ])
        recs = content_based_recommend(0, sim, n_recommendations=2)
        assert recs[0]['property_index'] == 1


class TestKNNRecommend:
    """Tests for knn_recommend()."""

    def test_correct_number_of_recommendations(self):
        X = np.random.rand(100, 5)
        recs = knn_recommend(X, property_index=0, n_recommendations=5)
        assert len(recs) == 5

    def test_query_not_in_results(self):
        X = np.random.rand(50, 5)
        recs = knn_recommend(X, property_index=10, n_recommendations=3)
        indices = [r['property_index'] for r in recs]
        assert 10 not in indices

    def test_result_structure(self):
        X = np.random.rand(50, 5)
        recs = knn_recommend(X, property_index=0, n_recommendations=3)
        for rec in recs:
            assert 'property_index' in rec
            assert 'distance' in rec

    def test_distances_non_negative(self):
        X = np.random.rand(50, 5)
        recs = knn_recommend(X, property_index=0, n_recommendations=5)
        for rec in recs:
            assert rec['distance'] >= 0
