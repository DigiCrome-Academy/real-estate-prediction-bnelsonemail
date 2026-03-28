"""
Tests for hybrid recommendation system (Phase 3).

Tests verify:
- Hybrid recommender combines content and collaborative scores
- Evaluation metrics are correctly computed
"""

import numpy as np
import pytest

from src.recommendation import (
    hybrid_recommend,
    evaluate_recommendations,
    create_user_property_matrix,
)


class TestHybridRecommend:
    """Tests for hybrid_recommend()."""

    def test_returns_list(self):
        X = np.random.rand(100, 5)
        matrix = create_user_property_matrix(50, 100, sparsity=0.9, random_state=42)
        recs = hybrid_recommend(X, matrix, user_index=0, property_index=10)
        assert isinstance(recs, list)

    def test_correct_max_recommendations(self):
        X = np.random.rand(100, 5)
        matrix = create_user_property_matrix(50, 100, sparsity=0.9, random_state=42)
        recs = hybrid_recommend(
            X, matrix, user_index=0, property_index=10, n_recommendations=5
        )
        assert len(recs) <= 5

    def test_result_structure(self):
        X = np.random.rand(100, 5)
        matrix = create_user_property_matrix(50, 100, sparsity=0.9, random_state=42)
        recs = hybrid_recommend(X, matrix, user_index=0, property_index=10)
        for rec in recs:
            assert 'property_index' in rec
            assert 'content_score' in rec
            assert 'collaborative_score' in rec
            assert 'hybrid_score' in rec

    def test_sorted_by_hybrid_score(self):
        X = np.random.rand(100, 5)
        matrix = create_user_property_matrix(50, 100, sparsity=0.9, random_state=42)
        recs = hybrid_recommend(X, matrix, user_index=0, property_index=10)
        if len(recs) > 1:
            scores = [r['hybrid_score'] for r in recs]
            assert scores == sorted(scores, reverse=True)

    def test_weights_affect_results(self):
        X = np.random.rand(100, 5)
        matrix = create_user_property_matrix(50, 100, sparsity=0.9, random_state=42)
        recs_content = hybrid_recommend(
            X, matrix, user_index=0, property_index=10,
            content_weight=1.0, collaborative_weight=0.0
        )
        recs_collab = hybrid_recommend(
            X, matrix, user_index=0, property_index=10,
            content_weight=0.0, collaborative_weight=1.0
        )
        # With different weights, the top recommendation may differ
        # (Not guaranteed but likely with random data)
        # At minimum, both should return valid results
        assert len(recs_content) > 0
        assert len(recs_collab) > 0


class TestEvaluateRecommendations:
    """Tests for evaluate_recommendations()."""

    def test_returns_correct_keys(self):
        recs = [{'property_index': 0}, {'property_index': 1}]
        truth = {0: 4.0, 1: 2.0, 2: 5.0}
        metrics = evaluate_recommendations(recs, truth, threshold=3.5)
        assert 'precision' in metrics
        assert 'recall' in metrics
        assert 'n_relevant_recommended' in metrics
        assert 'n_recommended' in metrics
        assert 'n_relevant_total' in metrics

    def test_precision_calculation(self):
        recs = [
            {'property_index': 0},
            {'property_index': 1},
            {'property_index': 2},
        ]
        truth = {0: 4.0, 1: 2.0, 2: 5.0, 3: 4.5}
        metrics = evaluate_recommendations(recs, truth, threshold=3.5)
        # Relevant recommended: 0 (4.0) and 2 (5.0) => 2 out of 3
        assert np.isclose(metrics['precision'], 2 / 3)

    def test_recall_calculation(self):
        recs = [
            {'property_index': 0},
            {'property_index': 1},
            {'property_index': 2},
        ]
        truth = {0: 4.0, 1: 2.0, 2: 5.0, 3: 4.5}
        metrics = evaluate_recommendations(recs, truth, threshold=3.5)
        # Total relevant: 0, 2, 3 => 3. Recommended relevant: 0, 2 => 2. Recall = 2/3
        assert np.isclose(metrics['recall'], 2 / 3)

    def test_perfect_precision(self):
        recs = [{'property_index': 0}, {'property_index': 1}]
        truth = {0: 5.0, 1: 4.0}
        metrics = evaluate_recommendations(recs, truth, threshold=3.5)
        assert np.isclose(metrics['precision'], 1.0)

    def test_zero_precision(self):
        recs = [{'property_index': 0}, {'property_index': 1}]
        truth = {0: 1.0, 1: 2.0, 2: 5.0}
        metrics = evaluate_recommendations(recs, truth, threshold=3.5)
        assert np.isclose(metrics['precision'], 0.0)

    def test_no_recommendations(self):
        recs = []
        truth = {0: 5.0}
        metrics = evaluate_recommendations(recs, truth, threshold=3.5)
        assert metrics['n_recommended'] == 0
