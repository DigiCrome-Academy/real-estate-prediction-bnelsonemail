"""
Tests for collaborative filtering recommendations (Phase 3).

Tests verify:
- User-property matrix is correctly generated
- User-based and item-based collaborative filtering produce valid recommendations
"""

import numpy as np
import pytest

from src.recommendation import (
    create_user_property_matrix,
    user_based_collaborative_filter,
    item_based_collaborative_filter,
)


class TestCreateUserPropertyMatrix:
    """Tests for create_user_property_matrix()."""

    def test_correct_shape(self):
        matrix = create_user_property_matrix(n_users=50, n_properties=100)
        assert matrix.shape == (50, 100)

    def test_values_in_range(self):
        matrix = create_user_property_matrix(n_users=50, n_properties=100)
        assert matrix.min() >= 0
        assert matrix.max() <= 5

    def test_sparsity(self):
        matrix = create_user_property_matrix(
            n_users=100, n_properties=200, sparsity=0.9
        )
        zero_ratio = (matrix == 0).sum() / matrix.size
        # Allow some tolerance around the target sparsity
        assert zero_ratio >= 0.8, "Matrix should be approximately sparse"

    def test_has_nonzero_entries(self):
        matrix = create_user_property_matrix(n_users=50, n_properties=100, sparsity=0.9)
        assert (matrix > 0).sum() > 0, "Matrix should have some ratings"

    def test_reproducibility(self):
        m1 = create_user_property_matrix(n_users=20, n_properties=30, random_state=42)
        m2 = create_user_property_matrix(n_users=20, n_properties=30, random_state=42)
        assert np.array_equal(m1, m2)

    def test_ratings_are_integers(self):
        matrix = create_user_property_matrix(n_users=30, n_properties=50)
        nonzero = matrix[matrix > 0]
        assert np.all(nonzero == nonzero.astype(int)), "Ratings should be integers 1-5"


class TestUserBasedCollaborativeFilter:
    """Tests for user_based_collaborative_filter()."""

    def test_returns_list(self):
        matrix = create_user_property_matrix(50, 100, sparsity=0.85, random_state=42)
        recs = user_based_collaborative_filter(matrix, user_index=0, n_recommendations=5)
        assert isinstance(recs, list)

    def test_correct_max_recommendations(self):
        matrix = create_user_property_matrix(50, 100, sparsity=0.85, random_state=42)
        recs = user_based_collaborative_filter(matrix, user_index=0, n_recommendations=5)
        assert len(recs) <= 5

    def test_result_structure(self):
        matrix = create_user_property_matrix(50, 100, sparsity=0.85, random_state=42)
        recs = user_based_collaborative_filter(matrix, user_index=0, n_recommendations=3)
        for rec in recs:
            assert 'property_index' in rec
            assert 'predicted_rating' in rec

    def test_recommends_unrated_properties(self):
        matrix = create_user_property_matrix(50, 100, sparsity=0.85, random_state=42)
        user_rated = set(np.where(matrix[0] > 0)[0])
        recs = user_based_collaborative_filter(matrix, user_index=0, n_recommendations=5)
        for rec in recs:
            assert rec['property_index'] not in user_rated, \
                "Should not recommend already-rated properties"

    def test_sorted_by_predicted_rating(self):
        matrix = create_user_property_matrix(50, 100, sparsity=0.85, random_state=42)
        recs = user_based_collaborative_filter(matrix, user_index=0, n_recommendations=5)
        if len(recs) > 1:
            ratings = [r['predicted_rating'] for r in recs]
            assert ratings == sorted(ratings, reverse=True)


class TestItemBasedCollaborativeFilter:
    """Tests for item_based_collaborative_filter()."""

    def test_returns_list(self):
        matrix = create_user_property_matrix(50, 100, sparsity=0.85, random_state=42)
        recs = item_based_collaborative_filter(matrix, user_index=0, n_recommendations=5)
        assert isinstance(recs, list)

    def test_correct_max_recommendations(self):
        matrix = create_user_property_matrix(50, 100, sparsity=0.85, random_state=42)
        recs = item_based_collaborative_filter(matrix, user_index=0, n_recommendations=5)
        assert len(recs) <= 5

    def test_result_structure(self):
        matrix = create_user_property_matrix(50, 100, sparsity=0.85, random_state=42)
        recs = item_based_collaborative_filter(matrix, user_index=0, n_recommendations=3)
        for rec in recs:
            assert 'property_index' in rec
            assert 'predicted_rating' in rec

    def test_recommends_unrated_properties(self):
        matrix = create_user_property_matrix(50, 100, sparsity=0.85, random_state=42)
        user_rated = set(np.where(matrix[0] > 0)[0])
        recs = item_based_collaborative_filter(matrix, user_index=0, n_recommendations=5)
        for rec in recs:
            assert rec['property_index'] not in user_rated
