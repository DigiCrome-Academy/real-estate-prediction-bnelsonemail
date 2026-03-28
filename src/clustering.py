"""
Phase 2: Clustering & Dimensionality Reduction Module

This module implements market segmentation using clustering algorithms
and dimensionality reduction with PCA.

Algorithms to implement:
- K-Means Clustering
- Hierarchical (Agglomerative) Clustering
- DBSCAN
- PCA for dimensionality reduction
"""

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, calinski_harabasz_score
from scipy.cluster.hierarchy import linkage, dendrogram


# =============================================================================
# Section 1: K-Means Clustering
# =============================================================================

def find_optimal_k(X, k_range=range(2, 11), random_state=42):
    """
    Find the optimal number of clusters using the Elbow method and Silhouette scores.

    Args:
        X (np.ndarray): Scaled feature matrix.
        k_range (range): Range of k values to test.
        random_state (int): Random seed.

    Returns:
        dict: {
            'inertias': list of inertia values for each k,
            'silhouette_scores': list of silhouette scores for each k,
            'k_range': list of k values tested,
            'best_k_silhouette': int (k with highest silhouette score)
        }

    Example:
        >>> from sklearn.datasets import make_blobs
        >>> X, _ = make_blobs(n_samples=300, centers=3, random_state=42)
        >>> results = find_optimal_k(X, k_range=range(2, 6))
        >>> len(results['inertias']) == 4
        True
        >>> results['best_k_silhouette'] >= 2
        True
    """
    # TODO: Implement this function
    # Hints:
    #   1. For each k in k_range, fit KMeans and record inertia_
    #   2. Compute silhouette_score for each clustering
    #   3. Find the k with the highest silhouette score
    raise NotImplementedError("Implement find_optimal_k()")


def perform_kmeans(X, n_clusters, random_state=42):
    """
    Perform K-Means clustering.

    Args:
        X (np.ndarray): Scaled feature matrix.
        n_clusters (int): Number of clusters.
        random_state (int): Random seed.

    Returns:
        dict: {
            'model': fitted KMeans object,
            'labels': cluster labels (np.ndarray),
            'centroids': cluster centers (np.ndarray),
            'inertia': float,
            'silhouette': float (silhouette score)
        }

    Example:
        >>> from sklearn.datasets import make_blobs
        >>> X, _ = make_blobs(n_samples=300, centers=3, random_state=42)
        >>> result = perform_kmeans(X, n_clusters=3)
        >>> len(np.unique(result['labels'])) == 3
        True
        >>> result['silhouette'] > 0
        True
    """
    # TODO: Implement this function
    raise NotImplementedError("Implement perform_kmeans()")


# =============================================================================
# Section 2: Hierarchical Clustering
# =============================================================================

def perform_hierarchical_clustering(X, n_clusters, linkage_method='ward'):
    """
    Perform Agglomerative (Hierarchical) Clustering.

    Args:
        X (np.ndarray): Scaled feature matrix.
        n_clusters (int): Number of clusters.
        linkage_method (str): Linkage criterion ('ward', 'complete', 'average', 'single').

    Returns:
        dict: {
            'model': fitted AgglomerativeClustering object,
            'labels': cluster labels (np.ndarray),
            'silhouette': float (silhouette score),
            'n_clusters': int
        }

    Example:
        >>> from sklearn.datasets import make_blobs
        >>> X, _ = make_blobs(n_samples=200, centers=3, random_state=42)
        >>> result = perform_hierarchical_clustering(X, n_clusters=3)
        >>> len(np.unique(result['labels'])) == 3
        True
    """
    # TODO: Implement this function
    raise NotImplementedError("Implement perform_hierarchical_clustering()")


def compute_linkage_matrix(X, method='ward'):
    """
    Compute the linkage matrix for dendrogram visualization.

    Args:
        X (np.ndarray): Scaled feature matrix.
        method (str): Linkage method.

    Returns:
        np.ndarray: Linkage matrix from scipy.

    Example:
        >>> X = np.random.rand(50, 3)
        >>> Z = compute_linkage_matrix(X)
        >>> Z.shape[1] == 4  # linkage matrix always has 4 columns
        True
    """
    # TODO: Implement this function
    # Hint: Use scipy.cluster.hierarchy.linkage
    raise NotImplementedError("Implement compute_linkage_matrix()")


# =============================================================================
# Section 3: DBSCAN Clustering
# =============================================================================

def perform_dbscan(X, eps=0.5, min_samples=5):
    """
    Perform DBSCAN clustering.

    Args:
        X (np.ndarray): Scaled feature matrix.
        eps (float): Maximum distance between two samples in same neighborhood.
        min_samples (int): Minimum samples in a neighborhood for a core point.

    Returns:
        dict: {
            'model': fitted DBSCAN object,
            'labels': cluster labels (np.ndarray, -1 = noise),
            'n_clusters': int (number of clusters, excluding noise),
            'n_noise': int (number of noise points),
            'silhouette': float or None (None if <2 clusters found)
        }

    Example:
        >>> from sklearn.datasets import make_blobs
        >>> X, _ = make_blobs(n_samples=300, centers=3, random_state=42)
        >>> result = perform_dbscan(X, eps=1.0, min_samples=5)
        >>> result['n_clusters'] >= 1
        True
        >>> result['n_noise'] >= 0
        True
    """
    # TODO: Implement this function
    # Hints:
    #   1. Fit DBSCAN on X
    #   2. Count unique labels (excluding -1 for noise)
    #   3. Only compute silhouette if there are >= 2 clusters
    raise NotImplementedError("Implement perform_dbscan()")


def tune_dbscan(X, eps_range=None, min_samples_range=None):
    """
    Tune DBSCAN hyperparameters by testing combinations of eps and min_samples.

    Args:
        X (np.ndarray): Scaled feature matrix.
        eps_range (list): List of eps values to test. Default: [0.3, 0.5, 0.7, 1.0, 1.5]
        min_samples_range (list): List of min_samples values. Default: [3, 5, 7, 10]

    Returns:
        pd.DataFrame: Results with columns ['eps', 'min_samples', 'n_clusters',
                       'n_noise', 'silhouette'].

    Example:
        >>> from sklearn.datasets import make_blobs
        >>> X, _ = make_blobs(n_samples=200, centers=3, random_state=42)
        >>> results = tune_dbscan(X, eps_range=[0.5, 1.0], min_samples_range=[3, 5])
        >>> isinstance(results, pd.DataFrame)
        True
        >>> 'silhouette' in results.columns
        True
    """
    # TODO: Implement this function
    raise NotImplementedError("Implement tune_dbscan()")


# =============================================================================
# Section 4: PCA & Dimensionality Reduction
# =============================================================================

def perform_pca(X, n_components=None):
    """
    Perform PCA on the feature matrix.

    Args:
        X (np.ndarray): Scaled feature matrix.
        n_components (int or None): Number of components.
            If None, keep all components.

    Returns:
        dict: {
            'model': fitted PCA object,
            'transformed': np.ndarray (transformed data),
            'explained_variance_ratio': np.ndarray,
            'cumulative_variance': np.ndarray,
            'n_components': int (number of components used)
        }

    Example:
        >>> X = np.random.rand(100, 8)
        >>> result = perform_pca(X, n_components=3)
        >>> result['transformed'].shape[1] == 3
        True
        >>> len(result['explained_variance_ratio']) == 3
        True
        >>> result['cumulative_variance'][-1] <= 1.0
        True
    """
    # TODO: Implement this function
    # Hints:
    #   1. Fit PCA with n_components
    #   2. Transform X
    #   3. Return explained_variance_ratio_ and its cumulative sum
    raise NotImplementedError("Implement perform_pca()")


def find_optimal_components(X, variance_threshold=0.95):
    """
    Find the minimum number of PCA components that explain at least
    the specified variance threshold.

    Args:
        X (np.ndarray): Scaled feature matrix.
        variance_threshold (float): Minimum cumulative explained variance (0.0 to 1.0).

    Returns:
        int: Minimum number of components needed.

    Example:
        >>> X = np.random.rand(200, 10)
        >>> n = find_optimal_components(X, variance_threshold=0.90)
        >>> 1 <= n <= 10
        True
    """
    # TODO: Implement this function
    # Hints:
    #   1. Fit PCA with all components
    #   2. Compute cumulative variance
    #   3. Find the first index where cumulative variance >= threshold
    raise NotImplementedError("Implement find_optimal_components()")


def cluster_with_pca(X, n_clusters, n_components=2, random_state=42):
    """
    Apply PCA for dimensionality reduction, then cluster using K-Means.

    Args:
        X (np.ndarray): Scaled feature matrix.
        n_clusters (int): Number of clusters for K-Means.
        n_components (int): Number of PCA components to use.
        random_state (int): Random seed.

    Returns:
        dict: {
            'pca_model': fitted PCA,
            'kmeans_model': fitted KMeans,
            'pca_data': np.ndarray (PCA-transformed data),
            'labels': np.ndarray (cluster labels),
            'silhouette': float
        }

    Example:
        >>> from sklearn.datasets import make_blobs
        >>> X, _ = make_blobs(n_samples=300, centers=3, n_features=10, random_state=42)
        >>> result = cluster_with_pca(X, n_clusters=3, n_components=2)
        >>> result['pca_data'].shape[1] == 2
        True
        >>> len(np.unique(result['labels'])) == 3
        True
    """
    # TODO: Implement this function
    raise NotImplementedError("Implement cluster_with_pca()")
