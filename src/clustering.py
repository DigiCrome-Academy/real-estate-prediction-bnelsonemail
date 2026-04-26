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
    inertias = []
    silhouette_scores = []
    k_list = list(k_range)

    for k in k_list:
        km = KMeans(n_clusters=k, random_state=random_state, n_init='auto')
        labels = km.fit_predict(X)
        inertias.append(km.inertia_)
        silhouette_scores.append(silhouette_score(X, labels))

    best_k_silhouette = k_list[int(np.argmax(silhouette_scores))]

    return {
        'inertias': inertias,
        'silhouette_scores': silhouette_scores,
        'k_range': k_list,
        'best_k_silhouette': best_k_silhouette,
    }


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
    model = KMeans(n_clusters=n_clusters, random_state=random_state, n_init='auto')
    labels = model.fit_predict(X)

    return {
        'model': model,
        'labels': labels,
        'centroids': model.cluster_centers_,
        'inertia': model.inertia_,
        'silhouette': silhouette_score(X, labels),
    }


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
    model = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage_method)
    labels = model.fit_predict(X)

    return {
        'model': model,
        'labels': labels,
        'silhouette': silhouette_score(X, labels),
        'n_clusters': n_clusters,
    }


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
    return linkage(X, method=method)


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
    model = DBSCAN(eps=eps, min_samples=min_samples)
    labels = model.fit_predict(X)

    unique_labels = set(labels)
    n_clusters = len(unique_labels - {-1})
    n_noise = int(np.sum(labels == -1))

    if n_clusters >= 2:
        # Silhouette requires at least 2 clusters and excludes noise points
        mask = labels != -1
        silhouette = silhouette_score(X[mask], labels[mask]) if mask.sum() > n_clusters else None
    else:
        silhouette = None

    return {
        'model': model,
        'labels': labels,
        'n_clusters': n_clusters,
        'n_noise': n_noise,
        'silhouette': silhouette,
    }


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
    if eps_range is None:
        eps_range = [0.3, 0.5, 0.7, 1.0, 1.5]
    if min_samples_range is None:
        min_samples_range = [3, 5, 7, 10]

    rows = []
    for eps in eps_range:
        for min_samples in min_samples_range:
            result = perform_dbscan(X, eps=eps, min_samples=min_samples)
            rows.append({
                'eps': eps,
                'min_samples': min_samples,
                'n_clusters': result['n_clusters'],
                'n_noise': result['n_noise'],
                'silhouette': result['silhouette'],
            })

    return pd.DataFrame(rows, columns=['eps', 'min_samples', 'n_clusters', 'n_noise', 'silhouette'])


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
    model = PCA(n_components=n_components)
    transformed = model.fit_transform(X)
    evr = model.explained_variance_ratio_

    return {
        'model': model,
        'transformed': transformed,
        'explained_variance_ratio': evr,
        'cumulative_variance': np.cumsum(evr),
        'n_components': model.n_components_,
    }


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
    pca = PCA(n_components=None)
    pca.fit(X)
    cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
    # np.searchsorted finds the first index where cumulative_variance >= threshold;
    # add 1 to convert from 0-based index to component count.
    n_components = int(np.searchsorted(cumulative_variance, variance_threshold, side='left')) + 1
    return min(n_components, X.shape[1])


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
    pca_result = perform_pca(X, n_components=n_components)
    pca_data = pca_result['transformed']

    kmeans_result = perform_kmeans(pca_data, n_clusters=n_clusters, random_state=random_state)

    return {
        'pca_model': pca_result['model'],
        'kmeans_model': kmeans_result['model'],
        'pca_data': pca_data,
        'labels': kmeans_result['labels'],
        'silhouette': kmeans_result['silhouette'],
    }
