# Phase 2 – Clustering & Dimensionality Reduction: Function Notes

**Source file:** `src/clustering.py`
**Date:** April 7, 2026

---

## Overview

This module implements unsupervised market segmentation for real-estate data.
Three clustering families are provided (K-Means, Hierarchical/Agglomerative, DBSCAN)
plus PCA-based dimensionality reduction.  A thin wrapper (`cluster_with_pca`) chains
PCA → K-Means for a common pattern in high-dimensional datasets.

All functions accept a **pre-scaled** feature matrix `X` (NumPy array).
Callers are responsible for applying `StandardScaler` (or equivalent) before passing
data in.  This keeps each function single-purpose and testable in isolation.

---

## Section 1 – K-Means Clustering

### Design philosophy

K-Means is the natural starting point because it is fast, interpretable, and
produces hard cluster assignments.  Its main weakness is that the caller must
supply the number of clusters `k` up-front.  The pair of functions below
separates the *search* for `k` from the *execution* of clustering, which makes
notebook workflows cleaner.

---

### `find_optimal_k(X, k_range, random_state)`

**Purpose:** Identify the best `k` before committing to a final model.

**Approach – two complementary metrics:**

1. **Inertia (Elbow method)** – KMeans minimises the within-cluster sum of
   squared distances (inertia).  As `k` increases, inertia always decreases;
   the "elbow" is where the rate of decrease flattens.  This is useful for
   visualisation but requires human judgement to locate the elbow precisely.

2. **Silhouette score** – For each point, the silhouette measures how similar it
   is to its own cluster versus the nearest neighbouring cluster.  Values range
   from −1 to +1; higher is better and the score is objective, so the function
   returns `best_k_silhouette` automatically.

**Why return both?** Elbow plots often disagree with silhouette peaks.  Returning
both lets the analyst look at the plot *and* have an algorithmic recommendation.

**Implementation logic:**

```
for each k in k_range:
    fit KMeans → record inertia_
    compute silhouette_score(X, labels)

best_k = k_range[ argmax(silhouette_scores) ]
```

`n_init='auto'` is passed to KMeans to silence a scikit-learn 1.4+ deprecation
warning and use the new smarter initialisation default.

**Key constraint:** `k_range` must start at 2 because silhouette score is
undefined for a single cluster.

---

### `perform_kmeans(X, n_clusters, random_state)`

**Purpose:** Fit a final K-Means model with a chosen `k` and return everything
needed for downstream analysis or plotting.

**Approach:** Straightforward – fit once, expose the five most-used artefacts:
the model object (for `predict` on new data), integer labels, centroids (useful
for profiling each segment), inertia (comparable across runs), and silhouette
(quality indicator).

---

## Section 2 – Hierarchical (Agglomerative) Clustering

### Design philosophy

Hierarchical clustering does **not** require `k` to be specified in advance; the
full merge tree (dendrogram) is built first and the user can choose the cut
level after inspecting it.  This makes it complementary to K-Means: use the
dendrogram to build intuition about natural groupings, then use K-Means (or the
agglomerative result itself) for the final assignment.

The two functions here are separated on purpose:

- `perform_hierarchical_clustering` – scikit-learn path, gives cluster labels efficiently.
- `compute_linkage_matrix` – SciPy path, gives the full linkage matrix needed
  by `scipy.cluster.hierarchy.dendrogram` for plotting.

scikit-learn's `AgglomerativeClustering` does not expose a linkage matrix in a
SciPy-compatible format, so both libraries are needed.

---

### `perform_hierarchical_clustering(X, n_clusters, linkage_method)`

**Approach:**

```
fit AgglomerativeClustering(n_clusters, linkage=linkage_method)
compute silhouette_score for quality comparison
```

**Linkage method trade-offs:**

| Method     | Merges clusters by…                         | Behaviour                                   |
|------------|---------------------------------------------|---------------------------------------------|
| `ward`     | Minimising within-cluster variance increase | Compact, equal-sized clusters (default)     |
| `complete` | Maximum pairwise distance                   | Compact but sensitive to outliers           |
| `average`  | Mean pairwise distance                      | Compromise between ward and single          |
| `single`   | Minimum pairwise distance                   | Can produce long "chaining" clusters        |

`ward` is the default because it tends to produce the most balanced segments,
which is desirable for market segmentation.

---

### `compute_linkage_matrix(X, method)`

**Approach:** Delegates entirely to `scipy.cluster.hierarchy.linkage`.

The returned matrix has shape `(n-1, 4)` where each row describes one merge:

```
[cluster_i, cluster_j, distance, new_cluster_size]
```

This is the format expected by `scipy.cluster.hierarchy.dendrogram` for
visualisation.  Having this as a dedicated function means the notebook can call
it once, store the result, and pass it to multiple dendrogram plots without
refitting.

---

## Section 3 – DBSCAN Clustering

### Design philosophy

DBSCAN (Density-Based Spatial Clustering of Applications with Noise) identifies
clusters as dense regions separated by sparse space.  Unlike K-Means and
agglomerative clustering it:

- Does **not** require `k` to be specified.
- Can find **arbitrarily shaped** clusters.
- Explicitly labels low-density points as **noise** (label `-1`).

This makes it attractive for real-estate data where the market may contain a
handful of very distinct neighbourhoods plus some outlier properties that don't
belong to any segment.

The main challenge is hyperparameter sensitivity: the results can change
dramatically with small changes to `eps` and `min_samples`.

---

### `perform_dbscan(X, eps, min_samples)`

**Approach:**

```
fit DBSCAN(eps, min_samples)
n_clusters = count of unique labels excluding -1
n_noise    = count of points with label == -1
silhouette = silhouette_score on non-noise points if n_clusters >= 2
```

**Silhouette guard:** The silhouette metric requires at least 2 clusters.
Additionally, each cluster must have at least one non-noise point; the guard
`mask.sum() > n_clusters` ensures there are enough points to compute a
meaningful score (otherwise `None` is returned).

**Why exclude noise from silhouette?** Noise points (label `-1`) are not
assigned to any cluster.  Including them would corrupt the inter-cluster
distance calculation.

---

### `tune_dbscan(X, eps_range, min_samples_range)`

**Purpose:** Systematic hyperparameter search because DBSCAN has no built-in
equivalent of the elbow plot or silhouette peak used in K-Means.

**Approach:** Exhaustive grid search over all `(eps, min_samples)` pairs,
calling `perform_dbscan` for each, collecting results into a DataFrame.

```
for eps in eps_range:
    for min_samples in min_samples_range:
        result = perform_dbscan(X, eps, min_samples)
        append row to results
return pd.DataFrame(results)
```

**How to read the output:**

- Rows where `n_clusters == 0` mean `eps` is too small (everything is noise).
- Rows where `n_clusters == 1` mean `eps` is too large (everything merged).
- The best candidates have 2–6 clusters, low `n_noise`, and a positive
  `silhouette`.

**Default ranges** (`eps`: 0.3 → 1.5, `min_samples`: 3 → 10) are calibrated
for standardised (z-scored) data.  For raw or differently-scaled inputs the
ranges should be adjusted.

---

## Section 4 – PCA & Dimensionality Reduction

### Design philosophy

Real-estate datasets often have many correlated features (square footage, room
counts, lot size, age, etc.).  PCA projects the data onto orthogonal axes
(principal components) ordered by the amount of variance they explain.
This serves two purposes:

1. **Noise reduction** – dropping components that explain little variance
   removes noise without losing signal.
2. **Visualisation** – reducing to 2 components allows scatter plots of
   cluster assignments.

The three PCA functions are layered: `perform_pca` is the primitive,
`find_optimal_components` selects the right number of components, and
`cluster_with_pca` composes both with K-Means.

---

### `perform_pca(X, n_components)`

**Approach:**

```
fit PCA(n_components)
transform X → pca_data
return explained_variance_ratio_ and its cumsum
```

`n_components=None` keeps all components, which is useful for
`find_optimal_components` where you need the full variance profile before
deciding how many to keep.

**Cumulative variance** is the running total of explained variance; e.g.
`[0.42, 0.65, 0.80, 0.91, ...]` means the first component captures 42 %,
the first two together 65 %, and so on.

---

### `find_optimal_components(X, variance_threshold)`

**Purpose:** Answer the question "how many PCA components do I need to retain
at least X% of the information?"

**Approach:**

```
fit full PCA (all components)
cumulative_variance = cumsum(explained_variance_ratio_)
n = searchsorted(cumulative_variance, threshold) + 1
```

`np.searchsorted` finds the first index where the cumulative variance meets or
exceeds the threshold.  Adding 1 converts from a 0-based index to a component
count.

**Default threshold = 0.95** – retaining 95 % of variance is a common
convention that balances dimensionality reduction against information loss.

---

### `cluster_with_pca(X, n_clusters, n_components, random_state)`

**Purpose:** Combine dimensionality reduction and clustering in a single call.
This is the most common workflow for high-dimensional real-estate data.

**Approach:**

```
pca_data = perform_pca(X, n_components).transformed
result   = perform_kmeans(pca_data, n_clusters, random_state)
```

**Why cluster in PCA space rather than original space?**

- Correlated features give K-Means a distorted notion of distance; PCA
  decorrelates them.
- Fewer dimensions means faster convergence and less sensitivity to the curse
  of dimensionality.
- 2-component PCA data can be directly scatter-plotted with cluster colours,
  making results immediately interpretable.

**Trade-off:** Silhouette is computed on the PCA-transformed data, not the
original space.  This means the score reflects cluster quality *in the reduced
space*, which may differ slightly from quality in the original space.

---

## Algorithm Comparison Summary

| Property              | K-Means         | Hierarchical        | DBSCAN              |
|-----------------------|-----------------|---------------------|---------------------|
| Requires `k` upfront  | Yes             | Yes (at cut time)   | No                  |
| Cluster shape         | Convex/spherical| Any (via linkage)   | Arbitrary           |
| Handles noise/outliers| No              | No                  | Yes (label = -1)    |
| Scalability           | High            | Medium (O(n² log n))| Medium              |
| Hyperparameter tuning | `k`             | `k`, linkage method | `eps`, `min_samples`|
| Deterministic         | No (random init)| Yes                 | Yes                 |

---

## Recommended Workflow

```
1. Scale features with StandardScaler before calling any function.

2. Run find_optimal_k  →  choose k from elbow plot + silhouette peak.
3. Run perform_kmeans  →  baseline segmentation.

4. Run compute_linkage_matrix  →  inspect dendrogram for natural groupings.
5. Run perform_hierarchical_clustering  →  compare labels to K-Means.

6. Run tune_dbscan  →  scan the results table for valid parameter ranges.
7. Run perform_dbscan with the best (eps, min_samples)  →  noise-aware segments.

8. Run find_optimal_components  →  decide how many PCA components to keep.
9. Run cluster_with_pca  →  final model if features are highly correlated or
   if a 2-D scatter plot of segments is needed.
```
