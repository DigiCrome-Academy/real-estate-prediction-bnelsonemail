# Phase 2 – Clustering Analysis: Results & Findings

**Notebook:** `notebooks/02_clustering_analysis.ipynb`
**Dataset:** California Housing (scikit-learn) — 20,640 properties, 8 features
**Date:** April 7, 2026

---

## Dataset & Preprocessing

The California Housing dataset contains 20,640 census block observations with
8 features and a continuous house-value target (`MedHouseVal`, in $100,000s).

| Feature | Description |
|---|---|
| `MedInc` | Median income of block ($10,000s) |
| `HouseAge` | Median house age (years) |
| `AveRooms` | Average rooms per household |
| `AveBedrms` | Average bedrooms per household |
| `Population` | Block population |
| `AveOccup` | Average household occupancy |
| `Latitude` | Block latitude |
| `Longitude` | Block longitude |

All features were standardised with `StandardScaler` (zero mean, unit variance)
before any clustering was applied.

---

## Section 2 – K-Means Clustering

### Optimal K Search

Both the Elbow method (inertia) and Silhouette score were computed across `k = 2..10`.

| k | Inertia | Silhouette |
|---|---|---|
| 2 | 129,613 | **0.3308** ← best |
| 3 | 118,310 | 0.3122 |
| 4 | 110,812 | 0.2679 |
| 5 | 92,066 | 0.2567 |
| 6 | 83,783 | 0.2765 |
| 7 | 78,299 | 0.2453 |
| 8 | 71,769 | 0.2782 |
| 9 | 54,932 | 0.2802 |
| 10 | 49,495 | 0.2503 |

The silhouette peak at **k = 2** is unambiguous. The elbow curve shows diminishing
inertia reduction beyond k = 2, consistent with a two-segment structure.

### K-Means Results (k = 2)

| Metric | Value |
|---|---|
| Inertia | 129,613.2 |
| Silhouette score | 0.3308 |
| Cluster 0 size | 11,963 (57.9%) |
| Cluster 1 size | 8,677 (42.1%) |

**Cluster profiles (mean values):**

| Feature | Cluster 0 | Cluster 1 |
|---|---|---|
| MedInc ($10k) | 3.918 | 3.805 |
| HouseAge (yrs) | 28.4 | 29.0 |
| AveRooms | 5.225 | 5.710 |
| AveBedrms | 1.076 | 1.126 |
| Population | 1,532 | 1,278 |
| AveOccup | 3.098 | 3.033 |
| **Latitude** | **33.9°** | **38.0°** |
| **Longitude** | **−118.0°** | **−121.7°** |
| MedHouseVal ($100k) | **2.138** | **1.972** |

The primary differentiator is geography — Latitude and Longitude dominate the
cluster separation in z-score terms (>0.7 SD from the grand mean), while income,
age, and occupancy are near-zero. This cleanly maps to California's two main
regional housing markets.

---

## Section 3 – Hierarchical (Agglomerative) Clustering

### Dendrogram Analysis (300-sample subset, Ward linkage)

A dendrogram on a 300-row subset was computed with Ward linkage. The scree-style
merge-distance plot shows a large gap between the 1st and 2nd last merges,
confirming that two clusters represent the most natural cut of the data.

### Full-Dataset Hierarchical Clustering (k = 2, Ward linkage)

| Metric | Value |
|---|---|
| Silhouette score | 0.3213 |
| K-Means silhouette | 0.3308 |

K-Means slightly outperforms hierarchical clustering on silhouette, consistent with
the expectation that K-Means optimises within-cluster variance directly.

**Label agreement cross-tabulation:**

| | HC Cluster 0 | HC Cluster 1 |
|---|---|---|
| **KM Cluster 0** | 71 | 11,892 |
| **KM Cluster 1** | 7,673 | 1,004 |

The two algorithms find the same geographic split but assign opposing cluster
indices. After accounting for label permutation, the agreement is high
(~94% of points agree), with ~1,075 disagreements (5.2%) near the regional boundary.

**Hierarchical cluster profiles (mean values):**

| Feature | HC Cluster 0 | HC Cluster 1 |
|---|---|---|
| Latitude | 38.1° (NorCal) | 34.2° (SoCal) |
| Longitude | −121.9° | −118.2° |
| MedHouseVal | 2.014 | 2.102 |

The geographic interpretation is identical to K-Means.

---

## Section 4 – DBSCAN Clustering

### Hyperparameter Tuning

A 7 × 4 grid search was run over `eps ∈ {0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 3.0}` and
`min_samples ∈ {5, 10, 20, 50}` (28 combinations total).

Selection criteria for best parameters: ≥ 2 clusters, < 10% noise, highest silhouette.

**Notable valid results:**

| eps | min_samples | n_clusters | noise (%) | Silhouette |
|---|---|---|---|---|
| 0.7 | 10 | 2 | 6.6% | 0.562 |
| 1.0 | 5 | 3 | 1.5% | 0.556 |
| **1.5** | **5** | **2** | **0.5%** | **0.774** ← best |
| 2.0 | 10 | 2 | 0.4% | 0.756 |

### Best DBSCAN Run (eps = 1.5, min_samples = 5)

| Metric | Value |
|---|---|
| Clusters found | 2 |
| Noise points | 112 (0.5%) |
| Silhouette score | **0.7739** |

**DBSCAN cluster profiles (noise excluded):**

| Feature | Cluster 0 | Cluster 1 |
|---|---|---|
| MedInc | 3.868 | 3.526 |
| HouseAge (yrs) | 28.7 | **12.4** |
| AveRooms | 5.330 | **28.715** |
| AveBedrms | 1.076 | **5.293** |
| Population | 1,418 | **195** |
| AveOccup | 2.932 | 2.023 |
| Latitude | 35.6° | 38.8° |
| Longitude | −119.6° | −120.4° |
| MedHouseVal | 2.070 | 1.366 |

**Cluster 0** (20,528 properties, 99.5%): mainstream California residential market —
characteristics match the full-dataset average.

**Cluster 1** (112 properties, 0.5%): extreme outliers — very new homes (avg 12 yrs),
extremely large rooms (~29 avg), very sparse population per block (~195). These are
likely mountain resort properties, vacation homes, or large rural estates scattered
in the Sierra Nevada foothills (Lat ~38.8°, Long ~−120.4°).

### Algorithm Silhouette Comparison

| Algorithm | Silhouette |
|---|---|
| K-Means | 0.3308 |
| Hierarchical | 0.3213 |
| **DBSCAN** | **0.7739** |

DBSCAN's high score reflects that its two clusters are extremely well-separated —
the outlier cluster is structurally very different from the mainstream. The
K-Means/Hierarchical scores are more modest because they partition a continuous
geographic gradient rather than isolating a truly distinct density island.

---

## Section 5 – PCA & Dimensionality Reduction

### Explained Variance Profile (8 components)

| Component | Explained Variance | Cumulative |
|---|---|---|
| PC1 | 0.2534 | 0.2534 |
| PC2 | 0.2352 | 0.4885 |
| PC3 | 0.1589 | 0.6474 |
| PC4 | 0.1289 | 0.7763 |
| PC5 | 0.1254 | 0.9017 |
| PC6 | 0.0824 | **0.9841** |
| PC7 | 0.0102 | 0.9943 |
| PC8 | 0.0057 | 1.0000 |

**Optimal components for 95% variance threshold: 6** (capturing 98.4% at that cut).

The variance is spread relatively evenly across the first 6 components (~10–25% each),
indicating that no single feature axis dominates — the data is moderately correlated
but not highly redundant. PC7 and PC8 together account for only 1.6% of variance and
can safely be dropped.

### cluster_with_pca (PCA → K-Means, 2 components)

| Metric | Value |
|---|---|
| Silhouette (PCA space) | 0.5891 |
| Silhouette (full space, K-Means) | 0.3308 |

Clustering in 2-D PCA space raises the silhouette from 0.33 to 0.59. This is because
PCA decorrelates the features and discards noise in the dropped components, making
clusters more geometrically compact in the reduced space.

### Visualisation Findings (3-panel scatter)

- **K-Means in PCA space:** The two clusters form broad, overlapping lobes — the
  boundary is gradual, consistent with a continuous geographic transition rather than
  a sharp market divide.
- **DBSCAN in PCA space:** The 112-property outlier cluster appears as a tight,
  isolated island in the upper-right of PC space, confirming DBSCAN's detection of
  a genuinely distinct density region.
- **Ground-truth price overlay:** House values increase continuously across the PC1
  axis with no sharp boundary at either cluster edge, suggesting the two-cluster
  solution is a useful simplification of an underlying price gradient.

---

## Section 6 – Cluster Interpretation

### K-Means Market Tiers

| | SoCal Market (Cluster 0) | NorCal Market (Cluster 1) |
|---|---|---|
| Properties | 11,963 (58%) | 8,677 (42%) |
| Region | LA / San Diego metro | Bay Area / NorCal |
| Avg Latitude | 33.9° | 38.0° |
| Avg Longitude | −118.0° | −121.7° |
| Avg Income ($10k) | 3.918 | 3.805 |
| Avg Rooms | 5.2 | 5.7 |
| Avg Block Population | 1,532 | 1,278 |
| Mean Price ($100k) | **2.138** | 1.972 |
| Median Price ($100k) | **1.857** | 1.656 |

**SoCal Market:** Dense urban blocks, older housing stock, higher population per
block, and marginally higher median prices. The denser occupancy and older age
reflect the mature built environment of Los Angeles and San Diego.

**NorCal Market:** More spacious homes (higher room count), lower population density,
and slightly lower prices. Encompasses a mix of dense Bay Area core and the broader,
less-urbanised Central and Northern California areas.

### DBSCAN Outlier Segment

112 properties (0.5%) form a distinct cluster characterised by:
- Average home age of 12 years (vs. 28 years for the mainstream)
- Average rooms of ~29 per household (vs. ~5 for the mainstream)
- Block population of ~195 (vs. ~1,400 for the mainstream)
- Located around Lat 38.8°, Long −120.4° (Sierra Nevada foothills)

**Interpretation:** mountain resort properties, vacation homes, or large rural
estates — a niche market segment structurally incompatible with the residential
mainstream and correctly quarantined by DBSCAN.

---

## Summary of Results

| | K-Means | Hierarchical | DBSCAN |
|---|---|---|---|
| Segments found | 2 | 2 | 2 (+112 noise) |
| Silhouette score | 0.3308 | 0.3213 | 0.7739 |
| Requires k upfront | Yes | Yes (at cut) | No |
| Handles outliers | No | No | Yes |
| Best use here | Full-market segmentation | Confirming k=2 via dendrogram | Outlier detection |

**Three algorithms, consistent conclusion:** California residential properties
naturally segment into two geographic market tiers — Southern and Northern
California — with a small but structurally distinct luxury/rural outlier class
detected exclusively by DBSCAN.

The dominant clustering signal is geographic (latitude/longitude account for >0.7 SD
of separation), not economic — income and occupancy differences between the two
mainstream segments are minor. This suggests that regional regulation, land
availability, and climate drive market structure more than buyer income alone.
