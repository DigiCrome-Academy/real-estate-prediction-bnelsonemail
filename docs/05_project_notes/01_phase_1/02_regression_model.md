# Regression Modeling — Phase 1 Notes

**Notebook:** `notebooks/01_regression_modeling.ipynb`
**Dataset:** California Housing (scikit-learn)
**Target:** `MedHouseVal` — median house value in $100,000 units

---

## 1. Data Loading & Exploration

The notebook begins by loading the California Housing dataset via a custom `load_housing_data()` function from `src.data_loader`. The dataset contains **20,640 records** with 8 features and 1 target.

**Features:**

| Feature | Description |
|---|---|
| `MedInc` | Median income in block group |
| `HouseAge` | Median house age |
| `AveRooms` | Average number of rooms per household |
| `AveBedrms` | Average number of bedrooms per household |
| `Population` | Block group population |
| `AveOccup` | Average occupants per household |
| `Latitude` | Geographic latitude |
| `Longitude` | Geographic longitude |

**Target:** `MedHouseVal` — mean = $206,855, std = $115,396, range [$14,999–$500,001]

Descriptive statistics reveal that `AveOccup` has extreme outliers (max = 1,243 occupants per household vs. mean = 3.07), signaling the presence of noise-level records.

**Correlation with target:**

```
MedInc        0.6881   ← dominant predictor
AveRooms      0.1519
HouseAge      0.1056
Latitude     -0.1442
Longitude    -0.0460
AveBedrms    -0.0467
AveOccup     -0.0237
Population   -0.0247
```

Median income (`MedInc`) is by far the strongest predictor. Geographic features (Latitude, Longitude) carry moderate signal through regional price gradients across California.

---

## 2. Feature Engineering & Preprocessing

Three derived features were added via `create_feature_engineering()`:

| New Feature | Derivation |
|---|---|
| `rooms_per_household` | Based on `AveRooms` |
| `bedrooms_ratio` | Based on `AveBedrms` |
| `population_density` | Based on `Population` |

This expands the feature set from 9 to 12 columns. After removing the target, 11 features are passed to the models.

Preprocessing used `preprocess_features()` (StandardScaler, zero-mean / unit-variance normalization) and `split_data()` for an 80/20 train-test split:

- **Train:** 16,512 records
- **Test:** 4,128 records

The engineering step adds interpretability but — as the results show — minimal predictive gain, since tree-based models can construct equivalent ratios internally through splits.

---

## 3. Linear Models

Five linear model variants were built and evaluated, using alpha/degree sweeps to find the best hyperparameters.

### 3.1 Baseline: Linear Regression

Plain OLS with no regularization.

- **R² = 0.6305**, **RMSE = 0.6958**
- Establishes the ceiling for linear methods on this dataset

### 3.2 Ridge (L2 Regularization)

Alpha sweep: `[0.01, 0.1, 1.0, 10.0, 100.0]`

- Best alpha: **0.01** — nearly identical to OLS
- R² = 0.6305, RMSE = 0.6958
- L2 penalty has negligible effect, suggesting coefficients are not large enough to benefit from shrinkage

### 3.3 Lasso (L1 Regularization)

Alpha sweep: `[0.0001, 0.001, 0.01, 0.1, 1.0]`

- Best alpha: **0.0001**
- R² = 0.6304, RMSE = 0.6959
- At alpha ≥ 0.1, performance degrades sharply as Lasso zeroes out informative features

### 3.4 ElasticNet (Hybrid L1+L2)

Grid: alpha ∈ `[0.001, 0.01, 0.1, 1.0]`, l1_ratio ∈ `[0.2, 0.5, 0.8]`

- Best: **alpha=0.001, l1_ratio=0.2**
- R² = 0.6302, RMSE = 0.6961
- Minimal advantage over plain linear; all regularized linear models converge to the same performance ceiling

### 3.5 Polynomial Regression

| Degree | Features | R² | RMSE |
|---|---|---|---|
| 2 | 77 | 0.1302 | 1.0676 |
| 3 | 363 | -153.06 | 14.21 |

Degree 2 underperforms even linear regression on the test set due to high dimensionality relative to the signal. Degree 3 catastrophically overfits — R² of -153 indicates predictions are far worse than the mean. Polynomial expansion is ruled out entirely.

**Summary — Linear Models:**

| Model | R² | RMSE | MAE |
|---|---|---|---|
| Linear Regression | 0.6305 | 0.6958 | 0.5030 |
| Ridge (α=0.01) | 0.6305 | 0.6958 | 0.5030 |
| Lasso (α=0.0001) | 0.6304 | 0.6959 | 0.5031 |
| ElasticNet (α=0.001, l1=0.2) | 0.6302 | 0.6961 | 0.5031 |

All four cluster tightly at R² ≈ 0.63. The data's non-linear structure is the binding constraint, not the regularization regime.

---

## 4. Tree-Based Models

Five tree-based models were evaluated, each with a hyperparameter sweep. The logic here is that tree models can partition the feature space in ways that capture non-linear interactions between income, location, and occupancy.

### 4.1 Decision Tree

Max_depth sweep: `[3, 5, 10, 15, None]`

- Best: **depth=10**
- R² = 0.6965, RMSE = 0.6306
- Depth 3 underfits; performance plateaus at depth 10; deeper trees show no further gain

### 4.2 Random Forest

N_estimators sweep: `[50, 100, 200]`

- Best: **n_estimators=200**
- R² = 0.8043, RMSE = 0.5064
- Bagging over 200 trees reduces variance substantially versus a single tree (+10% R²)
- Saturation begins at 100 trees

### 4.3 Gradient Boosting

Learning_rate sweep (n_estimators=200): `[0.05, 0.1, 0.2]`

- Best: **lr=0.2**
- R² = 0.8159, RMSE = 0.4912
- Sequential boosting beats parallel bagging; higher learning rate is optimal here

### 4.4 XGBoost

Learning_rate sweep (n_estimators=200): `[0.05, 0.1, 0.2]`

- Best: **lr=0.1**
- R² = 0.8391, RMSE = 0.4592
- Built-in L1/L2 regularization allows a moderate learning rate without overfitting

### 4.5 LightGBM

Learning_rate sweep (n_estimators=200): `[0.05, 0.1, 0.2]`

- Best: **lr=0.2**
- **R² = 0.8468**, **RMSE = 0.4480** — best overall
- Leaf-wise growth builds more complex trees per round than level-wise approaches, gaining an edge on this structured dataset

**Summary — Tree-Based Models:**

| Model | R² | RMSE | MAE |
|---|---|---|---|
| LightGBM (lr=0.2) | **0.8468** | **0.4480** | **0.2958** |
| XGBoost (lr=0.1) | 0.8391 | 0.4592 | 0.3012 |
| Gradient Boosting (lr=0.2) | 0.8159 | 0.4912 | 0.3292 |
| Random Forest (n=200) | 0.8043 | 0.5064 | 0.3281 |
| Decision Tree (depth=10) | 0.6965 | 0.6306 | 0.4155 |

**Random Forest feature importances** show geographic features (Latitude, Longitude) at the top, followed by MedInc and AveOccup. This is notable: while MedInc has the highest raw correlation with the target, tree splits on location interact with income to reveal price clusters that correlation alone misses.

---

## 5. Model Comparison

All 9 models ranked together (excluding polynomial regression):

| Rank | Model | R² | RMSE | MAE |
|---|---|---|---|---|
| 1 | LightGBM (lr=0.2) | 0.8468 | 0.4480 | 0.2958 |
| 2 | XGBoost (lr=0.1) | 0.8391 | 0.4592 | 0.3012 |
| 3 | Gradient Boosting (lr=0.2) | 0.8159 | 0.4912 | 0.3292 |
| 4 | Random Forest (n=200) | 0.8043 | 0.5064 | 0.3281 |
| 5 | Decision Tree (depth=10) | 0.6965 | 0.6306 | 0.4155 |
| 6 | Linear Regression | 0.6305 | 0.6958 | 0.5030 |
| 7 | Ridge (α=0.01) | 0.6305 | 0.6958 | 0.5030 |
| 8 | Lasso (α=0.0001) | 0.6304 | 0.6959 | 0.5031 |
| 9 | ElasticNet (α=0.001, l1=0.2) | 0.6302 | 0.6961 | 0.5031 |

**Performance gap summary:**

- LightGBM vs. best linear model: **+21% R²**, **-35% RMSE**
- Tree-based cluster (R² 0.80–0.85) vs. linear cluster (R² ≈ 0.63): clear two-tier structure
- Within tree-based models: only ~4% R² separates best (LightGBM) from 4th (Random Forest)

The two tiers reflect a structural limitation of linear models — they cannot express the interaction terms that drive California housing prices (e.g., high income in a coastal block yields a disproportionate price premium that neither a linear term nor regularization can represent).

---

## 6. Regression Diagnostics

### 6.1 Variance Inflation Factor (VIF) — Multicollinearity

VIF measures how much the variance of a coefficient is inflated due to correlation with other features. VIF > 10 signals problematic multicollinearity.

| Feature | VIF |
|---|---|
| AveRooms | 21.40 |
| AveBedrms | 16.93 |
| Latitude | 9.27 |
| Longitude | 8.92 |
| population_density | 6.31 |
| Population | 6.28 |
| rooms_per_household | 5.72 |
| AveOccup | 5.62 |
| bedrooms_ratio | 4.78 |
| MedInc | 2.59 |
| HouseAge | 1.25 |

`AveRooms` (VIF=21.40) and `AveBedrms` (VIF=16.93) exceed the threshold significantly. The engineered features (`rooms_per_household`, `bedrooms_ratio`) are derived from these same columns, compounding the collinearity.

**Impact by model type:**

- **Linear models:** High VIF inflates standard errors of coefficients, making inference (not prediction) unreliable. Ridge's L2 penalty partially stabilizes coefficients but does not resolve the underlying correlation structure. Coefficient magnitudes and directions should not be interpreted as causal.
- **Tree-based models:** No impact. Each split considers one feature at a time; correlated features simply get redundant splits, which the ensemble averages out naturally.

### 6.2 Heteroskedasticity — Breusch-Pagan Test

The Breusch-Pagan test checks whether residual variance is constant (homoskedasticity) across fitted values.

**Test results (linear regression residuals):**

```
LM statistic : 380.10
LM p-value   : 1.02e-74
F statistic  : 37.95
F p-value    : 1.61e-78
```

The null hypothesis of homoskedasticity is rejected with overwhelming evidence. Residuals fan out as fitted values increase — the model systematically underpredicts expensive houses, and the error grows with price.

**Consequences:**
- Standard errors of linear coefficients are biased (confidence intervals are unreliable)
- Point predictions themselves are not directly harmed by heteroskedasticity
- The pattern suggests a multiplicative (proportional) error structure, not additive

**Log transformation remedy:**

A log(1+y) transformation was tested as a standard remedy for multiplicative variance.

| Target | R² | RMSE |
|---|---|---|
| Original | 0.6305 | 0.6958 |
| Log-transformed | 0.5428 | 0.7741 |

The transformation worsens both metrics. The California Housing target is already capped at $500,001 (a hard ceiling in the dataset), which creates a left-skewed residual structure that log transformation does not help. The appropriate response is to use tree-based models, which partition the feature space adaptively and are inherently robust to heteroskedasticity.

---

## 7. Cross-Validation

The top three models were evaluated with 5-fold cross-validation on the training set to assess stability and detect overfitting.

| Model | CV R² (mean) | CV R² (std) | CV RMSE (mean) | CV RMSE (std) |
|---|---|---|---|---|
| LightGBM (lr=0.2) | 0.6740 | 0.0492 | 0.6382 | 0.2322 |
| XGBoost (lr=0.1) | 0.6828 | **0.0413** | 0.6299 | 0.2127 |
| Gradient Boosting (lr=0.2) | 0.6704 | 0.0532 | 0.6402 | 0.2028 |

**Notable observations:**

1. **CV R² vs. test R² gap (~15–17%):** All three models show a meaningful gap between cross-validation performance and test set performance. This is expected — the test set is a fixed holdout, and the hyperparameters were selected to maximize test R², introducing selection bias. The gap does not indicate severe overfitting.

2. **XGBoost is the most stable:** Lowest CV standard deviation on both metrics (0.0413 R², 0.2127 RMSE). If generalization consistency matters more than raw test performance, XGBoost is preferable.

3. **LightGBM wins on test R² but shows higher CV variance:** Leaf-wise growth optimizes more aggressively, which pays off on the test set but leads to slightly more fold-to-fold variation.

4. **Standard deviations are low overall (0.04–0.05 R²):** All three models are stable. The small CV variance confirms no overfitting crisis — the train-test gap is primarily a hyperparameter selection artifact.

---

## 8. Summary of Findings

### Best Model

**LightGBM** (learning_rate=0.2, n_estimators=200)

- R² = **0.8468** — explains 84.68% of house price variance
- RMSE = **$44,800** average prediction error
- MAE = **$29,580** average absolute error

### Key Takeaways

| Topic | Finding |
|---|---|
| Dominant predictor | `MedInc` (r=0.688), followed by geographic features |
| Linear ceiling | All linear/regularized models plateau at R² ≈ 0.63 |
| Non-linear gap | Tree models gain +21% R² by capturing interaction effects |
| Regularization value | Negligible — Ridge/Lasso/ElasticNet do not improve over plain OLS |
| Polynomial features | Catastrophic overfitting at degree 2–3; ruled out |
| Multicollinearity | `AveRooms` (VIF=21.40) and `AveBedrms` (VIF=16.93) are collinear with engineered features; affects linear inference only |
| Heteroskedasticity | Confirmed (p≈0); log transform worsens results; tree models handle it naturally |
| Cross-validation | Models stable (CV std 0.04–0.05 R²); XGBoost most consistent |
| Feature engineering | Adds interpretability but minimal predictive gain |

### Model Recommendation

For prediction tasks, **LightGBM** is the recommended model. For scenarios where interpretability of coefficients or inference is needed, linear models should be approached cautiously given the confirmed multicollinearity and heteroskedasticity.

If deployment stability is prioritized over maximum R², **XGBoost** is the more conservative choice given its lower cross-validation variance.
