[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/oNipybyq)
# Real Estate Price Prediction Engine

A comprehensive regression-based pricing engine for real estate valuation, combining supervised learning, unsupervised market segmentation, and recommendation systems.

## Project Structure

```
real-estate-prediction-engine/
├── .github/
│   └── workflows/
│       └── autograding.yml          # GitHub Actions autograding
├── notebooks/
│   ├── 01_regression_modeling.ipynb  # Phase 1: Regression models
│   ├── 02_clustering_analysis.ipynb  # Phase 2: Market segmentation
│   └── 03_recommendation_system.ipynb # Phase 3: Recommendations
├── src/
│   ├── __init__.py
│   ├── data_loader.py               # Data loading & preprocessing
│   ├── regression.py                # Phase 1: Regression models
│   ├── clustering.py                # Phase 2: Clustering & PCA
│   ├── recommendation.py            # Phase 3: Recommendation system
│   └── ensemble.py                  # Phase 4: Model ensembles
├── tests/
│   ├── conftest.py                  # Shared test fixtures
│   ├── test_phase1/
│   │   ├── test_data_preprocessing.py
│   │   ├── test_linear_models.py
│   │   ├── test_tree_models.py
│   │   └── test_regression_diagnostics.py
│   ├── test_phase2/
│   │   ├── test_kmeans.py
│   │   ├── test_hierarchical.py
│   │   ├── test_dbscan.py
│   │   └── test_pca.py
│   ├── test_phase3/
│   │   ├── test_content_based.py
│   │   ├── test_collaborative.py
│   │   └── test_hybrid.py
│   └── test_phase4/
│       ├── test_voting_ensemble.py
│       └── test_stacking_ensemble.py
├── data/                            # Data directory (gitignored)
├── models/                          # Saved models (gitignored)
├── dashboard/
│   └── app.py                       # Streamlit dashboard
├── requirements.txt
├── setup.py
└── README.md
```

## Setup

```bash
# Clone the repository
git clone <your-repo-url>
cd real-estate-prediction-engine

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt

# Download the dataset
python src/data_loader.py
```

## Phases & Deliverables

### Phase 1: Advanced Regression Modeling (Week 1–2)
- Implement functions in `src/regression.py`
- Complete notebook `notebooks/01_regression_modeling.ipynb`
- **Tests:** `tests/test_phase1/`

### Phase 2: Market Segmentation via Clustering (Week 2–3)
- Implement functions in `src/clustering.py`
- Complete notebook `notebooks/02_clustering_analysis.ipynb`
- **Tests:** `tests/test_phase2/`

### Phase 3: Recommendation System (Week 3–4)
- Implement functions in `src/recommendation.py`
- Complete notebook `notebooks/03_recommendation_system.ipynb`
- **Tests:** `tests/test_phase3/`

### Phase 4: Model Ensemble & Deployment Prep (Week 4)
- Implement functions in `src/ensemble.py`
- **Tests:** `tests/test_phase4/`

## Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run tests for a specific phase
pytest tests/test_phase1/ -v
pytest tests/test_phase2/ -v
pytest tests/test_phase3/ -v
pytest tests/test_phase4/ -v

# Run with score summary
pytest tests/ -v --tb=short
```

## Autograding

Tests run automatically on push via GitHub Actions. Each phase is weighted:

| Component                | Weight | Test Directory      |
|--------------------------|--------|---------------------|
| Regression Model Quality | 25%    | `test_phase1/`      |
| Clustering Implementation| 20%    | `test_phase2/`      |
| Recommendation System    | 20%    | `test_phase3/`      |
| Feature Engineering & PCA| 15%    | (within phases 1&2) |
| Model Ensemble           | 10%    | `test_phase4/`      |
| Dashboard & Presentation | 10%    | Manual review       |

## Evaluation Criteria

- **Regression Models:** Multiple models implemented, proper metrics, diagnostics
- **Clustering:** Multiple algorithms, optimal cluster selection, interpretation
- **Recommendations:** System design, relevance, evaluation metrics
- **Feature Engineering:** PCA, feature creation, multicollinearity handling
- **Ensemble:** Implementation quality, performance improvement over base models
- **Dashboard:** UI quality, visualizations, documentation
