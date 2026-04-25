# Real Estate Price Prediction Dashboard

A Streamlit dashboard for the Real Estate Price Prediction Engine. Provides three interactive pages:

- **Price Prediction** — Enter property features and get an estimated value from a trained ensemble model.
- **Property Recommendations** — Find similar properties using content-based (cosine similarity) or KNN filtering.
- **Market Segmentation** — Visualize market clusters via PCA dimensionality reduction and K-Means.

---

## Prerequisites

Trained model files must exist before launching the dashboard:

```
models/
├── stacking_ensemble.joblib
└── voting_ensemble.joblib
```

Run the Phase 4 notebook (`notebooks/04_ensemble_models.ipynb`) to generate them if they are missing.

---

## Running Locally

### 1. Clone the repository and navigate to the project root

```bash
git clone <repo-url>
cd 03_real_estate_prediction
```

### 2. Create and sync the environment with uv

[uv](https://docs.astral.sh/uv/) reads dependencies from `pyproject.toml` and manages the `.venv` automatically.

```bash
uv sync
```

This creates `.venv/` and installs all dependencies declared in `pyproject.toml`. To include the optional dev extras as well:

```bash
uv sync --extra dev
```

### 3. Launch the app

Run this command from the **project root** (not from inside `dashboard/`):

```bash
uv run streamlit run dashboard/app.py
```

Streamlit will print a local URL — open it in your browser:

```
Local URL:  http://localhost:8501
Network URL: http://192.168.x.x:8501
```

### Optional: specify a port

```bash
uv run streamlit run dashboard/app.py --server.port 8080
```

---

## Deploying on Streamlit Community Cloud

[Streamlit Community Cloud](https://streamlit.io/cloud) hosts public Streamlit apps for free directly from a GitHub repository. It detects `pyproject.toml` automatically and installs dependencies with pip.

### Steps

1. **Push your repository to GitHub** (must be public, or you must have a Streamlit account with private repo access).

2. **Go to** [share.streamlit.io](https://share.streamlit.io) and sign in with GitHub.

3. **Click "New app"** and fill in:

   | Field | Value |
   |---|---|
   | Repository | `your-github-username/your-repo-name` |
   | Branch | `main` (or whichever branch has the final code) |
   | Main file path | `dashboard/app.py` |

4. **Click "Deploy"**. Streamlit Cloud will install packages from `pyproject.toml` automatically.

5. Your app will be live at:
   ```
   https://<your-app-name>.streamlit.app
   ```

### Notes for Cloud Deployment

- The `models/` directory and `.joblib` files **must be committed to the repository** for the cloud app to load them. Verify they are not listed in `.gitignore`.
- Streamlit Cloud runs from the repository root, so the `sys.path` insert in `app.py` (`..` relative to `dashboard/`) resolves correctly.
- Free tier apps spin down after inactivity; the first load after spin-down may take ~30 seconds.

#### Dependency file precedence (`requirements.txt` vs `pyproject.toml`)

This project contains both `requirements.txt` and `pyproject.toml`. Streamlit Cloud scans for dependency files in this order and **stops at the first match**:

1. `requirements.txt` ← used if present
2. `pyproject.toml`
3. `Pipfile` / `setup.py`

Because `requirements.txt` exists, Streamlit Cloud will use it and ignore `pyproject.toml`. Locally, `uv sync` reads `pyproject.toml` only. This creates two sources of truth that can silently drift apart — a dependency added to one file but not the other will work locally but fail in production (or vice versa).

**Option A — Keep both files in sync (minimal change)**

Whenever you add or update a dependency, update both files:

```toml
# pyproject.toml
dependencies = [
    "new-package>=1.0.0",
    ...
]
```

```text
# requirements.txt
new-package>=1.0.0
```

Simple, but requires discipline to maintain.

**Option B — Delete `requirements.txt` (single source of truth)**

Remove `requirements.txt` so Streamlit Cloud falls back to `pyproject.toml`:

```bash
rm requirements.txt
git rm requirements.txt
git commit -m "chore: remove requirements.txt, use pyproject.toml as single source of truth"
```

After this, both `uv sync` (local) and Streamlit Cloud (production) read from the same file. This is the recommended approach if you are actively maintaining `pyproject.toml`.

---

## Project Structure (relevant files)

```
03_real_estate_prediction/
├── dashboard/
│   ├── app.py          # Streamlit application (this file's sibling)
│   └── README.md       # This file
├── src/
│   ├── data_loader.py
│   ├── clustering.py
│   ├── recommendation.py
│   └── ensemble.py
├── models/
│   ├── stacking_ensemble.joblib
│   └── voting_ensemble.joblib
└── pyproject.toml
```
