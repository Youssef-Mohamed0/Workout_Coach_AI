# Workout Coach AI (SmartGym AI)

> **Dual-Track Real-Time Biomechanics & Classical ML Fitness Lab** — Combining browser-based **MediaPipe Pose** computer vision with a multi-stage **Scikit-Learn** pipeline trained on 973 real Kaggle gym member records.

---

## Overview

**Workout Coach AI** analyzes gym workouts through two synchronized tracks in a single web dashboard:

1. **Track 1 — Live Pose & Movement Vision (Browser):** Uses MediaPipe Pose JS to track joint angles, rep counts (clean vs. bad form), spine tilt (`torso_lean_deg`), and left/right symmetry across 9 exercises plus an Auto-Detect mode.
2. **Track 2 — Biometric ML Pipeline (FastAPI + Scikit-Learn):** Combines user body/smartwatch metrics with live pose telemetry to project athletes onto a 3D PCA space, assign dynamic K-Means clusters, flag anomalies with Isolation Forest, estimate overtraining/injury risk, and predict calories burned.

---

## Key Features

- **10 Exercise Modes:** Auto-Detect (Full Body), Bicep Curl, Overhead Shoulder Press, Side Lateral Raise, Push-Up / Bench Press, Lat Pulldown / Back Row, Squat, Deadlift, Lunges, and Plank.
- **Zero-Leakage ML Pipeline:** Performs an 80/20 stratified `train_test_split` (`SEED = 42`) before fitting `StandardScaler`, `PCA`, `KMeans`, and `IsolationForest` strictly on training data.
- **Dynamic Cluster Naming:** Automatically generates human-readable English and Arabic cluster profiles directly from standardized cluster centroids (`backend/cluster_naming.py`).
- **Smartwatch-Optional Inference:** Requires only 4 body/session fields (`Age`, `Weight_kg`, `Height_m`, `Session_Duration_min`); missing smartwatch fields are imputed with training medians, and physiological risk is only scored when heart-rate/hydration values are actually provided.
- **Bilingual Coach Verdict:** Displays real-time Arabic/English coaching feedback that distinguishes between form issues (spine lean, asymmetry, bad reps) and physiological risk flags (high HR, low hydration, long session).

---

## Architecture & ML Pipeline

The backend pipeline (`backend/ml_pipeline.py`) trains on `backend/data/gym_members_clean.csv` (**973 records, 10 physiological features**) and bundles all artifacts into `backend/models/smart_gym_models.pkl`:

| Stage | Model / Component | Purpose |
| :--- | :--- | :--- |
| **Preprocessing** | `StandardScaler` | Standardizes the 10 biometric features (`Age`, `Weight_kg`, `Height_m`, `Max_BPM`, `Avg_BPM`, `Resting_BPM`, `Session_Duration_min`, `Fat_Percentage`, `Water_Intake_liters`, `BMI`) |
| **Stage 1 — Dimensionality Reduction** | `PCA (n_components=3)` | Compresses 10D biometrics into 3D coordinates (top 2 components plotted on the dashboard) |
| **Stage 2 — Clustering** | `KMeans (k=3)` + `DBSCAN` | Segments athletes into 3 profiles and appends `cluster_id` + `pca_1` + `pca_2` (13 enhanced features total) |
| **Stage 3 — Anomaly Detection** | `IsolationForest (contamination=0.06)` | Flags statistically unusual biometric patterns alongside rule-based pose anomaly checks |
| **Stage 4A — Risk Classification** | `RandomForestClassifier` | Predicts `Overtraining_Risk_Label` (benchmarked against Decision Tree depth-3, 10-feature RF, and Majority Class) |
| **Stage 4B — Calorie Prediction** | `RandomForestRegressor` | Predicts `Calories_Burned` (benchmarked against Linear Regression and Mean Baseline) |

---

## Project Structure

```text
Workout_Coach_AI/
├── backend/
│   ├── api.py                   # FastAPI server, Pydantic validation, CORS & routes
│   ├── config.py                # Paths, FEATURE_COLS & PHYSIO_KEYS
│   ├── ml_pipeline.py           # End-to-end training, evaluation & model serialization
│   ├── features.py              # Shared 13-feature builder (train & inference parity)
│   ├── cluster_naming.py        # Automatic EN/AR cluster descriptions from z-scores
│   ├── pose_analyzer.py         # Model loader & biometric + pose fusion engine
│   ├── pose_math.py             # 2D joint angles, torso lean, symmetry & movement risk
│   └── data/
│       └── gym_members_clean.csv              # Cleaned dataset (973 rows, 14 cols)
├── frontend/
│   └── index.html               # Single-page dashboard (Tailwind, Chart.js, MediaPipe Pose)
├── notebooks/
│   ├── 01_eda_pca.ipynb         # Team workspace: EDA & PCA
│   ├── 02_clustering.ipynb      # Team workspace: K-Means & DBSCAN clustering
│   ├── 03_anomaly_clf.ipynb     # Team workspace: Isolation Forest & Risk Classification
│   └── 04_regression_eval.ipynb # Team workspace: Calories Regression & Evaluation
├── requirements.txt             # Core backend & ML dependencies
└── requirements-notebooks.txt   # JupyterLab & plotting dependencies for notebooks
```

---

## Quick Start

> **Prerequisite:** Python **3.11+** is recommended (matches the pinned `requirements.txt` versions).

### 1. Clone & Create a Virtual Environment

```bash
git clone https://github.com/youssef-mohamed0/Workout_Coach_AI.git
cd Workout_Coach_AI

python -m venv .venv
```

Activate the virtual environment:

- **Windows (PowerShell):** `.venv\Scripts\Activate.ps1`
- **Windows (CMD):** `.venv\Scripts\activate.bat`
- **macOS / Linux:** `source .venv/bin/activate`

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Train & Save the Backend Models (Required Once)

`backend/models/smart_gym_models.pkl` is not committed to Git. Generate it before starting the server:

```bash
python -m backend.ml_pipeline
```

This prints the full held-out test metrics JSON (F1, ROC-AUC, R², MAE, RMSE, Silhouette scores, and cluster profiles) and saves `backend/models/smart_gym_models.pkl`.

### 4. Run the Web App

```bash
uvicorn backend.api:app --reload
```

Open **`http://127.0.0.1:8000`** in your browser.

> **Important:** Always open the app through `http://127.0.0.1:8000` or `http://localhost:8000` (not directly as a local `file://` HTML file or a different port), because CORS in `backend/api.py` is restricted to port `8000`.

---

## API Reference

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `GET` | `/` | Serves the interactive dashboard (`frontend/index.html`) |
| `GET` | `/api/dataset-overview` | Returns PCA scatter points, cluster profiles, anomaly counts, and held-out test metrics for all 973 athletes |
| `POST` | `/api/analyze-biometrics` | Validates body/smartwatch + pose inputs and returns PCA coordinates, cluster, predicted calories, risk score, and anomaly status |

If `smart_gym_models.pkl` has not been generated yet, API endpoints return **HTTP `503 Service Unavailable`** with instructions to run `python -m backend.ml_pipeline`.

---

## Team Notebooks Guide

The `notebooks/` directory is completely decoupled from the FastAPI backend—editing or running notebooks will not affect the web server.

1. Install both dependency files:
   ```bash
   pip install -r requirements.txt
   pip install -r requirements-notebooks.txt
   ```
2. Launch JupyterLab from the **project root**:
   ```bash
   jupyter lab
   ```
3. Load datasets using relative paths from inside `notebooks/`:
   - Data: `../backend/data/gym_members_clean.csv`
