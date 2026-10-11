# DiabetesRisk-AI

An end-to-end Machine Learning system for diabetes risk prediction. The project combines unsupervised clustering (K-Means) with supervised classification (Logistic Regression, KNN, SVM, Random Forest) to segment patients into risk profiles and predict the group of new observations. It ships as a fully containerised stack — FastAPI prediction service, Streamlit UI, MLflow experiment tracking, Apache Airflow pipeline orchestration, and PostgreSQL metadata store — all managed through Docker Compose.

> **Disclaimer:** The risk level is derived from statistical patterns in the dataset and does **not** constitute a medical diagnosis.

---

## Table of Contents

1. [Architecture Overview](#1-architecture-overview)
2. [Project Structure](#2-project-structure)
3. [Technology Stack](#3-technology-stack)
4. [Dataset](#4-dataset)
5. [Data Pipeline](#5-data-pipeline)
6. [Docker Services](#6-docker-services)
7. [API Reference](#7-api-reference)
8. [Getting Started](#8-getting-started)
9. [Notebooks](#9-notebooks)
10. [Limitations](#10-limitations)
11. [Future Improvements](#11-future-improvements)
12. [Author & Academic Context](#12-author--academic-context)

---

## 1. Architecture Overview

```
┌──────────────┐        ┌──────────────┐        ┌──────────────┐
│   Streamlit  │──POST──│   FastAPI     │──load──│   models/    │
│   :8501      │  /predict│   :8000     │        │   *.pkl      │
└──────────────┘        └──────────────┘        └──────────────┘
                              │
                        (reads model)
                              │
┌──────────────┐        ┌──────────────┐        ┌──────────────┐
│   Airflow    │──runs──│   src/       │──logs──│   MLflow     │
│   :8080      │        │  pipeline    │        │   :5000      │
└──────────────┘        └──────────────┘        └──────────────┘
                                                      │
                                                ┌──────────────┐
                                                │  PostgreSQL  │
                                                │  (metadata)  │
                                                └──────────────┘
```

**Flow summary:**

1. **Airflow** triggers the `retrain_diabetes_dag` on a monthly schedule.
2. The DAG executes three sequential tasks: **clean data → cluster → train & evaluate models**.
3. Every clustering and classification run logs parameters, metrics, and artifacts to **MLflow**, backed by **PostgreSQL**.
4. Trained model pipelines (`.pkl`) are saved to the `models/` directory.
5. **FastAPI** loads the `Logistic_Regression_model.pkl` at prediction time and exposes a `POST /predict` endpoint.
6. **Streamlit** provides a web form that sends patient data to the API and displays the prediction result.

---

## 2. Project Structure

```
DiabetesRisk-AI/
│
├── app/
│   ├── api/
│   │   └── main.py                  # FastAPI application (prediction endpoint)
│   └── ui/
│       └── streamlit_app.py         # Streamlit web interface
│
├── src/
│   ├── data_profiler.py             # DataProfiler class – EDA utilities
│   ├── data_cleaner.py              # DataCleaner class – cleaning pipeline
│   ├── clustering.py                # Clustering class – K-Means with MLflow tracking
│   └── classification/
│       └── classification.py        # Classification class – train, evaluate & compare models
│
├── dags/
│   └── retrain_diabetes_dag.py      # Airflow DAG – monthly retraining pipeline
│
├── data/
│   ├── raw/
│   │   └── dataset-diabete.csv      # Original clinical dataset
│   └── processed/
│       ├── cleaned_data_diabetes.csv    # Output of DataCleaner
│       └── clustered_data_diabetes.csv  # Output of Clustering (includes Cluster column)
│
├── models/
│   ├── Logistic_Regression_model.pkl
│   ├── KNN_model.pkl
│   ├── Random Forest_model.pkl
│   └── SVM_model.pkl
│
├── notebooks/
│   ├── 01_data_profiling.ipynb
│   ├── 02_eda_guide_graphes.ipynb
│   ├── 03_clustering.ipynb
│   └── 04_cluster_analysis.ipynb
│
├── docker/
│   ├── Dockerfile.api               # Python 3.11 + FastAPI/Uvicorn
│   ├── Dockerfile.streamlit         # Python 3.11 + Streamlit
│   ├── Dockerfile.airflow           # Apache Airflow 2.9.3 + project deps
│   ├── Dockerfile.mlflow            # Python 3.11 + MLflow server
│   └── init-db.sql                  # Creates the `mlflow` database on first run
│
├── tests/
│   └── test_model.py                # Basic model prediction smoke test
│
├── docker-compose.yml               # Orchestrates all services
├── requirements.txt                 # Full Python dependencies (Airflow image)
├── req_api.txt                      # Lightweight dependencies (API image)
├── .env                             # Environment variables for Docker Compose
├── .dockerignore
└── README.md
```

---

## 3. Technology Stack

| Category | Technology | Purpose |
|---|---|---|
| **Language** | Python 3.11 | Core language for all services |
| **Data** | Pandas, NumPy | Data manipulation and analysis |
| **Visualisation** | Matplotlib, Seaborn | EDA charts (notebooks) |
| **ML – Preprocessing** | Scikit-learn (`StandardScaler`, `KNNImputer`) | Feature scaling & missing-value imputation |
| **ML – Clustering** | Scikit-learn (`KMeans`) | Unsupervised patient segmentation |
| **ML – Classification** | Scikit-learn (`LogisticRegression`, `KNeighborsClassifier`, `SVC`, `RandomForestClassifier`) | Supervised risk group prediction |
| **ML – Tuning** | Scikit-learn (`GridSearchCV`, `Pipeline`) | Hyperparameter optimisation with cross-validation |
| **Serialisation** | Joblib | Model pipeline persistence (`.pkl` files) |
| **Experiment Tracking** | MLflow 2.17.2 | Log params, metrics, artifacts & confusion matrices |
| **API** | FastAPI 0.115.4 + Uvicorn | REST prediction service |
| **UI** | Streamlit 1.39.0 | Interactive web form for predictions |
| **Orchestration** | Apache Airflow 2.9.3 | Monthly retraining DAG |
| **Database** | PostgreSQL 15 | Metadata backend for Airflow & MLflow |
| **Containerisation** | Docker & Docker Compose | Multi-service deployment |

---

## 4. Dataset

**File:** `data/raw/dataset-diabete.csv`

The dataset contains clinical observations of patients with the following features:

| Feature | Description |
|---|---|
| `Pregnancies` | Number of pregnancies |
| `Glucose` | Plasma glucose concentration |
| `BloodPressure` | Diastolic blood pressure (mm Hg) |
| `SkinThickness` | Triceps skinfold thickness (mm) |
| `Insulin` | 2-hour serum insulin (μU/ml) |
| `BMI` | Body mass index (kg/m²) |
| `DiabetesPedigreeFunction` | Diabetes pedigree function (family history indicator) |
| `Age` | Age in years |

After clustering, a `Cluster` column is appended and used as the target variable for supervised classification.

---

## 5. Data Pipeline

The pipeline is implemented across three Python modules and orchestrated by an Airflow DAG.

### 5.1 Data Cleaning (`src/data_cleaner.py`)

The `DataCleaner` class performs the following steps:

1. Load the raw CSV.
2. Drop the spurious `Unnamed: 0` index column (if present).
3. Replace biologically impossible zeros (`Glucose`, `BloodPressure`, `SkinThickness`, `Insulin`, `BMI`) with `NaN`.
4. Detect outliers via the IQR method and cap them (winsorisation).
5. Impute remaining missing values with `KNNImputer` (k=3).
6. Save the cleaned dataset to `data/processed/cleaned_data_diabetes.csv`.

### 5.2 Clustering (`src/clustering.py`)

The `Clustering` class performs the following steps:

1. Load the cleaned data.
2. Standardise features with `StandardScaler`.
3. Determine the optimal number of clusters using the **Elbow Method** and **Silhouette Score** (utilities available, default `n_clusters=2`).
4. Fit a `KMeans` model.
5. Append the `Cluster` label to the dataset.
6. Save the result to `data/processed/clustered_data_diabetes.csv`.
7. Log parameters and artifacts to MLflow (experiment: `DiabetsRisk-AI-Clustering`).

### 5.3 Classification (`src/classification/classification.py`)

The `Classification` class performs the following steps:

1. Load the clustered dataset.
2. Split into features (`X`) and target (`Cluster` column / `y`).
3. Train/test split (80/20, stratified).
4. For each algorithm, build a `Pipeline` of `StandardScaler` + model, then run `GridSearchCV` (5-fold CV):
   - **Logistic Regression** – grid over `C`, `max_iter`, `solver`.
   - **SVM** – grid over `C`, `kernel`, `gamma`.
   - **Random Forest** – grid over `n_estimators`, `max_depth`, `min_samples_split`.
   - **KNN** – grid over `n_neighbors`, `weights`, `algorithm`.
5. Evaluate all best estimators on the test set: **Accuracy**, **Precision**, **Recall**, **F1-score**, **Confusion Matrix**.
6. Log every metric, parameter, and confusion matrix to MLflow (experiment: `DiabetsRisk-AI-Classification`).
7. Save each model pipeline to `models/<ModelName>_model.pkl` and log the artifact to MLflow.

### 5.4 Airflow DAG (`dags/retrain_diabetes_dag.py`)

| Property | Value |
|---|---|
| DAG ID | `retrain_diabetes_dag` |
| Schedule | `@monthly` |
| Catchup | `False` |

**Tasks (sequential):**

```
clean_data  ──▶  cluster_data  ──▶  train_model
```

1. `clean_data` — runs `DataCleaner.run_cleaning_pipeline()`.
2. `cluster_data` — runs `Clustering.run_clustering_pipeline(n_clusters=2)`.
3. `train_model` — runs `Classification.run_pipeline(target_column="Cluster")`.

### 5.5 Data Profiling (`src/data_profiler.py`)

The `DataProfiler` class provides exploratory analysis utilities:

- Shape & dtype inspection
- Missing value counts
- Duplicate detection
- Zero-value counts per column
- Numeric distribution statistics (`.describe()`)
- Outlier counts via IQR
- Correlation matrix

These methods are used interactively in notebooks and do not form part of the automated pipeline.

---

## 6. Docker Services

All services are defined in `docker-compose.yml` and configured via the `.env` file.

| Service | Container Name | Image | Port | Description |
|---|---|---|---|---|
| `postgres` | `diabete-postgres` | `postgres:15` | — | Metadata store for Airflow & MLflow |
| `mlflow` | `diabete-mlflow` | `diabete-mlflow` | `5000` | Experiment tracking server |
| `api` | `diabete-api` | `diabete-api` | `8000` | FastAPI prediction service |
| `streamlit` | `diabete-streamlit` | `diabete-streamlit` | `8501` | Streamlit web UI |
| `airflow-init` | `diabete-airflow-init` | `diabete-airflow` | — | One-shot: DB migration & admin user creation |
| `airflow-webserver` | `diabete-airflow-web` | `diabete-airflow` | `8080` | Airflow web UI |
| `airflow-scheduler` | `diabete-airflow-scheduler` | `diabete-airflow` | — | Airflow task scheduler |

**Named volumes:** `postgres_data`, `mlflow_artifacts`, `airflow_logs`.

### Service Dependencies

```
postgres (healthy)
  ├── mlflow (healthy)
  │     ├── api
  │     │     └── streamlit
  │     ├── airflow-init (completed)
  │     │     ├── airflow-webserver
  │     │     └── airflow-scheduler
```

---

## 7. API Reference

The API is built with **FastAPI** and served by **Uvicorn** on port `8000`.

### `GET /`

Health check.

**Response:**
```json
{
  "message": "Diabetes Risk API is running"
}
```

### `POST /predict`

Predict the diabetes risk group for a patient.

**Request body (JSON):**

```json
{
  "Pregnancies": 2.0,
  "Glucose": 140.0,
  "BloodPressure": 80.0,
  "SkinThickness": 25.0,
  "Insulin": 100.0,
  "BMI": 32.5,
  "DiabetesPedigreeFunction": 0.6,
  "Age": 35.0
}
```

All fields are required and must be `float`.

**Response (success):**

```json
{
  "prediction": "high risk",
  "probability_high_risk": 0.72,
  "probability_low_risk": 0.28
}
```

| Field | Type | Description |
|---|---|---|
| `prediction` | `string` | `"high risk"` (class 0) or `"low risk"` (class 1) |
| `probability_high_risk` | `float` | Probability of class 0 |
| `probability_low_risk` | `float` | Probability of class 1 |

> **Note:** The API loads the `Logistic_Regression_model.pkl` pipeline from the `models/` directory. This pipeline includes `StandardScaler` followed by `LogisticRegression`, so raw (unscaled) values should be sent.

### `GET /allo`

Debug endpoint — returns the resolved base directory and model file path.

---

## 8. Getting Started

### Prerequisites

- **Docker** and **Docker Compose** (v2+)
- **Git**

### 8.1 Clone the Repository

```bash
git clone https://github.com/abderrazakerrifaouy/DiabetesRisk-AI.git
cd DiabetesRisk-AI
```

### 8.2 Environment Variables

The `.env` file at the project root contains the following variables (default values shown):

```env
POSTGRES_USER=admin
POSTGRES_PASSWORD=admin
POSTGRES_DB=airflow
MLFLOW_TRACKING_URI=http://mlflow:5000
AIRFLOW_ADMIN_USER=admin
AIRFLOW_ADMIN_PASSWORD=admin
```

Update `POSTGRES_PASSWORD` and `AIRFLOW_ADMIN_PASSWORD` for non-development deployments.

### 8.3 Build and Start All Services

```bash
docker compose up --build -d
```

Verify that all containers are healthy:

```bash
docker compose ps
```

### 8.4 Access the Applications

| Application | URL |
|---|---|
| Streamlit UI | http://localhost:8501 |
| FastAPI Docs (Swagger) | http://localhost:8000/docs |
| MLflow Tracking UI | http://localhost:5000 |
| Airflow Web UI | http://localhost:8080 |

Log in to Airflow with the credentials defined in `.env` (`admin` / `admin` by default).

### 8.5 Trigger the Retraining Pipeline

1. Open the **Airflow UI** at http://localhost:8080.
2. Enable the `retrain_diabetes_dag` DAG.
3. Trigger it manually or wait for the monthly schedule.
4. Monitor task execution: `clean_data → cluster_data → train_model`.
5. Review logged experiments in the **MLflow UI** at http://localhost:5000.

### 8.6 Test the API

Using the Swagger UI:

1. Go to http://localhost:8000/docs.
2. Expand `POST /predict` → click **Try it out**.
3. Paste a sample JSON payload and click **Execute**.

Using `curl`:

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "Pregnancies": 6,
    "Glucose": 148,
    "BloodPressure": 72,
    "SkinThickness": 35,
    "Insulin": 125,
    "BMI": 33.6,
    "DiabetesPedigreeFunction": 0.627,
    "Age": 50
  }'
```

### 8.7 View Logs

```bash
# All services
docker compose logs -f

# Individual service
docker compose logs -f api
docker compose logs -f streamlit
docker compose logs -f mlflow
docker compose logs -f airflow-webserver
```

### 8.8 Stop the Stack

```bash
# Stop containers (data volumes preserved)
docker compose down

# Stop containers and remove volumes (⚠ destroys PostgreSQL & MLflow data)
docker compose down -v
```

---

## 9. Notebooks

The `notebooks/` directory contains Jupyter notebooks used during development:

| Notebook | Description |
|---|---|
| `01_data_profiling.ipynb` | Initial data profiling and quality assessment |
| `02_eda_guide_graphes.ipynb` | Exploratory data analysis with visualisations |
| `03_clustering.ipynb` | K-Means clustering experiments (Elbow, Silhouette) |
| `04_cluster_analysis.ipynb` | Interpretation and analysis of discovered clusters |

These notebooks are for exploration and are **not** part of the automated pipeline.

---

## 10. Limitations

- Predictions depend on the quality and representativeness of the training dataset.
- Cluster assignments from K-Means are statistical groupings, not clinically validated categories.
- The mapping between cluster IDs and risk levels is based on observed feature averages and may not generalise.
- Model performance should be validated on external data before any real-world use.
- The API does not implement authentication or rate limiting.
- Displayed results do not constitute medical advice.

---

## 11. Future Improvements

- Integrate additional and more diverse datasets.
- Add comprehensive unit and integration tests.
- Implement automated model validation gates before promoting to production.
- Add data drift detection and prediction monitoring.
- Secure the API with authentication and HTTPS.
- Set up a CI/CD pipeline for automated testing and deployment.
- Use the MLflow Model Registry for model versioning and stage management.

---

## 12. Author & Academic Context

| | |
|---|---|
| **Project** | Intelligent Diabetes Risk Prediction System |
| **Repository** | DiabetesRisk-AI |
| **Instructor** | Hamid OUFAKIR |
| **Brief date** | 23/09/2026 |
| **Implementation period** | 28/09/2026 – 09/10/2026 |
| **Programme** | RNCP – Développeur·se en Intelligence Artificielle |