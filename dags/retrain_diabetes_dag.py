from airflow import DAG, task
from airflow.operators.python import PythonOperator
from datetime import datetime
from src.data_cleaner import DataCleaner
from src.clustering import Clustering
from src.classification.classification import Classification
from pathlib import Path

PROJECT_DIR = Path("/opt/airflow/project")

RAW_DATA = PROJECT_DIR / "data" / "raw" / "dataset-diabete.csv"
PROCESSED_DATA = PROJECT_DIR / "data" / "processed" / "cleaned_data_diabetes.csv"
CLUSTRED_DATA = PROJECT_DIR / "data" / "processed" / "clustered_data_diabetes.csv"

dag = DAG(
    dag_id = "retrain_diabetes_dag",
    start_date = datetime(2026, 10, 9),
    schedule_interval = "@monthly" ,
    description = "DAG for retraining the diabetes risk model",
    catchup = False
)

task_clean_data = PythonOperator(
    task_id = "clean_data",
    python_callable = lambda: DataCleaner(RAW_DATA).run_cleaning_pipeline(PROCESSED_DATA),
    dag = dag
)

task_cluster_data = PythonOperator(
    task_id = "cluster_data",
    python_callable = lambda: Clustering(PROCESSED_DATA).run_clustering_pipeline(n_clusters=2),
    dag = dag
)

task_train_model = PythonOperator(
    task_id = "train_model",
    python_callable = lambda: Classification(CLUSTRED_DATA).run_pipeline(target_column="Cluster"),
    dag = dag
)

task_clean_data >> task_cluster_data >> task_train_model