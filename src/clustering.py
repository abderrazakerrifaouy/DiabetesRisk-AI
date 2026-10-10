from pathlib import Path

import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import mlflow



PROJECT_DIR = Path("/opt/airflow/project")

PROCESSED_DATA = PROJECT_DIR / "data" / "processed" / "cleaned_data_diabetes.csv"
CLUSTRED_DATA = PROJECT_DIR / "data" / "processed" / "clustered_data_diabetes.csv"




class Clustering:
    def __init__(self, path):
        self.path = path
        self.data = None
        self.standard = StandardScaler()
        self.dataStandardized = None

    def _loadData(self):
        try:
            self.data = pd.read_csv(self.path)
            mlflow.log_text("Data loaded successfully." , "data_loading_log.txt")
        except Exception as e:
            mlflow.log_text(f"Error loading data: {e}", "data_loading_log.txt")

    def _validateData(self):
        if self.data is None:
            mlflow.log_text("Data not loaded. Please load the data first." , "data_validation_log.txt")
            return False
        return True

    def _preprocessData(self):
        if not self._validateData():
            return
        try:
            self.dataStandardized = self.standard.fit_transform(self.data)
            mlflow.log_text("Data preprocessing completed successfully.", "data_preprocessing_log.txt")
        except Exception as e:
            mlflow.log_text(f"Error during data preprocessing: {e}", "data_preprocessing_log.txt")

    def _elbow_method(self, max_k=10):
        if not self._validateData():
            return

        distortions = []
        for k in range(1, max_k + 1):
            kmeans = KMeans(n_clusters=k, random_state=42)
            kmeans.fit(self.dataStandardized)
            distortions.append(kmeans.inertia_)

        plt.figure(figsize=(8, 4))
        plt.plot(range(1, max_k + 1), distortions, marker='o')
        plt.title('Elbow Method For Optimal k')
        plt.xlabel('Number of clusters (k)')
        plt.ylabel('Distortion')
        plt.show()

    def _silhouette_score(self, max_k=10):
        if not self._validateData():
            return

        from sklearn.metrics import silhouette_score

        silhouette_scores = []
        for k in range(2, max_k + 1):
            kmeans = KMeans(n_clusters=k, random_state=42)
            kmeans.fit(self.dataStandardized)
            score = silhouette_score(self.dataStandardized, kmeans.labels_)
            silhouette_scores.append(score)

        plt.figure(figsize=(8, 4))
        plt.plot(range(2, max_k + 1), silhouette_scores, marker='o')
        plt.title('Silhouette Score For Optimal k')
        plt.xlabel('Number of clusters (k)')
        plt.ylabel('Silhouette Score')
        plt.show()

    def _fitKMeans(self, n_clusters):
        if not self._validateData():
            return

        try:
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            kmeans.fit(self.dataStandardized)
            mlflow.log_text(f"KMeans clustering completed with {n_clusters} clusters.", "kmeans_log.txt")
            return kmeans.labels_
        except Exception as e:
            mlflow.log_text(f"Error during KMeans fitting: {e}", "kmeans_log.txt")
            return None

    def _add_labels_to_data(self, labels):
        if not self._validateData():
            return

        try:
            self.data['Cluster'] = labels
            mlflow.log_text("Cluster labels added to the data.", "data_adding_log.txt")
        except Exception as e:
            mlflow.log_text(f"Error adding labels to data: {e}", "data_adding_log.txt")

    def _save_clustered_data(self, output_path):
        if not self._validateData():
            return

        try:
            self.data.to_csv(output_path, index=False)
            mlflow.log_text(f"Clustered data saved to {output_path}.", "data_saving_log.txt")
        except Exception as e:
            mlflow.log_text(f"Error saving clustered data: {e}", "data_saving_log.txt")

    def run_clustering_pipeline(self, n_clusters=2):
        mlflow.set_tracking_uri("http://mlflow:5000")
        mlflow.set_experiment("DiabetsRisk-AI-Clustering")

        with mlflow.start_run(run_name="KMeans-Clustering"):
            self._loadData()
            self._preprocessData()
            labels = self._fitKMeans(n_clusters)
            if labels is not None:
                self._add_labels_to_data(labels)
                self._save_clustered_data(CLUSTRED_DATA)
                mlflow.log_param("n_clusters", n_clusters)
                mlflow.log_param('algorithm', 'KMeans')
                mlflow.log_param("random_state", 42)
                mlflow.log_artifact(CLUSTRED_DATA, artifact_path="clustered_data")
            mlflow.end_run()

if __name__ == "__main__":

    clustering = Clustering(PROCESSED_DATA)
    clustering.run_clustering_pipeline(n_clusters=2)