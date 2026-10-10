from pathlib import Path
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix , f1_score , recall_score , precision_score
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
import joblib
import mlflow

PROJECT_DIR = Path("/opt/airflow/project")

CLUSTRED_DATA = PROJECT_DIR / "data" / "processed" / "clustered_data_diabetes.csv"
MODEL_DIR = PROJECT_DIR / "models"


class Classification:
    def __init__(self , path ):
        self.path = path
        self.data = None
        self.X = None
        self.y = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None


    def loadData(self):
        try:
            self.data = pd.read_csv(self.path)
        except Exception as e:
            print(f"Error loading data: {e}")


    def splitData(self, target_column):
        if self.data is None:
            print("Data not loaded. Please load the data first.")
            return
        try:
            self.X = self.data.drop(columns=[target_column])
            self.y = self.data[target_column]
        except Exception as e:
            print(f"Error splitting data: {e}")



    def splitTrainTest(self):
        if self.X is None:
            print("Data not split. Please split the data first.")
            return
        try:
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=0.2, random_state=42 , stratify=self.y)

        except Exception as e:
            print(f"Error during data scaling: {e}")


    def grid_search(self, model, hyperparameters):
        try:
            pipeline = Pipeline([
                ("scaler", StandardScaler()),
                ("model", model)
            ])

            grid_search = GridSearchCV(
                pipeline,
                hyperparameters,
                cv=5
            )

            grid_search.fit(self.X_train, self.y_train)

            return grid_search.best_estimator_

        except Exception as e:
            print(f"Error during grid search: {e}")

            
    def train_logistic(self):
        if self.X is None or self.y is None:
            print("Data not prepared. Please ensure data is loaded, split, and scaled.")
            return
        try:
            model = LogisticRegression()
            hyperparameters = {
                'model__C': [0.1, 1.0, 10.0],
                'model__max_iter': [100, 1000, 10000],
                'model__solver': ['lbfgs'],
                'model__random_state': [42]
            }

            return self.grid_search(model, hyperparameters)

        except Exception as e:
            print(f"Error during model training: {e}")
    
    
    def train_SVM(self):
        if self.X is None or self.y is None:
            print("Data not prepared. Please ensure data is loaded, split, and scaled.")
            return
        try:
            model = SVC()
            hyperparameters = {
                'model__C': [0.1, 1.0, 10.0],
                'model__kernel': [ 'rbf'],
                'model__gamma': ['scale', 'auto'],
                'model__random_state': [42]
            }
            return self.grid_search(model, hyperparameters)

        except Exception as e:
            print(f"Error during model training: {e}")


    def train_random_forest(self):

        if self.X is None or self.y is None:
            print("Data not prepared. Please ensure data is loaded, split, and scaled.")
            return
        try:
            model = RandomForestClassifier()
            hyperparameters = {
                'model__n_estimators': [100, 200, 500],
                'model__max_depth': [None, 10, 20],
                'model__min_samples_split': [2, 5, 10],
                'model__random_state': [42]
            }
            return self.grid_search(model, hyperparameters)

        except Exception as e:
            print(f"Error during model training: {e}")


    def train_knn(self):
        if self.X is None or self.y is None:
            print("Data not prepared. Please ensure data is loaded, split, and scaled.")
            return
        try:
            model = KNeighborsClassifier()
            hyperparameters = {
                'model__n_neighbors': [3, 5, 7],
                'model__weights': ['uniform', 'distance'],
                'model__algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute']
            }
            return self.grid_search(model, hyperparameters)

        except Exception as e:
            print(f"Error during model training: {e}")

    
    def compare_models(self, models):
        if self.X is None or self.y is None:
            print("Data not prepared. Please ensure data is loaded, split, and scaled.")
            return
        try:
            results = []
            for model_name, model in models.items():

                with mlflow.start_run(run_name=model_name):

                    mlflow.log_param("model", model_name)
                    
                    predictions = model.predict(self.X_test)

                    accuracy = accuracy_score(self.y_test, predictions)
                    mlflow.log_metric("accuracy", accuracy)

                    precision = precision_score(self.y_test, predictions, average='weighted', zero_division=0)
                    mlflow.log_metric("precision", precision)

                    recall = recall_score(self.y_test, predictions, average='weighted', zero_division=0)
                    mlflow.log_metric("recall", recall)

                    f1 = f1_score(self.y_test, predictions, average='weighted', zero_division=0)
                    mlflow.log_metric("f1", f1)

                    cm = confusion_matrix(self.y_test, predictions)
                    mlflow.log_text(f"Confusion Matrix:\n{cm}", "confusion_matrix.txt")

                    results.append({
                        'Model': model_name,
                        'Accuracy': accuracy,
                        'Precision': precision,
                        'Recall': recall,
                        'F1 Score': f1,
                        'Confusion Matrix': cm.tolist()
                    })
                    self.save_model(model , model_name)
                    mlflow.log_text(f"Model: {model_name}\nAccuracy: {accuracy:.4f}\nPrecision: {precision:.4f}\nRecall: {recall:.4f}\nF1 Score: {f1:.4f}\nConfusion Matrix:\n{cm}", "model_metrics.txt")
            return results
        except Exception as e:
            print(f"Error during model comparison: {e}")



    def save_model(self, model, model_name):
        try:
            joblib.dump(model, MODEL_DIR / f"{model_name}_model.pkl")
            mlflow.log_artifact(MODEL_DIR / f"{model_name}_model.pkl", artifact_path="models")

        except Exception as e:
            print(f"Error saving model: {e}")

    def run_pipeline(self, target_column):
        mlflow.set_tracking_uri("http://mlflow:5000")  
        mlflow.set_experiment("DiabetsRisk-AI-Classification")

        self.loadData()
        self.splitData(target_column)
        self.splitTrainTest()
        models = {}
        models['Logistic_Regression'] = self.train_logistic()
        models['Random Forest'] = self.train_random_forest()
        models['SVM'] = self.train_SVM()
        models['KNN'] = self.train_knn()
        results = self.compare_models(models)
        return results

        
    
# if __name__ == "__main__":

#     path = CLUSTRED_DATA
#     target_column = "Cluster"  
#     classifier = Classification(path)
#     results = classifier.run_pipeline(target_column)
#     for result in results:
#         print("-" * 30)
#         print(f"Model: {result['Model']}")
#         print(f"Accuracy: {result['Accuracy']:.4f}")
#         print(f"Precision: {result['Precision']:.4f}")
#         print(f"Recall: {result['Recall']:.4f}")
#         print(f"F1 Score: {result['F1 Score']:.4f}")
#         print("Confusion Matrix:")
#         print(result['Confusion Matrix'])
#         print("-" * 30)





