from fastapi import FastAPI
from pathlib import Path
from pydantic import BaseModel
import joblib
import pandas as pd

app = FastAPI(
    title="Diabetes Risk API",
    description="API for diabetes risk prediction",
    version="1.0.0"
)

class UserData(BaseModel):
    Pregnancies: float
    Glucose: float
    BloodPressure: float
    SkinThickness: float
    Insulin: float
    BMI: float
    DiabetesPedigreeFunction: float
    Age: float


BASE_DIR = Path(__file__).resolve().parents[2]

model_path = BASE_DIR / "models" / "Logistic_Regression_model.pkl"
@app.get("/allo")
def dir():
    return {
        "base": str(BASE_DIR),
        "model_path": str(model_path)
    }

@app.get("/")
def root():
    return {
        "message": "Diabetes Risk API is running"
    }


@app.post("/predict")
def predict_diabetes_risk(data: UserData):
    try:
        model = joblib.load(str(model_path))
    except Exception as e:
        return {
            "error": str(e),
            "message": "Model was not loaded successfully",
            "path": str(model_path)
        }

    if model is None:
        return {
            "error": "Model not found",
            "message": "Please check the model path and ensure the model file exists."
        }
        
    input_data = pd.DataFrame([data.model_dump()])
    prediction = model.predict(input_data)

    return {
        "prediction": "high risk" if prediction[0] == 0 else "low risk"
    }



