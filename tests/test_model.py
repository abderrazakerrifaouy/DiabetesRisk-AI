import joblib
import pandas as pd



def test_model(new_user_data):
    path = "data/diabetes.csv"
    model = joblib.load("models/logistic_regression_model.pkl")

    return model.predict(new_user_data)

if __name__ == "__main__":
    new_user_data = pd.DataFrame({
        'Pregnancies': [6.0 , 6.0 , 1.0 , 0.0],
        'Glucose': [183.0 , 14.0 , 85.0 , 89.0],
        'BloodPressure': [64.0 , 72.0 , 66.0 , 66.0],
        'SkinThickness': [35.0 , 5.0 , 29.0 , 29.0],
        'Insulin': [125.33333333333331 , 15.33333333333331 , 66.66666666666667 , 33.0],
        'BMI': [23.3 , 3.6 , 26.6 , 28.1],
        'DiabetesPedigreeFunction': [0.5 , 0.627 , 0.31 , 0.351],
        'Age': [30 , 0 , 31.0 , 31.0]
    } 
    )

    

    for index, row in new_user_data.iterrows():
        prediction = test_model(row.to_frame().T)
        print(f"Prediction for user {index + 1}: {prediction[0]}")

