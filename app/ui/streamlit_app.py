
import streamlit as st
import requests


st.set_page_config(
    page_title="Diabetes Risk Prediction",
    layout="centered"
)

st.title("🩺 Diabetes Risk Prediction")
st.write("Enter the patient's information to predict diabetes risk.")

API_URL = "http://api:8000/predict"

with st.form("diabetes_form"):

    st.subheader("Patient Information")

    pregnancies = st.number_input(
        "Pregnancies", min_value=0, value=0, step=1
    )

    glucose = st.number_input(
        "Glucose", min_value=0.0, value=0.0
    )

    blood_pressure = st.number_input(
        "Blood Pressure", min_value=0.0, value=0.0
    )

    skin_thickness = st.number_input(
        "Skin Thickness", min_value=0.0, value=0.0
    )

    insulin = st.number_input(
        "Insulin", min_value=0.0, value=10.0
    )

    bmi = st.number_input(
        "BMI", min_value=0.0, value=0.0
    )

    diabetes_pedigree = st.number_input(
        "Diabetes Pedigree Function",
        min_value=0.0,
        value=0.0
    )

    age = st.number_input(
        "Age", min_value=1, value=20, step=1
    )

    submitted = st.form_submit_button(
        "Predict Diabetes Risk"
    )

if submitted:

    payload = {
        "Pregnancies": pregnancies,
        "Glucose": glucose,
        "BloodPressure": blood_pressure,
        "SkinThickness": skin_thickness,
        "Insulin": insulin,
        "BMI": bmi,
        "DiabetesPedigreeFunction": diabetes_pedigree,
        "Age": age
    }

    try:
        with st.spinner("Predicting..."):

            response = requests.post(
                API_URL,
                json=payload,
                timeout=30
            )

        response.raise_for_status()

        result = response.json()

        st.subheader("Prediction Result")

        prediction = result["prediction"]
        high_risk = result["probability_high_risk"]
        low_risk = result["probability_low_risk"]

        if prediction == "high risk":
            st.error(f"Prediction: {prediction}")
        else:
            st.success(f"Prediction: {prediction}")

        st.metric(
            "Probability of High Risk",
            f"{float(high_risk) * 100:.2f}%"
        )

        st.metric(
            "Probability of Low Risk",
            f"{float(low_risk) * 100:.2f}%"
        )

        st.write("API Response:")
        st.json(result)

    except requests.exceptions.ConnectionError:
        st.error(
            "Cannot connect to the API. "
            "Make sure FastAPI is running."
        )

    except requests.exceptions.Timeout:
        st.error("The API request timed out.")

    except requests.exceptions.HTTPError as e:
        st.error(f"HTTP Error: {e}")
        st.code(response.text)

    except (KeyError, ValueError) as e:
        st.error(f"Invalid API response: {e}")

    except requests.exceptions.RequestException as e:
        st.error(f"Request failed: {e}")
