# app.py

import os
import pickle
import pandas as pd
import numpy as np
import streamlit as st
import shap

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")

MODEL_PATH = os.path.join(MODEL_DIR, "trained_model.pkl")
SCALER_PATH = os.path.join(MODEL_DIR, "scaler.pkl")
FEATURE_NAMES_PATH = os.path.join(MODEL_DIR, "feature_names.csv")

# Load files
with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

with open(SCALER_PATH, "rb") as f:
    scaler = pickle.load(f)

feature_names = pd.read_csv(FEATURE_NAMES_PATH).squeeze().tolist()

class_mapping = {
    0: "No complication",
    1: "Moderate complication",
    2: "Severe complication"
}

# Streamlit page
st.set_page_config(
    page_title="Myocardial Complication Prediction",
    layout="centered"
)

st.title("Myocardial Complication Prediction")
st.write("This app predicts the risk category of post-myocardial infarction complications.")

st.sidebar.header("Patient Information")

# Input patient values
age = st.sidebar.number_input("Age", min_value=1, max_value=120, value=75)
sex = st.sidebar.selectbox("Sex", ["Female", "Male"])

# Create patient dictionary
new_patient = {col: 0 for col in feature_names}

if "AGE" in feature_names:
    new_patient["AGE"] = age

if "SEX" in feature_names:
    new_patient["SEX"] = 1 if sex == "Male" else 0
elif "SEX_1" in feature_names:
    new_patient["SEX_1"] = 1 if sex == "Male" else 0

new_patient_df = pd.DataFrame([new_patient])
new_patient_df = new_patient_df[feature_names]

# Prediction button
if st.button("Predict Complication Risk"):

    new_patient_scaled = scaler.transform(new_patient_df)

    prediction = model.predict(new_patient_scaled)[0]
    prediction_proba = model.predict_proba(new_patient_scaled)[0]

    predicted_label = class_mapping.get(prediction, "Unknown class")
    predicted_probability = max(prediction_proba)

    st.subheader("Prediction Result")

    st.write(f"### {predicted_label}")
    st.write(f"Probability: **{predicted_probability:.2%}**")

    st.subheader("Class Probabilities")

    probabilities_df = pd.DataFrame({
        "Class": [class_mapping.get(cls, str(cls)) for cls in model.classes_],
        "Probability": prediction_proba
    })

    st.dataframe(probabilities_df)

    # SHAP explanation
    st.subheader("Top SHAP Factors")

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(new_patient_scaled)

    predicted_class_index = list(model.classes_).index(prediction)

    if isinstance(shap_values, list):
        selected_shap_values = shap_values[predicted_class_index][0]
    else:
        selected_shap_values = shap_values[0, :, predicted_class_index]

    shap_df = pd.DataFrame({
        "Feature": feature_names,
        "SHAP Value": selected_shap_values
    })

    shap_df["Absolute SHAP Value"] = shap_df["SHAP Value"].abs()
    shap_df = shap_df.sort_values("Absolute SHAP Value", ascending=False).head(10)

    st.dataframe(shap_df[["Feature", "SHAP Value"]])

    st.bar_chart(
        shap_df.set_index("Feature")["Absolute SHAP Value"]
    )