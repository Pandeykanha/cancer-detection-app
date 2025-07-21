
import streamlit as st
import numpy as np
import joblib

# Load the trained model
model = joblib.load(open("cancer_model.pkl", "rb"))

# App title
st.title("Cancer Prediction App")

# Input fields for 5 features
radius_mean = st.number_input("Radius Mean", format="%.4f")
texture_mean = st.number_input("Texture Mean", format="%.4f")
perimeter_mean = st.number_input("Perimeter Mean", format="%.4f")
area_mean = st.number_input("Area Mean", format="%.4f")
smoothness_mean = st.number_input("Smoothness Mean", format="%.4f")

# Predict button
if st.button("Predict"):
    # Prepare input data
    input_data = np.array([[radius_mean, texture_mean, perimeter_mean, area_mean, smoothness_mean]])

    # Ensure input is the right shape
    if input_data.shape != (1, 5):
        st.error("Invalid input shape. Please enter all 5 values.")
    else:
        # Make prediction
        prediction = model.predict(input_data)

        # Interpret result
        result = "Malignant" if prediction[0] == 0 else "Benign"

        # Show result
        st.success(f"The tumor is likely: {result}")