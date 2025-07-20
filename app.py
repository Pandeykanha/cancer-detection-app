
import streamlit as st
import numpy as np
import pickle

# Load model
model = pickle.load(open("cancer_model.pkl", "rb"))

st.title("Cancer Prediction App")

# Input Fields
radius_mean = st.number_input("Radius Mean")
texture_mean = st.number_input("Texture Mean")
perimeter_mean = st.number_input("Perimeter Mean")
area_mean = st.number_input("Area Mean")
smoothness_mean = st.number_input("Smoothness Mean")

# Predict
if st.button("Predict"):
    input_data = np.array([[radius_mean, texture_mean, perimeter_mean, area_mean, smoothness_mean]])
    prediction = model.predict(input_data)
    result = "Malignant" if prediction[0] == 0 else "Benign"
    st.success(f"The tumor is likely: {result}")
