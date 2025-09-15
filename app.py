import streamlit as st
import joblib
import numpy as np

# Load model
model = joblib.load('diabetes_model.pkl')

st.title("Diabetes Prediction")

# Input fields
pregnancies = st.number_input("Pregnancies", 0, 20, 1)
glucose = st.number_input("Glucose", 0, 300, 120)
blood_pressure = st.number_input("Blood Pressure", 0, 200, 80)
skin_thickness = st.number_input("Skin Thickness", 0, 100, 20)
insulin = st.number_input("Insulin", 0, 900, 80)
bmi = st.number_input("BMI", 0.0, 70.0, 25.0)
diabetes_pedigree = st.number_input("Diabetes Pedigree", 0.0, 3.0, 0.5)
age = st.number_input("Age", 1, 120, 30)

# Predict button
if st.button("Predict"):
    input_data = np.array([[pregnancies, glucose, blood_pressure, skin_thickness,
                          insulin, bmi, diabetes_pedigree, age]])
    
    prediction = model.predict(input_data)[0]
    
    if prediction == 1:
        st.error("High Risk of Diabetes")
    else:
        st.success("Low Risk of Diabetes")