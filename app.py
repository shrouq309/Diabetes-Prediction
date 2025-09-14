import streamlit as st
import joblib
import numpy as np
import pandas as pd

# Load the trained model
@st.cache_resource
def load_model():
    return joblib.load('diabetes_model.pkl')

def main():
    st.title("🩺 Diabetes Prediction App")
    st.write("Enter the following information to predict diabetes risk")
    
    # Load model
    model = load_model()
    
    # Create input fields for all features
    col1, col2 = st.columns(2)
    
    with col1:
        pregnancies = st.number_input("Pregnancies", min_value=0, max_value=20, value=0)
        glucose = st.number_input("Glucose Level", min_value=0, max_value=300, value=120)
        blood_pressure = st.number_input("Blood Pressure", min_value=0, max_value=200, value=80)
        skin_thickness = st.number_input("Skin Thickness", min_value=0, max_value=100, value=20)
    
    with col2:
        insulin = st.number_input("Insulin", min_value=0, max_value=900, value=80)
        bmi = st.number_input("BMI", min_value=0.0, max_value=70.0, value=25.0)
        diabetes_pedigree = st.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=3.0, value=0.5)
        age = st.number_input("Age", min_value=1, max_value=120, value=30)
    
    # Make prediction
    if st.button("Predict Diabetes Risk"):
        # Prepare input data
        input_data = np.array([[pregnancies, glucose, blood_pressure, skin_thickness, 
                              insulin, bmi, diabetes_pedigree, age]])
        
        # Make prediction
        prediction = model.predict(input_data)
        prediction_proba = model.predict_proba(input_data)
        
        # Display results
        st.subheader("Prediction Results")
        
        if prediction[0] == 1:
            st.error("⚠️ High Risk of Diabetes")
        else:
            st.success("✅ Low Risk of Diabetes")
        
        st.write(f"Confidence: {prediction_proba[0].max():.2%}")
        
        # Show input summary
        st.subheader("Input Summary")
        input_df = pd.DataFrame({
            'Feature': ['Pregnancies', 'Glucose', 'Blood Pressure', 'Skin Thickness', 
                       'Insulin', 'BMI', 'Diabetes Pedigree', 'Age'],
            'Value': [pregnancies, glucose, blood_pressure, skin_thickness, 
                     insulin, bmi, diabetes_pedigree, age]
        })
        st.table(input_df)
    
    # Add some information about the features
    with st.expander("ℹ️ Feature Information"):
        st.write("""
        **Pregnancies**: Number of times pregnant
        
        **Glucose**: Plasma glucose concentration over 2 hours in an oral glucose tolerance test
        
        **Blood Pressure**: Diastolic blood pressure (mm Hg)
        
        **Skin Thickness**: Triceps skin fold thickness (mm)
        
        **Insulin**: 2-Hour serum insulin (mu U/ml)
        
        **BMI**: Body mass index (weight in kg/(height in m)^2)
        
        **Diabetes Pedigree Function**: A function that scores likelihood of diabetes based on family history
        
        **Age**: Age in years
        """)

if __name__ == "__main__":
    main()