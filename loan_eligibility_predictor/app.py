import streamlit as st
import pandas as pd
import os
from src.model import load_model, predict

st.title("🏦 Loan Eligibility Predictor")
st.write("Enter the applicant's information:")

# User input form
gender = st.selectbox("Gender", ["Male", "Female"])
married = st.selectbox("Married", ["Yes", "No"])
dependents = st.selectbox("Dependents", ["0", "1", "2", "3+"])
education = st.selectbox("Education", ["Graduate", "Not Graduate"])
self_employed = st.selectbox("Self Employed", ["Yes", "No"])
applicant_income = st.number_input("Applicant Income", min_value=0)
coapplicant_income = st.number_input("Coapplicant Income", min_value=0)
loan_amount = st.number_input("Loan Amount (in thousands)", min_value=0)
loan_term = st.selectbox("Loan Term", [360, 180, 120, 84])
credit_history = st.selectbox("Credit History", [1.0, 0.0])
property_area = st.selectbox("Property Area", ["Urban", "Semiurban", "Rural"])

# Manual encoding (should match training)
input_dict = {
    "Gender": 1 if gender == "Male" else 0,
    "Married": 1 if married == "Yes" else 0,
    "Dependents": {"0": 0, "1": 1, "2": 2, "3+": 3}[dependents],
    "Education": 1 if education == "Graduate" else 0,
    "Self_Employed": 1 if self_employed == "Yes" else 0,
    "ApplicantIncome": applicant_income,
    "CoapplicantIncome": coapplicant_income,
    "LoanAmount": loan_amount,
    "Loan_Amount_Term": loan_term,
    "Credit_History": credit_history,
    "Property_Area": {"Urban": 2, "Semiurban": 1, "Rural": 0}[property_area]
}

input_df = pd.DataFrame([input_dict])

# Prediction
if st.button("Predict Loan Approval"):
    try:
        model = load_model()
        result = predict(model, input_df)[0]
        st.success("✅ Loan Approved" if result == 1 else "❌ Loan Not Approved")
    except Exception as e:
        st.error(f"Error during prediction: {e}")
