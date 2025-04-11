import pandas as pd
import streamlit as st
from sklearn.preprocessing import LabelEncoder

def load_data(filepath: str) -> pd.DataFrame:
    try:
        return pd.read_csv(filepath)
    except FileNotFoundError:
        st.error(f"❌ File not found: {filepath}")
        st.stop()

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    # Drop ID column if it exists (fixes the ValueError)
    if 'Loan_ID' in df.columns:
        df = df.drop(columns=['Loan_ID'])

    # Fill missing values
    df['Gender'] = df['Gender'].fillna('Male')
    df['Married'] = df['Married'].fillna(df['Married'].mode()[0])
    df['Dependents'] = df['Dependents'].fillna(df['Dependents'].mode()[0])
    df['Self_Employed'] = df['Self_Employed'].fillna('No')
    df['Credit_History'] = df['Credit_History'].fillna(1.0)
    df['Loan_Amount_Term'] = df['Loan_Amount_Term'].fillna(df['Loan_Amount_Term'].mode()[0])
    df['LoanAmount'] = df['LoanAmount'].fillna(df['LoanAmount'].median())

    return df

def encode_features(df: pd.DataFrame) -> pd.DataFrame:
    cols_to_encode = ['Gender', 'Married', 'Education', 'Self_Employed', 'Property_Area', 'Loan_Approved', 'Dependents']
    for col in cols_to_encode:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
    return df

def preprocess(filepath: str) -> pd.DataFrame:
    df = load_data(filepath)
    df = clean_data(df)
    df = encode_features(df)
    return df
