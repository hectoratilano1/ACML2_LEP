
import pandas as pd
from sklearn.preprocessing import LabelEncoder

def load_data(filepath: str) -> pd.DataFrame:
    return pd.read_csv(filepath)

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    # Fill missing values for categorical variables
    df['Gender'].fillna('Male', inplace=True)
    df['Married'].fillna(df['Married'].mode()[0], inplace=True)
    df['Dependents'].fillna(df['Dependents'].mode()[0], inplace=True)
    df['Self_Employed'].fillna('No', inplace=True)
    df['Credit_History'].fillna(1.0, inplace=True)
    df['Loan_Amount_Term'].fillna(df['Loan_Amount_Term'].mode()[0], inplace=True)
    
    # Fill missing LoanAmount with median
    df['LoanAmount'].fillna(df['LoanAmount'].median(), inplace=True)

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
