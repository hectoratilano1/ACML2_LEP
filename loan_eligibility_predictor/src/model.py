
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import joblib

def split_data(df: pd.DataFrame):
    X = df.drop(columns=['Loan_Approved'])
    y = df['Loan_Approved']
    return train_test_split(X, y, test_size=0.2, random_state=42)

def train_model(X_train, y_train):
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    return model

def save_model(model, path='model.joblib'):
    joblib.dump(model, path)

def load_model(path='model.joblib'):
    return joblib.load(path)

def predict(model, input_data: pd.DataFrame):
    return model.predict(input_data)
