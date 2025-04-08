import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import joblib
import logging

# Optional: logging config in case this file is run directly (safe fallback)
logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')

def split_data(df: pd.DataFrame):
    logging.info("Splitting data into features and target...")
    X = df.drop(columns=['Loan_Approved'])
    y = df['Loan_Approved']
    return train_test_split(X, y, test_size=0.2, random_state=42)

def train_model(X_train, y_train):
    logging.info("Training Logistic Regression model...")
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    logging.info("Model training completed.")
    return model

def save_model(model, path='model.joblib'):
    try:
        joblib.dump(model, path)
        logging.info(f"Model saved to {path}")
    except Exception as e:
        logging.error(f"Failed to save model: {e}", exc_info=True)

def load_model(path='model.joblib'):
    try:
        logging.info(f"Loading model from {path}...")
        return joblib.load(path)
    except FileNotFoundError:
        logging.error(f"Model file '{path}' not found.")
        raise
    except Exception as e:
        logging.error(f"Error loading model: {e}", exc_info=True)
        raise

def predict(model, input_data: pd.DataFrame):
    logging.info("Making predictions...")
    return model.predict(input_data)
