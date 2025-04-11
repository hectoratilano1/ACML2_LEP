import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import joblib
import logging
import os

# Optional: logging config
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

def save_model(model, filename='model.joblib'):
    try:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.abspath(os.path.join(current_dir, "..", filename))  # ensure normalized path
        joblib.dump(model, model_path)
        logging.info(f"✅ Model saved to {model_path}")
    except Exception as e:
        logging.error(f"❌ Failed to save model: {e}", exc_info=True)

def load_model(filename='model.joblib'):
    try:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.abspath(os.path.join(current_dir, "..", filename))  # ensure normalized path
        logging.info(f"📦 Loading model from {model_path}...")
        return joblib.load(model_path)
    except FileNotFoundError:
        logging.error(f"❌ Model file '{filename}' not found at {model_path}.")
        raise
    except Exception as e:
        logging.error(f"❌ Error loading model: {e}", exc_info=True)
        raise

def predict(model, input_data: pd.DataFrame):
    logging.info("🔮 Making predictions...")
    return model.predict(input_data)
