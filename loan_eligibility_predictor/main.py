import logging
from src import preprocessing, model, evaluate

# Set up logging config (you can change the filename or format)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    filename="pipeline.log",  # You can remove this line to print to terminal instead
    filemode="w"  # Overwrites the log each time; change to 'a' to append
)

def run_pipeline():
    try:
        logging.info("Starting ML pipeline...")

        # Load and preprocess data
        logging.info("Loading and preprocessing data...")
        df = preprocessing.preprocess("data/credit.csv")

        # Split data
        logging.info("Splitting into train/test sets...")
        X_train, X_test, y_train, y_test = model.split_data(df)

        # Train model
        logging.info("Training model...")
        clf = model.train_model(X_train, y_train)

        # Save model
        logging.info("Saving model to model.joblib...")
        model.save_model(clf)

        # Evaluate model
        logging.info("Evaluating model performance...")
        acc, report, matrix = evaluate.evaluate_model(clf, X_test, y_test)

        print(f"Model Accuracy: {acc:.2f}")
        print("\nClassification Report:\n", report)
        print("Confusion Matrix:\n", matrix)

        logging.info(f"Model accuracy: {acc:.2f}")
        logging.info("Pipeline finished successfully.")

    except Exception as e:
        logging.error(f"An error occurred in the pipeline: {e}", exc_info=True)
        print("Oops! Something went wrong. Check pipeline.log for details.")

if __name__ == "__main__":
    run_pipeline()
