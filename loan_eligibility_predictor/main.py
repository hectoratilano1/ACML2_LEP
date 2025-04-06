
from src import preprocessing, model, evaluate

def run_pipeline():
    # Load and preprocess data
    df = preprocessing.preprocess('data/credit.csv')

    # Split data
    X_train, X_test, y_train, y_test = model.split_data(df)

    # Train model
    clf = model.train_model(X_train, y_train)

    # Save model
    model.save_model(clf)

    # Evaluate model
    acc, report, matrix = evaluate.evaluate_model(clf, X_test, y_test)
    
    print(f"Model Accuracy: {acc:.2f}")
    print("\nClassification Report:\n", report)
    print("Confusion Matrix:\n", matrix)

if __name__ == "__main__":
    run_pipeline()
