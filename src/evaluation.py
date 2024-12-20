# src/evaluation.py
from sklearn.metrics import classification_report

def evaluate_model(model, X_test, y_test):
    """
    Evaluate the model and return the classification report.
    """
    y_pred = model.predict(X_test)  # Generate predictions
    report = classification_report(y_test, y_pred)
    return report
