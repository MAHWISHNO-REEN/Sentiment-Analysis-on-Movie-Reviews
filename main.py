from src.data_preprocessing import load_data, preprocess_data
from src.model_training import vectorize_data, train_and_evaluate, save_model, load_model
from src.evaluation import evaluate_model
import pandas as pd
import os

def main():
    # Step 1: Load and preprocess the data
    print("Loading and preprocessing data...")
    df = load_data(r'E:\7th Semester\NLP\sentiment_analysis_project\data\IMDB Dataset.csv')  # Adjust to your data path
    if df.empty:
        print("Error: DataFrame is empty after loading.")
        return

    df = preprocess_data(df)

    print("Columns in the dataset:", df.columns)
    print("Cleaned text preview:")
    print(df['cleaned_text'].head())

    # Step 2: Vectorize the data and split it into training and test sets
    X_train, X_test, y_train, y_test, vectorizer = vectorize_data(df)

    # Check if model and vectorizer already exist
    if not os.path.exists('naive_bayes_model.pkl') or not os.path.exists('vectorizer.pkl'):
        print("Training and evaluating the model...")
        report, model = train_and_evaluate(X_train, X_test, y_train, y_test, model_type='naive_bayes')
        print("Classification Report:\n", report)

        # Save the trained model and vectorizer
        save_model(model, vectorizer)
        print("Model and vectorizer saved successfully!")
    else:
        print("Model and vectorizer already exist. Loading...")
        model, vectorizer = load_model('naive_bayes_model.pkl', 'vectorizer.pkl')
        
        # Generate classification report again
        print("Re-generating classification report...")
        report = evaluate_model(model, X_test, y_test)
        print("Classification Report:\n", report)

    try:
        with open('classification_report.txt', 'w') as f:
            f.write(report)
        print("Classification report saved successfully!")
    except Exception as e:
        print(f"Error while saving classification report: {e}")

if __name__ == "__main__":
    main()