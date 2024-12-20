from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB  # Import Naive Bayes classifier
from sklearn.metrics import classification_report
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib  # For saving the model after training

from src.data_preprocessing import preprocess_data

# TF-IDF vectorization and data split
def vectorize_data(df):
    # Using TfidfVectorizer to transform the text data into TF-IDF features
    vectorizer = TfidfVectorizer(max_features=5000)  # Using top 5000 features (words)
    X = vectorizer.fit_transform(df['cleaned_text']).toarray()  # Convert sparse matrix to dense array
    y = df['sentiment'].values  # Extract labels (sentiment)

    # Return the separate variables
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test, vectorizer


# Train and evaluate model
def train_and_evaluate(X_train, X_test, y_train, y_test, model_type='naive_bayes'):
    # Select the model based on the model_type argument
    if model_type == 'naive_bayes':  # Naive Bayes classifier
        model = MultinomialNB()
    else:
        raise ValueError(f"Model type '{model_type}' not supported.")
    
    # Train the model using the training data
    model.fit(X_train, y_train)
    
    # Predict the labels on the test data
    y_pred = model.predict(X_test)
    
    # Return both classification report and the trained model
    return classification_report(y_test, y_pred), model


# Save trained model and vectorizer
def save_model(model, vectorizer, model_filename='naive_bayes_model.pkl', vectorizer_filename='vectorizer.pkl'):
    # Save the trained model and vectorizer to disk using joblib
    joblib.dump(model, model_filename)
    joblib.dump(vectorizer, vectorizer_filename)

# Load trained model and vectorizer
def load_model(model_filename='naive_bayes_model.pkl', vectorizer_filename='vectorizer.pkl'):
    # Load the trained model and vectorizer from disk
    model = joblib.load(model_filename)
    vectorizer = joblib.load(vectorizer_filename)
    return model, vectorizer

