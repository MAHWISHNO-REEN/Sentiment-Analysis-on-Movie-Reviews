import streamlit as st
import pandas as pd
from src.data_preprocessing import preprocess_data
from src.model_training import load_model

# Load model and vectorizer
model, vectorizer = load_model('naive_bayes_model.pkl', 'vectorizer.pkl')

# Function to predict sentiment
def predict_sentiment(review):
    df = pd.DataFrame({'review': [review]})
    df = preprocess_data(df)
    vectorized_review = vectorizer.transform(df['cleaned_text'])
    prediction = model.predict(vectorized_review)
    return "Positive" if prediction[0] == 'positive' else "Negative"

# Streamlit UI
st.title("🎬 Sentiment Analysis for Movie Reviews")
st.write("Enter a review or upload a file to predict sentiment!")

# Input single review
review = st.text_area("Enter a Movie Review:")

if st.button("Predict Sentiment"):
    if review.strip():
        sentiment = predict_sentiment(review)
        st.success(f"The predicted sentiment is: **{sentiment}**")
    else:
        st.warning("Please enter a review to analyze.")

# Upload CSV file for batch processing
st.write("---")
uploaded_file = st.file_uploader("Upload a CSV File (containing 'review' column):", type=['csv'])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    if 'review' in df.columns:
        df = preprocess_data(df)
        vectorized_data = vectorizer.transform(df['cleaned_text'])
        predictions = model.predict(vectorized_data)
        df['Predicted Sentiment'] = ["Positive" if pred == 'positive' else "Negative" for pred in predictions]
        st.write("Predictions for Uploaded File:")
        st.dataframe(df[['review', 'Predicted Sentiment']])
    else:
        st.error("The uploaded file must contain a 'review' column.")


# Streamlit UI for showing classification report
st.write("---")
if st.button("Show Classification Report"):
    try:
        with open('classification_report.txt', 'r') as file:
            report = file.read()
            st.text("Classification Report:")
            st.code(report)  # Display as formatted code block
    except FileNotFoundError:
        st.error("Classification report not found. Please run the training phase.")
