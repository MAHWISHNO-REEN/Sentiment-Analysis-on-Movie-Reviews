import os
import re
import nltk
import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# Ensure NLTK resources are downloaded
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

nltk.download('punkt_tab')

# Try clearing and setting paths to avoid issues
nltk.data.path.append(os.path.join(os.getenv('APPDATA'), 'nltk_data'))

# Function to load data
def load_data(file_path):
    try:
        # Load the dataset
        df = pd.read_csv(file_path)
        print(f"Successfully loaded data from {file_path}.")
        print(f"Total data points in the dataset: {len(df)}")
        return df
    except Exception as e:
        print(f"Error loading data: {e}")
        return pd.DataFrame()  # Return an empty DataFrame if loading fails

# Function to clean text
def clean_text(text):
    text = text.lower()  # Convert text to lowercase
    text = re.sub(r'http\S+|www\S+', '', text)  # Remove URLs
    text = re.sub(r'[^a-zA-Z\s]', '', text)  # Remove non-alphabetic characters
    text = re.sub(r'\s+', ' ', text)  # Replace multiple spaces with a single space
    return text

# Function to remove stopwords
def remove_stopwords(tokens):
    stop_words = set(stopwords.words('english'))
    return [word for word in tokens if word not in stop_words]

# Function to apply lemmatization
def lemmatize_tokens(tokens):
    lemmatizer = WordNetLemmatizer()
    return [lemmatizer.lemmatize(word) for word in tokens]

# # Tokenize and preprocess the data
def preprocess_data(df):
    # Apply the cleaning, tokenization, stopword removal, and lemmatization
    df['cleaned_text'] = df['review'].apply(clean_text)

    # Tokenization using word_tokenize
    df['tokens'] = df['cleaned_text'].apply(word_tokenize)

    # Removing stopwords and applying lemmatization
    df['filtered_tokens'] = df['tokens'].apply(remove_stopwords)
    df['lemmatized_tokens'] = df['filtered_tokens'].apply(lemmatize_tokens)

    # Remove rows where cleaned text is empty after preprocessing
    df = df[df['cleaned_text'].str.strip() != '']
    
    return df


