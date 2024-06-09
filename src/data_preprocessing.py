import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

def load_data(filepath):
    data = pd.read_csv(filepath)
    return data

def preprocess_data(data):
    # Data cleaning and preprocessing steps
    # Example: lowercasing, removing stopwords, etc.
    return processed_data

def split_data(data, target, test_size=0.2):
    X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=test_size, random_state=42)
    return X_train, X_test, y_train, y_test

def vectorize_data(data, method='tfidf'):
    if method == 'tfidf':
        vectorizer = TfidfVectorizer()
    else:
        vectorizer = CountVectorizer()
    return vectorizer.fit_transform(data)


