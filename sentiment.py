import pandas as pd
import re
from datetime import datetime
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score,classification_report, mean_squared_error
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
import matplotlib.pyplot as plt
import seaborn as sns
df=pd.read_csv('drug_reviews.csv')
df

# Drop rows where review or rating is missing
df.dropna(subset=['review', 'rating'], inplace=True)

# Fill missing conditions with "Unknown"
df['condition'].fillna("Unknown", inplace=True)
# Define a function to convert ratings
def convert_rating(rating):
    return np.ceil(rating / 2).astype(int)

# Apply the conversion to the rating column
df['rating_5'] = df['rating'].apply(convert_rating)

# Select only the columns of interest
filtered_data = df[['review', 'rating', 'rating_5']]

print(filtered_data)

import re
from datetime import datetime
from sklearn.preprocessing import LabelEncoder

import html

df['review'] = df['review'].astype(str).apply(lambda text: html.unescape(text))
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')

# Load default stopwords from NLTK
stop_words = set(stopwords.words("english"))

# Define negation words that should NOT be removed
negation_words = {"no", "not", "never", "none", "nobody", "nowhere", "nothing", "neither"}

# Create a new stopword list that excludes negations
filtered_stopwords = stop_words - negation_words

def remove_stopwords_nltk(text):
    if not isinstance(text, str):
        return ""

    words = text.split()  # Simple word split
    filtered_words = [word for word in words if word.lower() not in filtered_stopwords]  # Keep negation words

    return " ".join(filtered_words)

df['clean_review'] = df['review'].astype(str).apply(remove_stopwords_nltk)
print(df)
vectorizer = TfidfVectorizer(max_features=30000)
X_text = vectorizer.fit_transform(df['clean_review'])
df['tokenized_review'] = df['review'].apply(lambda text: text.split())
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet

# Download necessary NLTK resources
nltk.download('wordnet')
nltk.download('omw-1.4')

# Initialize the lemmatizer
lemmatizer = WordNetLemmatizer()

# Function to lemmatize text
def lemmatize_text(text):
    if not isinstance(text, str):
        return ""

    words = text.split()  # Split into words
    lemmatized_words = [lemmatizer.lemmatize(word) for word in words]  # Lemmatize each word
    return " ".join(lemmatized_words)  # Join words back into a sentence

# Apply lemmatization to the 'clean_review' column
df['review'] = df['review'].apply(lemmatize_text)
# Display cleaned & lemmatized reviews
#print(df[['review', 'clean_review']].head())
from nltk.stem import PorterStemmer

# Download necessary NLTK resources
nltk.download('punkt')

# Initialize the stemmer
stemmer = PorterStemmer()

# Function to perform stemming
def stem_text(text):
    if not isinstance(text, str):
        return ""

    words = text.split()  # Split text into words
    return " ".join(stemmer.stem(word) for word in words)  # Stem each word

# Apply stemming to the 'review' column
df['cleaned_review'] = df['review'].apply(stem_text)

# Display results
print(df[['review']].head())
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.datasets import make_classification

# Convert ratings to sentiment labels
def get_sentiment(rating_5):
    if rating_5 >= 4:
        return 2  # Positive sentiment
    elif rating_5 == 3:
        return 1  # Neutral sentiment
    else:
        return 0  # Negative sentiment

df['sentiment'] = df['rating'].apply(get_sentiment)

# Sample dataset for faster training
df_sampled = df.sample(n=2000, random_state=42)

# TF-IDF Vectorization with 30,000 features
vectorizer = TfidfVectorizer(max_features=3000, stop_words='english', ngram_range=(1,2))
X_sampled = vectorizer.fit_transform(df_sampled['cleaned_review'])
y_sampled = df_sampled['sentiment']

# Handle class imbalance using SMOTE
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_sampled, y_sampled)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

# Train Logistic Regression Model
logistic_model = LogisticRegression(max_iter=1000, C=2, solver='liblinear')
logistic_model.fit(X_train, y_train)

# Save model and vectorizer
joblib.dump(logistic_model, 'sentiment_model.pkl')
joblib.dump(vectorizer, 'vectorizer.pkl')

# Predictions
y_pred = logistic_model.predict(X_test)

# Evaluate Model
accuracy = accuracy_score(y_test, y_pred)
print(f'Model Accuracy: {accuracy:.4f}')

# Confusion Matrix
def plot_confusion_matrix(cm, classes):
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    plt.show()

cm = confusion_matrix(y_test, y_pred)
plot_confusion_matrix(cm, classes=['Negative', 'Neutral', 'Positive'])

# Classification Report
print("Classification Report:")
print(classification_report(y_test, y_pred, target_names=['Negative', 'Neutral', 'Positive']))
