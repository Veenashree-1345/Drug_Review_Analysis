import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv("drug_data.csv")

# Drop missing values
df.dropna(subset=['review', 'condition'], inplace=True)

# Remove conditions that appear less than 5 times
condition_counts = df['condition'].value_counts()
df = df[df['condition'].isin(condition_counts[condition_counts >= 5].index)]

# Encode condition labels
label_encoder = LabelEncoder()
df['condition_encoded'] = label_encoder.fit_transform(df['condition'])

# Convert text reviews into numerical features using TF-IDF
vectorizer = TfidfVectorizer(max_features=5000, stop_words='english', ngram_range=(1, 2))
X = vectorizer.fit_transform(df['review'])
y = df['condition_encoded']

# Split data using train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# Train Multinomial Naïve Bayes model
nb_model = MultinomialNB(alpha=0.1)  # Using alpha=0.1 for better smoothing
nb_model.fit(X_train, y_train)

# Save the model, vectorizer, and label encoder
joblib.dump(nb_model, "nb_condition_model.pkl")
joblib.dump(vectorizer, "con_vectorizer.pkl")
joblib.dump(label_encoder, "label_encoder.pkl")

# Make predictions
y_pred = nb_model.predict(X_test)

# Evaluate model
accuracy = accuracy_score(y_test, y_pred)
print(f"🔹 **Model Accuracy:** {accuracy:.4f}\n")

print("🔹 **Classification Report:**")
print(classification_report(y_test, y_pred, target_names=label_encoder.inverse_transform(np.unique(y_test))))

# Plot Confusion Matrix
plt.figure(figsize=(10, 6))
cm = confusion_matrix(y_test, y_pred)

