import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from wordcloud import WordCloud
from collections import Counter
import joblib

# Load dataset
df = pd.read_csv("drug_reviews.csv")

# Drop missing values
df.dropna(subset=['review', 'condition'], inplace=True)

# Extracting most helpful reviews
## Consider reviews with most words as "helpful" proxy
df['review_length'] = df['review'].apply(lambda x: len(str(x).split()))
helpful_reviews = df.nlargest(10, 'review_length')
print("Top 10 most detailed reviews:")
print(helpful_reviews[['drugName', 'condition', 'review']])

# Understanding patients with negative reviews
negative_reviews = df[df['rating'] <= 3]
print(f"Total negative reviews: {len(negative_reviews)}")
print("Common conditions with negative reviews:")
print(negative_reviews['condition'].value_counts().head(10))

# Visualizing common words in negative reviews
negative_text = " ".join(negative_reviews['review'].astype(str))
wordcloud = WordCloud(width=800, height=400, background_color='white').generate(negative_text)
plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.title("Most Common Words in Negative Reviews")
plt.show()

# Topic Modeling using LDA
vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
X = vectorizer.fit_transform(df['review'])
lda = LatentDirichletAllocation(n_components=5, random_state=42)
lda.fit(X)

# Displaying Top Words in Each Topic
feature_names = vectorizer.get_feature_names_out()
def print_topics(model, feature_names, n_top_words):
    for topic_idx, topic in enumerate(model.components_):
        print(f"Topic {topic_idx+1}:")
        print(" ".join([feature_names[i] for i in topic.argsort()[:-n_top_words - 1:-1]]))
        print()
print_topics(lda, feature_names, 10)

# Save models for deployment
joblib.dump(vectorizer, 'lda_vectorizer.pkl')
joblib.dump(lda, 'lda_model.pkl')