import streamlit as st
import joblib
import numpy as np

# Load models and vectorizers
vectorizer = joblib.load("vectorizer.pkl")  # Ensuring the correct vectorizer
con_vectorizer = joblib.load("con_vectorizer.pkl")  # Loading the missing con_vectorizer
sentiment_model = joblib.load("sentiment_model.pkl")
nb_condition_model = joblib.load("nb_condition_model.pkl")
label_encoder = joblib.load("label_encoder.pkl")
tokenizer = joblib.load("tokenizer.pkl")
lda_model = joblib.load("lda_model.pkl")
lda_vectorizer = joblib.load("lda_vectorizer.pkl")

# Streamlit UI
st.title("💊 Drug Review Analysis")
st.subheader("Choose a functionality")
option = st.selectbox("", ["Sentiment Analysis", "Condition Prediction", "Rating Prediction", "Topic Modeling"])

if option == "Sentiment Analysis":
    st.subheader("🔍 Predict Sentiment of a Drug Review")
    review = st.text_area("Enter your review:")
    if st.button("Predict Sentiment"):
        if review:
            transformed_review = vectorizer.transform([review])
            sentiment_prediction = sentiment_model.predict(transformed_review)[0]
            sentiment_map = {0: "Negative", 1: "Neutral", 2: "Positive"}
            st.success(f"Predicted Sentiment: {sentiment_map.get(sentiment_prediction, 'Unknown')}")

elif option == "Condition Prediction":
    st.subheader("🩺 Predict Condition from Review")
    review = st.text_area("Enter your review:")
    if st.button("Predict Condition"):
        if review:
            transformed_review = con_vectorizer.transform([review])  # Using con_vectorizer
            condition_pred = nb_condition_model.predict(transformed_review)[0]
            predicted_condition = label_encoder.inverse_transform([condition_pred])[0]
            st.success(f"Predicted Condition: {predicted_condition}")

elif option == "Rating Prediction":
    st.subheader("⭐ Estimate Drug Rating from Review")
    review = st.text_area("Enter your review:")
    if st.button("Estimate Rating"):
        if review:
            transformed_review = vectorizer.transform([review])
            sentiment_prediction = sentiment_model.predict(transformed_review)[0]
            sentiment_map = {0: "Negative", 1: "Neutral", 2: "Positive"}
            rating_estimate = {0: 3.0, 1: 6.0, 2: 9.0}  # Mapping sentiment to rating
            estimated_rating = rating_estimate.get(sentiment_prediction, 5.0)
            st.success(f"Estimated Rating: {estimated_rating} out of 10")

elif option == "Topic Modeling":
    st.subheader("📑 Identify Topics in Review")
    review = st.text_area("Enter your review:")
    if st.button("Identify Topics"):
        if review:
            transformed_review = lda_vectorizer.transform([review])
            topic_distribution = lda_model.transform(transformed_review)
            top_topic = np.argmax(topic_distribution)
            st.success(f"This review is mostly about Topic #{top_topic + 1}")
