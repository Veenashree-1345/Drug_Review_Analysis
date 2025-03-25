import streamlit as st
import joblib
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load models and vectorizers
vectorizer = joblib.load("vectorizer.pkl")  # Ensuring the correct vectorizer
con_vectorizer = joblib.load("con_vectorizer.pkl")  # Loading the missing con_vectorizer
sentiment_model = joblib.load("sentiment_model.pkl")
nb_condition_model = joblib.load("nb_condition_model.pkl")
label_encoder = joblib.load("label_encoder.pkl")
tokenizer = joblib.load("tokenizer.pkl")
helpfulness_model = joblib.load("helpfulness_model.pkl")  # New model for helpfulness analysis

# Streamlit UI
st.title("💊 Drug Review Analysis")
st.subheader("Choose a functionality")
option = st.selectbox("", ["Sentiment Analysis", "Condition Prediction", "Rating Prediction", "Identify Helpful Reviews"])

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
    st.subheader("🧐 Predict Condition from Review")
    review = st.text_area("Enter your review:")
    if st.button("Predict Condition"):
        if review:
            transformed_review = con_vectorizer.transform([review])  # Using con_vectorizer
            condition_pred = nb_condition_model.predict(transformed_review)[0]
            predicted_condition = label_encoder.inverse_transform([condition_pred])[0]
            st.success(f"Predicted Condition: {predicted_condition}")

elif option == "Rating Prediction":
    st.subheader("⭐ Predict Drug Rating from Review")
    review = st.text_area("Enter your review:")
    if st.button("Predict Rating"):
        if review:
            review_seq = tokenizer.texts_to_sequences([review])
            review_padded = pad_sequences(review_seq, maxlen=100, padding='post')
            st.warning("Rating prediction model is not available.")

elif option == "Identify Helpful Reviews":
    st.subheader("📚 Identify Elements that Make Reviews Helpful")
    review = st.text_area("Enter your review:")
    if st.button("Analyze Helpfulness"):
        if review:
            transformed_review = vectorizer.transform([review])
            helpfulness_score = helpfulness_model.predict(transformed_review)[0]  # Predict helpfulness score
            st.success(f"Helpfulness Score: {round(helpfulness_score, 2)} (Higher is better)")
