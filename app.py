
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

# Predefined drug names
drug_names = ["Paracetamol", "Ibuprofen", "Aspirin", "Metformin", "Amoxicillin", "Omeprazole", "Losartan", "Atorvastatin", "Simvastatin", "Gabapentin", "Ciprofloxacin", "Cetirizine", "Prednisone", "Hydrochlorothiazide", "Levothyroxine"]

# Streamlit UI
st.set_page_config(page_title="Drug Review Analysis", page_icon="💊", layout="wide")
st.title("💊 Drug Review Analysis App")
st.markdown("---")

# Dropdown for Function Selection
option = st.selectbox("Select a Functionality:", [
    "Sentiment Analysis",
    "Condition Prediction",
    "Understand Negative Reviews"
])

st.markdown("---")

if option == "Sentiment Analysis":
    st.header("🔍 Predict Sentiment of a Drug Review")
    review = st.text_area("Enter your review:", height=150)
    if st.button("Analyze Sentiment", use_container_width=True):
        if review:
            transformed_review = vectorizer.transform([review])
            sentiment_probs = sentiment_model.predict_proba(transformed_review)[0]
            sentiment_prediction = np.argmax(sentiment_probs)  # Ensure correct classification
            sentiment_map = {0: "😡 Negative", 1: "😐 Neutral", 2: "😊 Positive"}
            sentiment_label = sentiment_map.get(sentiment_prediction, "Unknown")
            st.success(f"Predicted Sentiment: {sentiment_label}")

            # Identify key elements within sentiment analysis
            words = review.split()
            key_elements = []
            
            if len(words) > 50:
                key_elements.append("✔️ Detailed Explanation")
            if any(word.lower() in ["effective", "works", "relief", "helped"] for word in words):
                key_elements.append("✔️ Mentions Effectiveness")
            if any(word.lower() in ["side effect", "bad", "pain", "issues"] for word in words):
                key_elements.append("✔️ Mentions Side Effects/Problems")
            if any(word.lower() in ["doctor", "prescribed", "recommend"] for word in words):
                key_elements.append("✔️ Includes Medical Advice")
            
            if key_elements:
                st.info("Key Elements: " + ", ".join(key_elements))

elif option == "Condition Prediction":
    st.header("🧐 Predict Condition from Review")
    review = st.text_area("Enter your review:", height=150)
    if st.button("Predict Condition", use_container_width=True):
        if review:
            transformed_review = con_vectorizer.transform([review])  # Using con_vectorizer
            condition_pred = nb_condition_model.predict(transformed_review)[0]
            predicted_condition = label_encoder.inverse_transform([condition_pred])[0]
            st.success(f"Predicted Condition: **{predicted_condition}**")

elif option == "Understand Negative Reviews":
    st.header("📉 Analyze Negative Reviews & Improve the Drug")
    review = st.text_area("Enter your review:", height=150)
    drug_name = st.selectbox("Select a Drug Name", drug_names)
    if st.button("Analyze Negative Review", use_container_width=True):
        if review:
            transformed_review = vectorizer.transform([review])
            sentiment_probs = sentiment_model.predict_proba(transformed_review)[0]
            sentiment_prediction = np.argmax(sentiment_probs)
            
            if sentiment_prediction == 0:
                st.error("❌ This review indicates dissatisfaction!")
                st.write("### Possible Issues & Suggested Improvements:")
                if "side effect" in review.lower() or "bad" in review.lower():
                    st.warning("- Reduce side effects by modifying drug composition.")
                if "not working" in review.lower() or "ineffective" in review.lower():
                    st.warning("- Improve effectiveness by adjusting dosage or formulation.")
                if "expensive" in review.lower():
                    st.warning("- Consider making the drug more affordable or providing discounts.")
                st.info(f"Feedback considered for {drug_name}.")
            else:
                st.success("This review does not seem highly negative.")
