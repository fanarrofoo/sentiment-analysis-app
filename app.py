import streamlit as st
import joblib

import os

# Get the directory where app.py is located
current_dir = os.path.dirname(os.path.abspath(__file__))

# Create full paths to the files
model_path = os.path.join(current_dir, 'sentiment_svc_model.pkl')
tfidf_path = os.path.join(current_dir, 'tfidf_vectorizer.pkl')

# Load the files using the full paths
model = joblib.load(model_path)
tfidf = joblib.load(tfidf_path)
# 1. Load the winning model and vectorizer
model = joblib.load('sentiment_svc_model.pkl')
tfidf = joblib.load('tfidf_vectorizer.pkl')

# 2. Map the numbers to names (Adjust these to match your actual sentiments!)
sentiment_map = {
    0: "Sadness",
    1: "Happiness",
    2: "Fear",
    3: "Anger",
    4: "Disgust",
    5: "Surprise",
    6: "Sarcastic"
}

# 3. App UI Layout
st.set_page_config(page_title="Sentiment Analyzer", page_icon="🪄")
st.title("Kurdish Sentiment Analysis")
st.write("By Fanar Rofoo")
st.write("This sentiment analysis app is the culmination of PhD research, employing a LinearSVC model that achieves 86% accuracy in classifying sentiment across seven distinct categories. The model is trained on a rich, purpose-built dataset collected from social media platforms, providing a nuanced and real-world understanding of contemporary emotional expression.")

# 4. User Input
user_input = st.text_area("Enter a sentence to analyse:", placeholder="Kurdish text only")

if st.button("Analyze Sentiment"):
    if user_input.strip() == "":
        st.warning("Please enter some text first.")
    else:
        # 5. Prediction Logic
        vec = tfidf.transform([user_input])
        prediction = model.predict(vec)[0]
        label = sentiment_map.get(prediction, "Unknown")
        
        # 6. Display Result
        st.divider()
        st.subheader(f"Predicted Sentiment: **{label}**")
        st.info(f"Class ID: {prediction}")
        
        # Fun visual feedback based on class
        if prediction >= 4: st.balloons()