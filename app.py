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

import pandas as pd
import os

# 1. After showing the prediction, add a feedback section
st.divider()
st.subheader("🛠️ Help Improve the AI")

with st.expander("Report an incorrect prediction"):
    st.write("If the model got this wrong, please let us know the correct sentiment from the list below:")
st.write("""ID	Sentiment
0	Sadness
1	Happines
2	Fear
3	Anger
4	Disgust
5	Surprise
6	Sarcastic""")
    # User selects what the label SHOULD have been
    correct_label = st.selectbox(
        "What is the correct sentiment (0-6)?", 
        options=[0, 1, 2, 3, 4, 5, 6],
        index=3 # Defaults to Neutral
    )
    
    if st.button("Submit Feedback"):
        # Create a small dictionary of the error
        feedback_data = {
            "user_input": user_input,      # The text the user entered
            "model_prediction": prediction, # What the model guessed
            "correct_label": correct_label  # What the user says is right
        }
        
        # Save to a CSV file (Appends to the file if it exists)
        df_feedback = pd.DataFrame([feedback_data])
        df_feedback.to_csv("feedback_log.csv", mode='a', header=not os.path.exists("feedback_log.csv"), index=False)
        
        st.success("✅ Thank you! Your feedback has been logged for the next model retraining.")
        # 6. Display Result
        st.divider()
        st.subheader(f"Predicted Sentiment: **{label}**")
        st.info(f"Class ID: {prediction}")

    
        # Fun visual feedback based on class
        if prediction >= 4: st.balloons()