import streamlit as st
import joblib
import os
import pandas as pd

# --- 1. Load Model & Vectorizer ---
current_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(current_dir, 'sentiment_svc_model.pkl')
tfidf_path = os.path.join(current_dir, 'tfidf_vectorizer.pkl')

@st.cache_resource # This keeps the model in memory so it doesn't reload every time
def load_assets():
    model = joblib.load(model_path)
    tfidf = joblib.load(tfidf_path)
    return model, tfidf

model, tfidf = load_assets()

sentiment_map = {
    0: "Sadness", 1: "Happiness", 2: "Fear", 
    3: "Anger", 4: "Disgust", 5: "Surprise", 6: "Sarcastic"
}

# --- 2. App UI Layout ---
st.set_page_config(page_title="Sentiment Analyzer", page_icon="🪄")
st.title("Kurdish Sentiment Analysis")
st.write("By Fanar Rofoo")
st.write("This sentiment analysis app is the culmination of PhD research, employing a LinearSVC model that achieves 86% accuracy.")

# --- 3. Prediction Logic ---
user_input = st.text_area("Enter a sentence to analyse:", placeholder="Kurdish text only")

# Initialize session state variables if they don't exist
if 'prediction' not in st.session_state:
    st.session_state.prediction = None
if 'label' not in st.session_state:
    st.session_state.label = None

if st.button("Analyze Sentiment"):
    if user_input.strip() == "":
        st.warning("Please enter some text first.")
    else:
        vec = tfidf.transform([user_input])
        # Store results in session_state so they persist
        st.session_state.prediction = model.predict(vec)[0]
        st.session_state.label = sentiment_map.get(st.session_state.prediction, "Unknown")

# --- 4. Display Result (Only if a prediction has been made) ---
if st.session_state.prediction is not None:
    st.divider()
    st.subheader(f"Predicted Sentiment: **{st.session_state.label}**")
    st.info(f"Class ID: {st.session_state.prediction}")
    
    if st.session_state.prediction >= 4: 
        st.balloons()

    # --- 5. Feedback Section ---
    st.divider()
    st.subheader("🛠️ Help Improve the AI")
    
    with st.expander("Report an incorrect prediction"):
        # Fixed the formatting for the table view
        st.markdown("""
        **ID | Sentiment** 0 | Sadness  
        1 | Happiness  
        2 | Fear  
        3 | Anger  
        4 | Disgust  
        5 | Surprise  
        6 | Sarcastic
        """)
        
        correct_label = st.selectbox(
            "What is the correct sentiment (0-6)?", 
            options=list(sentiment_map.keys()),
            format_func=lambda x: f"{x} - {sentiment_map[x]}"
        )
        
        if st.button("Submit Feedback"):
            feedback_data = {
                "user_input": user_input,
                "model_prediction": st.session_state.prediction,
                "correct_label": correct_label
            }
            
            df_feedback = pd.DataFrame([feedback_data])
            df_feedback.to_csv("feedback_log.csv", mode='a', header=not os.path.exists("feedback_log.csv"), index=False)
            st.success("✅ Thank you! Your feedback has been logged.")