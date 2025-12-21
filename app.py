import streamlit as st
import joblib
import os
import pandas as pd
import numpy as np
from st_supabase_connection import SupabaseConnection
import plotly.express as px

# --- 1. App Configuration ---
st.set_page_config(page_title="Kurdish Sentiment Analyser", page_icon="🪄", layout="wide")

# --- 2. Load Model & Assets ---
current_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(current_dir, 'sentiment_svc_model.pkl')
tfidf_path = os.path.join(current_dir, 'tfidf_vectorizer.pkl')

@st.cache_resource
def load_assets():
    model = joblib.load(model_path)
    tfidf = joblib.load(tfidf_path)
    return model, tfidf

model, tfidf = load_assets()

sentiment_map = {
    0: "Sadness", 1: "Happiness", 2: "Fear", 
    3: "Anger", 4: "Disgust", 5: "Surprise", 6: "Sarcastic"
}

color_map = {
    0: "#1f77b4", # Sadness (Blue)
    1: "#2ca02c", # Happiness (Green)
    2: "#9467bd", # Fear (Purple)
    3: "#d62728", # Anger (Red)
    4: "#8c564b", # Disgust (Brown)
    5: "#ff7f0e", # Surprise (Orange)
    6: "#e377c2"  # Sarcastic (Pink)
}

# Database Connection
conn = st.connection("supabase", type=SupabaseConnection)

# --- 3. Initialize Session State (Prevents AttributeErrors) ---
if 'prediction' not in st.session_state:
    st.session_state.prediction = None
if 'label' not in st.session_state:
    st.session_state.label = None
if 'confidence' not in st.session_state:
    st.session_state.confidence = 0.0
if 'current_color' not in st.session_state:
    st.session_state.current_color = "#f0f2f6"

# --- 4. Sidebar Navigation ---
st.sidebar.title("📌 Menu")
app_mode = st.sidebar.radio("Go to:", ["Sentiment Analyzer", "User Guide"])

# --- 5. User Guide Function ---
def render_user_guide():
    st.title("📖 User Guide & Instructions")
    st.info("Thank you for participating in this Kurdish NLP research.")
    
    st.subheader("1. How to use")
    st.write("Enter Kurdish text and click 'Analyze'. The AI will determine the emotional tone.")
    
    st.subheader("2. Privacy & Ethics")
    st.write("Your data is anonymized. No personal identifiers are collected.")
    
    st.subheader("3. Citation")
    st.code("Rofoo, F. (2025). Kurdish Sentiment Analysis: A LinearSVC Approach [Web App].", language="text")
    
    st.subheader("4. Open Science")
    st.write("Your feedback helps build an open-source dataset for the Kurdish language.")

# --- 6. Main App Logic ---
if app_mode == "User Guide":
    render_user_guide()

else:
    # --- Sentiment Analyzer Page ---
    st.title("Kurdish Sentiment Analysis")
    st.write("By Fanar Rofoo | PhD Research Project")
    
    with st.expander("📖 About this App"):
        st.markdown("""
        This app is a part of a PhD, in which a **Linear Support Vector Classifier (LinearSVC)** is used after testing many other classifiers. 
        Kurdish is a low-resource language, and this project aims to improve AI understanding of its dialects and sentiments.
        """)

    with st.expander("🔐 Data Privacy"):
        st.write("By providing feedback, you consent to the storage of anonymized text for academic purposes.")

    # Input Area
    user_input = st.text_area("Enter a Kurdish sentence:", placeholder="e.g., Ez pir kêfxweş im", height=150)

    if st.button("Analyze Sentiment", type="primary"):
        if user_input.strip() == "":
            st.warning("Please enter some text first.")
        else:
            # A. Vectorize and Predict
            vec = tfidf.transform([user_input])
            pred_id = int(model.predict(vec)[0])
            
            # B. Calculate Confidence using Softmax on Decision Function
            # Softmax: P(y_i) = exp(z_i) / sum(exp(z_j))
            decision_scores = model.decision_function(vec)[0]
            exp_scores = np.exp(decision_scores - np.max(decision_scores))
            probabilities = exp_scores / exp_scores.sum()
            conf_score = probabilities[pred_id] * 100

            # C. Update Session State
            st.session_state.prediction = pred_id
            st.session_state.label = sentiment_map.get(pred_id, "Unknown")
            st.session_state.confidence = conf_score
            st.session_state.current_color = color_map.get(pred_id, "#f0f2f6")

    # --- 7. High-Visibility Result Display ---
    if st.session_state.prediction is not None:
        st.divider()
        
        res_label = st.session_state.label
        res_conf = st.session_state.confidence
        res_color = st.session_state.current_color

        # Result Card
        sentiment_html = f"""
            <div style="
                background-color: #f8f9fb; 
                padding: 30px; 
                border-radius: 15px; 
                border-left: 15px solid {res_color};
                text-align: center;
                box-shadow: 2px 2px 10px rgba(0,0,0,0.05);
                margin-bottom: 10px;">
                <p style="color: #555; font-size: 18px; margin-bottom: 5px; font-weight: bold;">The AI detected:</p>
                <h1 style="color: {res_color}; margin: 0; font-size: 65px; text-transform: uppercase;">
                    {res_label}
                </h1>
            </div>
        """
        st.markdown(sentiment_html, unsafe_allow_html=True)
        
        # Confidence Meter
        st.write(f"**Model Confidence:** {res_conf:.1f}%")
        st.progress(res_conf / 100)
        st.success("✅ Analysis completed successfully!")

        # --- 8. Feedback Section ---
        st.divider()
        with st.expander("🛠️ Report an incorrect prediction"):
            correct_label = st.selectbox(
                "What is the correct sentiment?", 
                options=list(sentiment_map.keys()),
                format_func=lambda x: f"{x} - {sentiment_map[x]}"
            )
            consent = st.checkbox("I consent to the anonymized storage of this text.")
            
            if consent:
                if st.button("Submit Feedback"):
                    try:
                        feedback_data = {
                            "user_input": user_input,
                            "model_prediction": int(st.session_state.prediction),
                            "correct_label": int(correct_label)
                        }
                        conn.table("sentiment_feedback").insert(feedback_data).execute()
                        st.success("✅ Thank you! Your feedback has been logged.")
                    except Exception as e:
                        st.error(f"Error: {e}")
            else:
                st.button("Submit Feedback", disabled=True, help="Check consent box first")

    # --- 9. Admin Dashboard ---
    st.divider()
    with st.expander("🔐 Admin Access"):
        pwd = st.text_input("Admin Password", type="password")
        if pwd == st.secrets["ADMIN_PASSWORD"]:
            try:
                res = conn.table("sentiment_feedback").select("*").execute()
                if res.data:
                    df = pd.DataFrame(res.data)
                    
                    # Metrics
                    total = len(df)
                    acc = (len(df[df['model_prediction'] == df['correct_label']]) / total) * 100 if total > 0 else 0
                    
                    c1, c2 = st.columns(2)
                    c1.metric("Total Submissions", total)
                    c2.metric("Live Accuracy", f"{acc:.1f}%")

                    # Confusion Matrix Plot
                    df['Predicted'] = df['model_prediction'].map(sentiment_map)
                    df['Actual'] = df['correct_label'].map(sentiment_map)
                    matrix = pd.crosstab(df['Predicted'], df['Actual'])
                    st.plotly_chart(px.imshow(matrix, text_auto=True, title="Confusion Matrix"), use_container_width=True)
                    
                    st.dataframe(df)
                else:
                    st.info("Waiting for data...")
            except Exception as e:
                st.error(f"Fetch Error: {e}")
