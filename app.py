import streamlit as st
import joblib
import pandas as pd
import numpy as np
import logging
from typing import Dict, Any
from pathlib import Path
import plotly.express as px
from sqlalchemy import text

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Constants ---
SENTIMENT_MAP: Dict[int, str] = {
    0: "Sadness", 1: "Happiness", 2: "Fear", 
    3: "Anger", 4: "Disgust", 5: "Surprise", 6: "Sarcastic"
}

COLOR_MAP: Dict[int, str] = {
    0: "#1f77b4", 1: "#2ca02c", 2: "#9467bd", 
    3: "#d62728", 4: "#8c564b", 5: "#ff7f0e", 6: "#e377c2"
}

# --- 1. App Configuration ---
st.set_page_config(page_title="Kurdish Sentiment Analyser", page_icon="🪄", layout="wide")

# --- 2. Database Connection ---
def get_db_connection():
    try:
        # Note: Ensure your secrets.toml [connections.sqlserver] url uses:
        # mssql+pyodbc://user:pass@140.82.39.222:1744/TestDB?driver=ODBC+Driver+17+for+SQL+Server
        conn = st.connection("sqlserver", type="sql", ttl=0)
        return conn
    except Exception as e:
        logger.error(f"DB Connection failed: {e}")
        return None

# --- 3. Asset Loading ---
@st.cache_resource
def load_assets():
    curr_dir = Path(__file__).parent
    model = joblib.load(curr_dir / 'sentiment_svc_model.pkl')
    tfidf = joblib.load(curr_dir / 'tfidf_vectorizer.pkl')
    return model, tfidf

# --- 4. Logic Functions ---
def calculate_confidence(decision_scores, prediction):
    exp_scores = np.exp(decision_scores - np.max(decision_scores))
    probs = exp_scores / exp_scores.sum()
    return float(probs[prediction] * 100), probs

# Initialize Session State
if 'prediction' not in st.session_state:
    st.session_state.update({
        'prediction': None, 'label': None, 'confidence': 0.0,
        'current_color': "#f0f2f6", 'user_input': "", 'probabilities': None
    })

model, tfidf = load_assets()
db = get_db_connection()

# --- 5. Navigation ---
st.sidebar.title("📌 Menu")
app_mode = st.sidebar.radio("Go to:", ["Sentiment Analyzer", "User Guide", "Admin Dashboard"])

if app_mode == "User Guide":
    st.title("📖 User Guide")
    st.write("This tool is part of a PhD research project on Kurdish NLP.")
    st.info("Input Kurdish text to detect emotions. Your feedback helps improve accuracy!")

elif app_mode == "Sentiment Analyzer":
    st.title("Kurdish Sentiment Analysis")
    st.caption("PhD Research Project | Fanar Rofoo")
    
    with st.expander("🔐 Data Privacy"):
        st.write("Feedback is stored anonymized for academic research.")

    user_input = st.text_area("Enter a Kurdish sentence:", height=150)

    if st.button("Analyze Sentiment", type="primary"):
        if user_input.strip():
            vec = tfidf.transform([user_input])
            pred_id = int(model.predict(vec)[0])
            conf, probs = calculate_confidence(model.decision_function(vec)[0], pred_id)
            
            st.session_state.update({
                'prediction': pred_id, 'label': SENTIMENT_MAP[pred_id],
                'confidence': conf, 'current_color': COLOR_MAP[pred_id],
                'user_input': user_input, 'probabilities': probs
            })
            st.rerun()

    if st.session_state.prediction is not None:
        st.divider()
        st.markdown(f"""
            <div style="background-color: #f8f9fb; padding: 25px; border-radius: 15px; 
            border-left: 15px solid {st.session_state.current_color}; text-align: center;">
                <h1 style="color: {st.session_state.current_color};">{st.session_state.label}</h1>
            </div>
        """, unsafe_allow_html=True)
        st.write(f"Confidence: {st.session_state.confidence:.1f}%")

        # --- Feedback Section ---
        with st.expander("🛠️ Report Correction"):
            correct_label = st.selectbox("Correct Sentiment?", options=list(SENTIMENT_MAP.keys()), 
                                         format_func=lambda x: SENTIMENT_MAP[x])
            consent = st.checkbox("I consent to storage of this text.")
            
            # The button is now INSIDE the expander where 'consent' and 'correct_label' are defined
            if st.button("Submit Feedback"):
                if not consent:
                    st.error("Please check the consent box first.")
                elif db:
                    try:
                        with db.session as s:
                            query = text("INSERT INTO MyData (user_input, model_prediction, correct_label) VALUES (:ui, :mp, :cl)")
                            s.execute(query, {"ui": st.session_state.user_input, "mp": st.session_state.prediction, "cl": correct_label})
                            s.commit()
                        st.success("✅ Feedback logged!")
                    except Exception as e:
                        st.error(f"Database error: {e}")
                else:
                    st.error("Database connection missing.")

elif app_mode == "Admin Dashboard":
    st.title("🔐 Admin Dashboard")
    pwd = st.text_input("Admin Password", type="password")
    
    if pwd == st.secrets.get("ADMIN_PASSWORD"):
        if db:
            try:
                df = db.query("SELECT * FROM MyData", ttl=0)
                if not df.empty:
                    st.metric("Total Submissions", len(df))
                    st.dataframe(df, use_container_width=True)
                else:
                    st.info("No data found.")
            except Exception as e:
                st.error(f"Query Error: {e}")
