import streamlit as st
import joblib
import os
import pandas as pd
import numpy as np
import logging
from typing import Tuple, Optional, Dict, Any
from pathlib import Path
import plotly.express as px
from sqlalchemy import text

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Constants ---
PAGE_TITLE = "Kurdish Sentiment Analyser"
PAGE_ICON = "🪄"
SENTIMENT_MAP: Dict[int, str] = {
    0: "Sadness", 1: "Happiness", 2: "Fear", 
    3: "Anger", 4: "Disgust", 5: "Surprise", 6: "Sarcastic"
}

COLOR_MAP: Dict[int, str] = {
    0: "#1f77b4", 1: "#2ca02c", 2: "#9467bd", 
    3: "#d62728", 4: "#8c564b", 5: "#ff7f0e", 6: "#e377c2"
}

MAX_TEXT_LENGTH = 5000
MIN_TEXT_LENGTH = 1

# --- 1. App Configuration ---
st.set_page_config(page_title=PAGE_TITLE, page_icon=PAGE_ICON, layout="wide")

# --- 2. Database Connection ---
def get_db_connection():
    try:
        # Use st.connection with "pool_pre_ping" to handle drops
        conn = st.connection(
            "sqlserver", 
            type="sql", 
            ttl=0, 
            pool_pre_ping=True
        )
        return conn
    except Exception as e:
        logger.error(f"SQL Connection Error: {str(e)}")
        return None

# --- Inside the Feedback Section ---
if st.button("Submit Feedback", disabled=not consent):
    if db:
        try:
            # Using the 'with' block ensures the session is closed correctly
            with db.session as s:
                # Use standard SQL Parameter markers (?) for pyodbc
                sql = text("INSERT INTO MyData (user_input, model_prediction, correct_label) VALUES (:ui, :mp, :cl)")
                s.execute(sql, {
                    "ui": st.session_state.user_input, 
                    "mp": int(st.session_state.prediction), 
                    "cl": int(correct_label)
                })
                s.commit()
            st.success("✅ Thank you! Feedback logged.")
        except Exception as e:
            st.error(f"❌ Handshake failed: {str(e)}")
    else:
        st.error("❌ Database connection unavailable.")

# --- 3. Utility Functions ---
def initialize_session_state():
    defaults = {
        'prediction': None, 'label': None, 'confidence': 0.0,
        'current_color': "#f0f2f6", 'user_input': "", 'probabilities': None
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

@st.cache_resource
def load_assets():
    current_dir = Path(__file__).parent
    model = joblib.load(current_dir / 'sentiment_svc_model.pkl')
    tfidf = joblib.load(current_dir / 'tfidf_vectorizer.pkl')
    return model, tfidf

def calculate_confidence_scores(decision_scores, prediction):
    exp_scores = np.exp(decision_scores - np.max(decision_scores))
    probabilities = exp_scores / exp_scores.sum()
    return float(probabilities[prediction] * 100), probabilities

def calculate_classification_metrics(df):
    if df.empty: return None
    from sklearn.metrics import precision_recall_fscore_support, confusion_matrix
    y_true, y_pred = df['correct_label'], df['model_prediction']
    
    accuracy = (y_true == y_pred).mean() * 100
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=list(SENTIMENT_MAP.keys()), average=None, zero_division=0
    )
    
    metrics_df = pd.DataFrame({
        'Sentiment': [SENTIMENT_MAP[i] for i in SENTIMENT_MAP.keys()],
        'Precision (%)': precision * 100,
        'Recall (%)': recall * 100,
        'F1-Score (%)': f1 * 100
    })
    
    return {
        'total': len(df), 'accuracy': accuracy, 'metrics_df': metrics_df,
        'cm': confusion_matrix(y_true, y_pred, labels=list(SENTIMENT_MAP.keys()))
    }

# --- 4. UI Components ---
def render_sentiment_card(label: str, color: str):
    st.markdown(f"""
        <div style="background-color: #f8f9fb; padding: 30px; border-radius: 15px; 
            border-left: 15px solid {color}; text-align: center; box-shadow: 2px 2px 10px rgba(0,0,0,0.05); margin-bottom: 10px;">
            <p style="color: #555; font-size: 18px; margin-bottom: 5px; font-weight: bold;">The AI detected:</p>
            <h1 style="color: {color}; margin: 0; font-size: 65px; text-transform: uppercase;">{label}</h1>
        </div>
    """, unsafe_allow_html=True)

# --- 5. Application Initiation ---
initialize_session_state()
model, tfidf = load_assets()
db = get_db_connection()

# --- 6. Sidebar ---
st.sidebar.title("📌 Menu")
app_mode = st.sidebar.radio("Go to:", ["Sentiment Analyzer", "User Guide", "Admin Dashboard"])

# --- 7. Main Logic ---
if app_mode == "User Guide":
    st.title("📖 User Guide & Instructions")
    st.write("Thank you for participating in this Kurdish NLP research.")
    st.subheader("1. Entering Text")
    st.info("The model is optimized for Kurdish Unicode. Using other languages may result in inaccurate predictions.")
    st.subheader("2. Providing Feedback")
    st.write("If the AI is wrong, use the 'Report' section. This helps retrain the model for better Kurdish dialect support.")

elif app_mode == "Sentiment Analyzer":
    st.title("Kurdish Sentiment Analysis")
    st.write("By Fanar Rofoo | PhD Research Project")
    
    with st.expander("📖 About this App"):
        st.markdown("This app uses a **Linear Support Vector Classifier (LinearSVC)**. Kurdish is a low-resource language, and your feedback helps bridge the AI gap.")

    with st.expander("🔐 Data Privacy"):
        st.write("By providing feedback, you consent to the storage of anonymized text for academic purposes.")

    user_input = st.text_area("Enter a Kurdish sentence:", height=150, max_chars=MAX_TEXT_LENGTH)

    if st.button("Analyze Sentiment", type="primary", use_container_width=True):
        if not user_input.strip():
            st.warning("⚠️ Please enter some text first.")
        else:
            with st.spinner("🔍 Analyzing..."):
                vec = tfidf.transform([user_input])
                pred_id = int(model.predict(vec)[0])
                conf, probs = calculate_confidence_scores(model.decision_function(vec)[0], pred_id)
                
                st.session_state.update({
                    'prediction': pred_id, 'label': SENTIMENT_MAP[pred_id],
                    'confidence': conf, 'current_color': COLOR_MAP[pred_id],
                    'user_input': user_input, 'probabilities': probs
                })
                st.rerun()

    if st.session_state.prediction is not None:
        st.divider()
        render_sentiment_card(st.session_state.label, st.session_state.current_color)
        st.write(f"**Model Confidence:** {st.session_state.confidence:.1f}%")
        st.progress(st.session_state.confidence / 100)

        with st.expander("🛠️ Report an incorrect prediction"):
            correct_label = st.selectbox("What is the correct sentiment?", options=list(SENTIMENT_MAP.keys()), format_func=lambda x: f"{x} - {SENTIMENT_MAP[x]}")
            consent = st.checkbox("I consent to the anonymized storage of this text.")
            
            if st.button("Submit Feedback", disabled=not consent):
                if db:
                    try:
                        with db.session as s:
                            s.execute(
                                text("INSERT INTO MyData (user_input, model_prediction, correct_label) VALUES (:ui, :mp, :cl)"),
                                {"ui": st.session_state.user_input, "mp": st.session_state.prediction, "cl": correct_label}
                            )
                            s.commit()
                        st.success("✅ Thank you! Feedback logged.")
                    except Exception as e:
                        st.error(f"❌ Database error: {str(e)}")
                else:
                    st.error("❌ Database connection unavailable.")

elif app_mode == "Admin Dashboard":
    st.title("🔐 Admin Dashboard")
    
    if "admin_logged_in" not in st.session_state:
        st.session_state.admin_logged_in = False

    if not st.session_state.admin_logged_in:
        pwd = st.text_input("Admin Password", type="password")
        if st.button("Login"):
            if pwd == st.secrets.get("ADMIN_PASSWORD"):
                st.session_state.admin_logged_in = True
                st.rerun()
            else:
                st.error("❌ Incorrect password.")
    else:
        if st.button("Logout"):
            st.session_state.admin_logged_in = False
            st.rerun()
            
        if db:
            try:
                # Use st.connection's query method
                df = db.query("SELECT * FROM MyData", ttl=0)
                # ... rest of your metrics code ...
            except Exception as e:
                st.error(f"Connection established but query failed: {e}")
        else:
            st.error("Database connection could not be established. Please check server firewall settings.")
