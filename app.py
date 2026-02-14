import streamlit as st
import joblib
import pandas as pd
import numpy as np
import logging
from typing import Dict
from pathlib import Path
import plotly.express as px
from sqlalchemy import text

# --- 1. Page Configuration ---
st.set_page_config(
    page_title="Kurdish Sentiment Analyser",
    page_icon="🪄",
    layout="wide"
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- 2. Constants & Style ---
SENTIMENT_MAP: Dict[int, str] = {
    0: "Sadness", 1: "Happiness", 2: "Fear", 
    3: "Anger", 4: "Disgust", 5: "Surprise", 6: "Sarcastic"
}

COLOR_MAP: Dict[int, str] = {
    0: "#1f77b4", 1: "#2ca02c", 2: "#9467bd", 
    3: "#d62728", 4: "#8c564b", 5: "#ff7f0e", 6: "#e377c2"
}

# --- 3. Database Connection ---
def get_db_connection():
    """
    Connects to the database using the raw string defined in secrets.toml.
    Uses 'ttl=0' to prevent stale connections in Streamlit Cloud.
    """
    try:
        # We use the connection defined in secrets.toml [connections.sqlserver]
        conn = st.connection("sqlserver", type="sql", ttl=0)
        return conn
    except Exception as e:
        logger.error(f"Database connection failed: {e}")
        return None

# --- 4. Asset Loading ---
@st.cache_resource
def load_assets():
    curr_dir = Path(__file__).parent
    try:
        model = joblib.load(curr_dir / 'sentiment_svc_model.pkl')
        tfidf = joblib.load(curr_dir / 'tfidf_vectorizer.pkl')
        return model, tfidf
    except FileNotFoundError:
        st.error("❌ Critical Error: Model files not found. Please check GitHub repo.")
        st.stop()

# --- 5. Helper Functions ---
def calculate_confidence(decision_scores, prediction_idx):
    # Softmax function to convert decision scores to probabilities
    exp_scores = np.exp(decision_scores - np.max(decision_scores))
    probs = exp_scores / exp_scores.sum()
    return float(probs[prediction_idx] * 100), probs

# --- 6. Session State Initialization ---
if 'prediction' not in st.session_state:
    st.session_state.update({
        'prediction': None, 'label': None, 'confidence': 0.0,
        'current_color': "#f0f2f6", 'user_input': "", 'probabilities': None,
        'admin_logged_in': False
    })

# --- 7. Main Application Logic ---
model, tfidf = load_assets()
db = get_db_connection()

# Sidebar
st.sidebar.title("📌 Navigation")
app_mode = st.sidebar.radio("Go to:", ["Sentiment Analyzer", "Project Info", "Admin Dashboard"])

if app_mode == "Project Info":
    st.title("📖 About the Research")
    st.info("This tool is part of a PhD study on Kurdish Natural Language Processing (NLP).")
    st.markdown("""
    **Goal:** To develop robust sentiment analysis models for the Kurdish language.
    
    **How you can help:** 1. Enter text in various Kurdish dialects.
    2. Review the AI's prediction.
    3. If the AI is wrong, use the **Report Correction** tool.
    """)

elif app_mode == "Sentiment Analyzer":
    st.title("Kurdish Sentiment Analysis")
    st.caption("PhD Research Project | Fanar Rofoo")
    
    with st.expander("🔐 Data Privacy Notice"):
        st.write("Your input is stored anonymously for model training purposes only. No personal identifiers are collected.")

    # Input Area
    user_input = st.text_area("Enter a Kurdish sentence:", height=150, placeholder="Min zor dڵxoşim...")

    # Analysis Button
    if st.button("Analyze Sentiment", type="primary", use_container_width=True):
        if user_input.strip():
            # 1. Transform input
            vec = tfidf.transform([user_input])
            
            # 2. Predict
            pred_id = int(model.predict(vec)[0])
            scores = model.decision_function(vec)[0]
            conf, probs = calculate_confidence(scores, pred_id)
            
            # 3. Update State
            st.session_state.update({
                'prediction': pred_id,
                'label': SENTIMENT_MAP[pred_id],
                'confidence': conf,
                'current_color': COLOR_MAP[pred_id],
                'user_input': user_input,
                'probabilities': probs
            })
            st.rerun()
        else:
            st.warning("⚠️ Please enter some text first.")

    # Result Display
    if st.session_state.prediction is not None:
        st.divider()
        st.markdown(f"""
            <div style="background-color: #f8f9fb; padding: 25px; border-radius: 15px; 
            border-left: 15px solid {st.session_state.current_color}; text-align: center; box-shadow: 0 4px 6px rgba(0,0,0,0.1);">
                <p style="color: #666; font-size: 14px; margin-bottom: 5px;">Detected Emotion</p>
                <h1 style="color: {st.session_state.current_color}; margin:0; font-size: 48px;">{st.session_state.label}</h1>
            </div>
        """, unsafe_allow_html=True)
        
        st.write(f"**Confidence:** {st.session_state.confidence:.1f}%")
        st.progress(st.session_state.confidence / 100)

        # Feedback Mechanism
        with st.expander("🛠️ Report Incorrect Prediction"):
            st.write("Is the AI wrong? Help us improve.")
            correct_label = st.selectbox("Select the correct emotion:", options=list(SENTIMENT_MAP.keys()), 
                                         format_func=lambda x: SENTIMENT_MAP[x])
            
            consent = st.checkbox("I consent to this text being used for retraining.")
            
            if st.button("Submit Correction", disabled=not consent):
                if db:
                    try:
                        with db.session as s:
                            # Using parameterized query for security
                            # Explicitly using [dbo].[MyData] to avoid schema confusion
                            query = text("INSERT INTO [dbo].[MyData] (user_input, model_prediction, correct_label) VALUES (:ui, :mp, :cl)")
                            s.execute(query, {
                                "ui": st.session_state.user_input, 
                                "mp": st.session_state.prediction, 
                                "cl": correct_label
                            })
                            s.commit()
                        st.success("✅ Feedback saved! Thank you for contributing.")
                    except Exception as e:
                        st.error(f"❌ Database Error: {e}")
                else:
                    st.error("⚠️ Database connection is currently unavailable.")

elif app_mode == "Admin Dashboard":
    st.title("🔐 Admin Dashboard")
    
    # Simple Login System
    if not st.session_state.admin_logged_in:
        pwd = st.text_input("Admin Password", type="password")
        if st.button("Login"):
            # Ensure ADMIN_PASSWORD is set in secrets
            if pwd == st.secrets.get("ADMIN_PASSWORD", "admin123"): 
                st.session_state.admin_logged_in = True
                st.rerun()
            else:
                st.error("❌ Incorrect password.")
    else:
        # Admin Interface
        if st.button("Logout"):
            st.session_state.admin_logged_in = False
            st.rerun()
        
        st.divider()

        if db:
            # --- IDENTITY CHECK (DEBUGGING BLOCK) ---
            try:
                with db.session as s:
                    # This query reveals exactly WHO the database thinks is connected
                    identity = s.execute(text("SELECT SUSER_NAME() as LoginName, USER_NAME() as UserName")).fetchone()
                    
                    st.info(f"👤 **Connection Identity:** Login=`{identity[0]}` | User=`{identity[1]}`")
                    
                    # If mapped to 'guest', warn the user
                    if identity[1] == 'guest':
                        st.error("⚠️ CRITICAL ISSUE: Your login is mapping to the 'guest' user! You need to fix User Mapping in SQL Server.")
            except Exception as e:
                st.warning(f"Could not verify identity: {e}")
            # ----------------------------------------

            try:
                # Explicitly using [dbo].[MyData]
                df = db.query("SELECT * FROM [dbo].[MyData]", ttl=0)
                
                if not df.empty:
                    c1, c2 = st.columns(2)
                    c1.metric("Total Samples", len(df))
                    
                    # Accuracy Calculation
                    correct = df[df['model_prediction'] == df['correct_label']].shape[0]
                    accuracy = (correct / len(df)) * 100
                    c2.metric("Current Accuracy", f"{accuracy:.1f}%")
                    
                    st.subheader("📥 Recent Submissions")
                    st.dataframe(df.tail(10), use_container_width=True)
                    
                    st.subheader("📊 Confusion Matrix")
                    cm = pd.crosstab(df['correct_label'], df['model_prediction'], rownames=['Actual'], colnames=['Predicted'])
                    fig = px.imshow(cm, text_auto=True, title="Model Performance Matrix")
                    st.plotly_chart(fig)
                else:
                    st.info("Database is connected, but the 'MyData' table is empty.")
            except Exception as e:
                st.error(f"Error loading admin data: {e}")
        else:
            st.error("Database connection unavailable.")
