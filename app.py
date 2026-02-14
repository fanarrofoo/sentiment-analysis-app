import streamlit as st
import joblib
import pandas as pd
import numpy as np
import logging
from typing import Tuple, Optional, Dict, Any
from pathlib import Path
import plotly.express as px

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

# --- 1. App Configuration ---
st.set_page_config(page_title=PAGE_TITLE, page_icon=PAGE_ICON, layout="wide")

# --- 2. Database Connection (The Modern Way) ---
def get_db_connection():
    try:
        # This will attempt to connect and cache the connection
        conn = st.connection("sqlserver", type="sql")
        return conn
    except Exception as e:
        # This will show you the EXACT error on the app interface
        st.sidebar.error(f"DB Error: {str(e)}")
        return None

# --- 3. Utility Functions ---
def initialize_session_state():
    defaults = {
        'prediction': None, 'label': None, 'confidence': 0.0,
        'current_color': "#f0f2f6", 'user_input': "", 'probabilities': None
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val

@st.cache_resource
def load_assets():
    curr_dir = Path(__file__).parent
    try:
        model = joblib.load(curr_dir / 'sentiment_svc_model.pkl')
        tfidf = joblib.load(curr_dir / 'tfidf_vectorizer.pkl')
        return model, tfidf
    except Exception as e:
        st.error(f"Model files missing or corrupt: {e}")
        st.stop()

def calculate_confidence_scores(decision_scores, prediction):
    exp_scores = np.exp(decision_scores - np.max(decision_scores))
    probs = exp_scores / exp_scores.sum()
    return float(probs[prediction] * 100), probs

def calculate_metrics(df):
    if df.empty: return None
    from sklearn.metrics import precision_recall_fscore_support, confusion_matrix
    y_true, y_pred = df['correct_label'], df['model_prediction']
    acc = (y_true == y_pred).mean() * 100
    p, r, f, _ = precision_recall_fscore_support(y_true, y_pred, labels=list(SENTIMENT_MAP.keys()), average=None, zero_division=0)
    
    return {
        'total': len(df), 'accuracy': acc,
        'metrics_df': pd.DataFrame({
            'Sentiment': [SENTIMENT_MAP[i] for i in SENTIMENT_MAP.keys()],
            'Precision (%)': p * 100, 'Recall (%)': r * 100, 'F1-Score (%)': f * 100
        }),
        'cm': confusion_matrix(y_true, y_pred, labels=list(SENTIMENT_MAP.keys()))
    }

# --- 4. UI Components ---
def render_sentiment_card(label, color):
    st.markdown(f"""
        <div style="background-color: #f8f9fb; padding: 30px; border-radius: 15px; 
        border-left: 15px solid {color}; text-align: center; box-shadow: 2px 2px 10px rgba(0,0,0,0.05);">
            <p style="color: #555; font-weight: bold;">AI Prediction:</p>
            <h1 style="color: {color}; font-size: 60px;">{label}</h1>
        </div>
    """, unsafe_allow_html=True)

# --- 5. Main Logic ---
initialize_session_state()
model, tfidf = load_assets()
db = get_db_connection()

st.sidebar.title("📌 Navigation")
app_mode = st.sidebar.radio("Go to:", ["Analyzer", "Admin Dashboard"])

if app_mode == "Analyzer":
    st.title("Kurdish Sentiment Analysis")
    st.caption("PhD Research Project by Fanar Rofoo")

    user_input = st.text_area("Enter Kurdish text:", height=150, max_chars=MAX_TEXT_LENGTH)
    
    if st.button("Analyze Sentiment", type="primary"):
        if not user_input.strip():
            st.warning("Please enter some text.")
        else:
            vec = tfidf.transform([user_input])
            pred_id = int(model.predict(vec)[0])
            dec_scores = model.decision_function(vec)[0]
            conf, probs = calculate_confidence_scores(dec_scores, pred_id)
            
            st.session_state.update({
                'prediction': pred_id, 'label': SENTIMENT_MAP[pred_id],
                'confidence': conf, 'current_color': COLOR_MAP[pred_id],
                'user_input': user_input, 'probabilities': probs
            })
            st.rerun()

    if st.session_state.prediction is not None:
        st.divider()
        render_sentiment_card(st.session_state.label, st.session_state.current_color)
        st.write(f"**Confidence:** {st.session_state.confidence:.1f}%")
        st.progress(st.session_state.confidence / 100)

        with st.expander("🛠️ Report Correction"):
            correct_label = st.selectbox("Correct Sentiment?", options=list(SENTIMENT_MAP.keys()), 
                                         format_func=lambda x: SENTIMENT_MAP[x])
            if st.button("Submit Feedback"):
                if db:
                    # Use the connection to execute an insert
                    with db.session as session:
                        from sqlalchemy import text
                        query = text("INSERT INTO MyData (user_input, model_prediction, correct_label) VALUES (:ui, :mp, :cl)")
                        session.execute(query, {"ui": st.session_state.user_input, "mp": st.session_state.prediction, "cl": correct_label})
                        session.commit()
                    st.success("Feedback saved to TestDB!")
                else:
                    st.error("Database connection missing.")

elif app_mode == "Admin Dashboard":
    st.title("📈 Model Performance")
    pwd = st.text_input("Admin Password", type="password")
    
    if pwd == st.secrets.get("ADMIN_PASSWORD"):
        if db:
            # Query the database
            df = db.query("SELECT * FROM MyData")
            m = calculate_metrics(df)
            
            if m:
                c1, c2 = st.columns(2)
                c1.metric("Total Samples", m['total'])
                c2.metric("Accuracy", f"{m['accuracy']:.2f}%")
                
                st.subheader("Class Breakdown")
                st.dataframe(m['metrics_df'], hide_index=True)
                
                fig = px.imshow(m['cm'], text_auto=True, 
                                x=list(SENTIMENT_MAP.values()), y=list(SENTIMENT_MAP.values()),
                                labels=dict(x="Actual", y="Predicted"), title="Confusion Matrix")
                st.plotly_chart(fig)
            else:
                st.info("No data in database yet.")
        else:
            st.error("Database not connected.")
