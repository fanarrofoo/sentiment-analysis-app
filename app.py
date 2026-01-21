"""
Kurdish Sentiment Analysis Application
=======================================
A Streamlit-based web application for Kurdish NLP sentiment analysis using LinearSVC.

Author: Fanar Rofoo
PhD Research Project
Supervised by: Prof. Dr. Shareef M. Shareef, Dr. Polla Fattah
"""

import streamlit as st
import joblib
import os
import pandas as pd
import numpy as np
import logging
from typing import Tuple, Optional, Dict, Any
from pathlib import Path
from st_supabase_connection import SupabaseConnection
import plotly.express as px
from plotly.graph_objects import Figure

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Constants ---
PAGE_TITLE = "Kurdish Sentiment Analyser"
PAGE_ICON = "🪄"
SENTIMENT_MAP: Dict[int, str] = {
    0: "Sadness",
    1: "Happiness",
    2: "Fear",
    3: "Anger",
    4: "Disgust",
    5: "Surprise",
    6: "Sarcastic"
}

COLOR_MAP: Dict[int, str] = {
    0: "#1f77b4",  # Sadness (Blue)
    1: "#2ca02c",  # Happiness (Green)
    2: "#9467bd",  # Fear (Purple)
    3: "#d62728",  # Anger (Red)
    4: "#8c564b",  # Disgust (Brown)
    5: "#ff7f0e",  # Surprise (Orange)
    6: "#e377c2"   # Sarcastic (Pink)
}

# Input validation constants
MAX_TEXT_LENGTH = 5000
MIN_TEXT_LENGTH = 1

# --- 1. App Configuration ---
st.set_page_config(page_title=PAGE_TITLE, page_icon=PAGE_ICON, layout="wide")

# --- 2. Utility Functions ---
def initialize_session_state() -> None:
    """Initialize all session state variables to prevent AttributeErrors."""
    defaults = {
        'prediction': None,
        'label': None,
        'confidence': 0.0,
        'current_color': "#f0f2f6",
        'user_input': "",
        'probabilities': None
    }
    for key, default_value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = default_value

def validate_text_input(text: str) -> Tuple[bool, Optional[str]]:
    """
    Validate user input text.
    
    Args:
        text: Input text to validate
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    if not text or not text.strip():
        return False, "Please enter some text first."
    
    text = text.strip()
    
    if len(text) < MIN_TEXT_LENGTH:
        return False, f"Text is too short (minimum {MIN_TEXT_LENGTH} character)."
    
    if len(text) > MAX_TEXT_LENGTH:
        return False, f"Text is too long (maximum {MAX_TEXT_LENGTH} characters). Please shorten your input."
    
    # Check for suspicious patterns (basic security check)
    if len(text.split()) > 1000:  # Very long inputs might be spam
        return False, "Input appears to be too long. Please provide a sentence or short paragraph."
    
    return True, None

@st.cache_resource
def load_assets() -> Tuple[Any, Any]:
    """
    Load model and vectorizer with error handling.
    
    Returns:
        Tuple of (model, tfidf_vectorizer)
        
    Raises:
        FileNotFoundError: If model files are missing
        Exception: If model loading fails
    """
    current_dir = Path(__file__).parent
    model_path = current_dir / 'sentiment_svc_model.pkl'
    tfidf_path = current_dir / 'tfidf_vectorizer.pkl'
    
    # Check file existence
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    if not tfidf_path.exists():
        raise FileNotFoundError(f"Vectorizer file not found: {tfidf_path}")
    
    try:
        logger.info("Loading model and vectorizer...")
        model = joblib.load(model_path)
        tfidf = joblib.load(tfidf_path)
        logger.info("Model and vectorizer loaded successfully")
        return model, tfidf
    except Exception as e:
        logger.error(f"Error loading models: {str(e)}")
        raise

def calculate_confidence_scores(decision_scores: np.ndarray, prediction: int) -> Tuple[float, np.ndarray]:
    """
    Calculate confidence scores using softmax normalization.
    
    Args:
        decision_scores: Raw decision function scores from model
        prediction: Predicted class index
        
    Returns:
        Tuple of (confidence_percentage, probabilities_array)
    """
    # Numerical stability: subtract max before exp to prevent overflow
    exp_scores = np.exp(decision_scores - np.max(decision_scores))
    probabilities = exp_scores / exp_scores.sum()
    confidence_pct = float(probabilities[prediction] * 100)
    return confidence_pct, probabilities

def safe_predict(text: str, model: Any, tfidf: Any) -> Tuple[Optional[int], Optional[str], Optional[float], Optional[np.ndarray]]:
    """
    Safely perform prediction with error handling.
    
    Args:
        text: Input text to predict
        model: Trained model
        tfidf: TF-IDF vectorizer
        
    Returns:
        Tuple of (prediction_id, label, confidence, probabilities)
        Returns (None, None, None, None) on error
    """
    try:
        # Vectorize input
        vec = tfidf.transform([text])
        
        # Get prediction
        pred_id = int(model.predict(vec)[0])
        
        # Validate prediction is in valid range
        if pred_id not in SENTIMENT_MAP:
            logger.warning(f"Invalid prediction ID: {pred_id}")
            return None, None, None, None
        
        # Calculate confidence scores
        decision_scores = model.decision_function(vec)[0]
        confidence, probabilities = calculate_confidence_scores(decision_scores, pred_id)
        
        label = SENTIMENT_MAP[pred_id]
        
        return pred_id, label, confidence, probabilities
        
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        return None, None, None, None

def calculate_classification_metrics(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Calculate comprehensive classification metrics.
    
    Args:
        df: DataFrame with 'model_prediction' and 'correct_label' columns
        
    Returns:
        Dictionary with various metrics
    """
    if df.empty or len(df) == 0:
        return {
            'total': 0,
            'accuracy': 0.0,
            'precision': {},
            'recall': {},
            'f1_score': {}
        }
    
    from sklearn.metrics import precision_recall_fscore_support, confusion_matrix
    
    y_true = df['correct_label'].values
    y_pred = df['model_prediction'].values
    
    # Overall accuracy
    accuracy = (y_true == y_pred).mean() * 100
    
    # Per-class metrics
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=list(SENTIMENT_MAP.keys()), average=None, zero_division=0
    )
    
    # Create dictionaries with sentiment names
    precision_dict = {SENTIMENT_MAP[i]: float(p) * 100 for i, p in enumerate(precision)}
    recall_dict = {SENTIMENT_MAP[i]: float(r) * 100 for i, r in enumerate(recall)}
    f1_dict = {SENTIMENT_MAP[i]: float(f) * 100 for i, f in enumerate(f1)}
    
    return {
        'total': len(df),
        'accuracy': float(accuracy),
        'precision': precision_dict,
        'recall': recall_dict,
        'f1_score': f1_dict,
        'confusion_matrix': confusion_matrix(y_true, y_pred, labels=list(SENTIMENT_MAP.keys()))
    }

def secure_password_check(input_password: str, stored_password: str) -> bool:
    """
    Securely compare passwords to prevent timing attacks.
    
    Args:
        input_password: Password provided by user
        stored_password: Stored password from secrets
        
    Returns:
        True if passwords match, False otherwise
    """
    import secrets
    return secrets.compare_digest(input_password, stored_password)

# --- 3. UI Components ---
def render_sentiment_card(label: str, color: str) -> None:
    """Render the sentiment result card with proper styling."""
    sentiment_html = f"""
        <div style="
            background-color: #f8f9fb; 
            padding: 30px; 
            border-radius: 15px; 
            border-left: 15px solid {color};
            text-align: center;
            box-shadow: 2px 2px 10px rgba(0,0,0,0.05);
            margin-bottom: 10px;">
            <p style="color: #555; font-size: 18px; margin-bottom: 5px; font-weight: bold;">The AI detected:</p>
            <h1 style="color: {color}; margin: 0; font-size: 65px; text-transform: uppercase;">
                {label}
            </h1>
        </div>
    """
    st.markdown(sentiment_html, unsafe_allow_html=True)

def render_user_guide() -> None:
    """Display comprehensive user guide page."""
    st.title("📖 User Guide & Instructions")
    st.write("Thank you for participating in this Kurdish NLP research. Please follow these steps to use the tool effectively.")
    
    st.divider()
    
    # Step 1
    st.subheader("1. Entering Text")
    st.write("""
    * Type or paste your Kurdish sentence into the text area.
    * **Note:** The model is optimized for Kurdish. Using other languages may result in inaccurate predictions.
    * Click the **'Analyze Sentiment'** button.
    """)
    
    # Step 2
    st.subheader("2. Understanding Results")
    st.write("""
    The AI will classify your text into one of seven categories:
    * **Happiness / Sadness / Fear / Anger**: Standard emotional states.
    * **Surprise**: For unexpected events.
    * **Disgust**: For expressing strong dislike.
    * **Sarcastic**: For sentences where the meaning is the opposite of the words used.
    """)
    
    # Step 3
    st.subheader("3. Providing Feedback (Crucial for Research)")
    st.write("""
    If you feel the AI made a mistake:
    1. Scroll down to **'Report an incorrect prediction'**.
    2. Select what you believe is the **correct** sentiment from the dropdown.
    3. Read the **Privacy Notice** and check the **Consent Box**.
    4. Click **'Submit Feedback'**.
    """)
    
    st.info("💡 Your feedback directly helps retrain the model to better understand Kurdish linguistic nuances.")
    
    # Step 4
    st.subheader("4. Technical FAQ")
    with st.expander("Is my data saved?"):
        st.write("Only if you click 'Submit Feedback'. Regular analysis is not permanently logged.")
    
    with st.expander("Why did it get my sentence wrong?"):
        st.write("Sentiment analysis is complex, especially in Kurdish due to various dialects. Your corrections help the AI learn these differences!")
    
    st.success("Ready to start? Switch to 'Sentiment Analyzer' in the sidebar!")

# --- 4. Initialize Application ---
initialize_session_state()

# Load models with error handling
try:
    model, tfidf = load_assets()
except (FileNotFoundError, Exception) as e:
    st.error(f"❌ **Critical Error**: {str(e)}")
    st.error("Please ensure the model files (sentiment_svc_model.pkl and tfidf_vectorizer.pkl) are in the same directory as this application.")
    st.stop()

# Initialize Supabase connection with error handling
try:
    conn = st.connection("supabase", type=SupabaseConnection)
    db_connected = True
except Exception as e:
    logger.warning(f"Supabase connection failed: {str(e)}")
    st.warning("⚠️ Database connection unavailable. Feedback features will be disabled.")
    conn = None
    db_connected = False

# --- 5. Sidebar Navigation ---
st.sidebar.title("📌 Menu")
app_mode = st.sidebar.radio("Go to:", ["Sentiment Analyzer", "User Guide"])

# --- 6. Main Application Logic ---
if app_mode == "User Guide":
    render_user_guide()
else:
    # --- Sentiment Analyzer Page ---
    st.title("Kurdish Sentiment Analysis")
    st.write("By Fanar Rofoo | PhD Research Project. Supervised by: Prof. Dr. Shareef M. Shareef, Dr. Polla Fattah")
    st.write(" ")
    
    with st.expander("📖 About this App"):
        st.markdown("""
        This app is a part of a PhD research project, in which a **Linear Support Vector Classifier (LinearSVC)** 
        is used after testing many other classifiers. Kurdish is a low-resource language, and this project aims 
        to improve AI understanding of its dialects and sentiments.
        """)
    
    with st.expander("🔐 Data Privacy"):
        st.write("By providing feedback, you consent to the storage of anonymized text for academic purposes.")
    
    # Input Area with character count
    user_input = st.text_area(
        "Enter a Kurdish sentence:",
        placeholder="Kurdish text only, preferred Kurdish unicode",
        height=150,
        max_chars=MAX_TEXT_LENGTH,
        key="text_input"
    )
    
    # Show character count
    if user_input:
        char_count = len(user_input)
        st.caption(f"Character count: {char_count}/{MAX_TEXT_LENGTH}")
    
    # Analyze button
    if st.button("Analyze Sentiment", type="primary", use_container_width=True):
        is_valid, error_msg = validate_text_input(user_input)
        
        if not is_valid:
            st.warning(f"⚠️ {error_msg}")
        else:
            with st.spinner("🔍 Analyzing sentiment..."):
                pred_id, label, confidence, probabilities = safe_predict(user_input, model, tfidf)
                
                if pred_id is not None:
                    # Update session state
                    st.session_state.prediction = pred_id
                    st.session_state.label = label
                    st.session_state.confidence = confidence
                    st.session_state.current_color = COLOR_MAP.get(pred_id, "#f0f2f6")
                    st.session_state.user_input = user_input
                    st.session_state.probabilities = probabilities
                    st.rerun()
                else:
                    st.error("❌ Prediction failed. Please try again or contact support if the issue persists.")
    
    # --- 7. Result Display ---
    if st.session_state.prediction is not None:
        st.divider()
        st.markdown("### 🔍 Analysis Result")
        
        render_sentiment_card(
            st.session_state.label,
            st.session_state.current_color
        )
        
        # Confidence Meter
        st.write(f"**Model Confidence:** {st.session_state.confidence:.1f}%")
        st.progress(st.session_state.confidence / 100)
        
        # Show all class probabilities (optional, can be collapsed)
        if st.session_state.probabilities is not None:
            with st.expander("📊 View all sentiment probabilities"):
                prob_df = pd.DataFrame({
                    'Sentiment': [SENTIMENT_MAP[i] for i in range(len(SENTIMENT_MAP))],
                    'Probability (%)': (st.session_state.probabilities * 100).round(2)
                }).sort_values('Probability (%)', ascending=False)
                st.dataframe(prob_df, use_container_width=True, hide_index=True)
        
        st.write(f"**Classification Metadata:** Model ID {st.session_state.prediction}")
        st.success("✅ Analysis completed successfully!")
        
        # --- 8. Feedback Section ---
        st.divider()
        with st.expander("🛠️ Report an incorrect prediction"):
            correct_label = st.selectbox(
                "What is the correct sentiment?",
                options=list(SENTIMENT_MAP.keys()),
                format_func=lambda x: f"{x} - {SENTIMENT_MAP[x]}"
            )
            consent = st.checkbox("I consent to the anonymized storage of this text.")
            
            if consent:
                if st.button("Submit Feedback", type="primary"):
                    if not db_connected:
                        st.error("⚠️ Cannot submit feedback: Database connection unavailable.")
                    else:
                        try:
                            feedback_data = {
                                "user_input": st.session_state.user_input,
                                "model_prediction": int(st.session_state.prediction),
                                "correct_label": int(correct_label)
                            }
                            conn.table("sentiment_feedback").insert(feedback_data).execute()
                            st.success("✅ Thank you! Your feedback has been logged.")
                            logger.info(f"Feedback submitted: prediction={st.session_state.prediction}, correct={correct_label}")
                        except Exception as e:
                            st.error(f"❌ Error submitting feedback: {str(e)}")
                            logger.error(f"Feedback submission error: {str(e)}")
            else:
                st.button("Submit Feedback", disabled=True, help="Please check the consent box first")
    
    # --- 9. Admin Dashboard ---
    st.divider()
    with st.expander("🔐 Admin Access"):
        pwd = st.text_input("Admin Password", type="password", key="admin_pwd")
        
        if pwd:
            try:
                if "ADMIN_PASSWORD" not in st.secrets:
                    st.error("❌ Admin password not configured in secrets.")
                elif secure_password_check(pwd, st.secrets["ADMIN_PASSWORD"]):
                    if not db_connected:
                        st.error("⚠️ Cannot access admin dashboard: Database connection unavailable.")
                    else:
                        try:
                            with st.spinner("Loading admin data..."):
                                res = conn.table("sentiment_feedback").select("*").execute()
                                
                                if res.data:
                                    df = pd.DataFrame(res.data)
                                    
                                    # Calculate comprehensive metrics
                                    metrics = calculate_classification_metrics(df)
                                    
                                    # Display metrics
                                    st.subheader("📈 Performance Overview")
                                    c1, c2, c3 = st.columns(3)
                                    c1.metric("Total Submissions", metrics['total'])
                                    c2.metric("Overall Accuracy", f"{metrics['accuracy']:.2f}%")
                                    
                                    # Per-class metrics
                                    st.subheader("📊 Per-Class Performance")
                                    metrics_df = pd.DataFrame({
                                        'Sentiment': list(metrics['precision'].keys()),
                                        'Precision (%)': list(metrics['precision'].values()),
                                        'Recall (%)': list(metrics['recall'].values()),
                                        'F1-Score (%)': list(metrics['f1_score'].values())
                                    })
                                    st.dataframe(metrics_df, use_container_width=True, hide_index=True)
                                    
                                    # Confusion Matrix
                                    st.subheader("🧠 Confusion Matrix")
                                    cm_df = pd.DataFrame(
                                        metrics['confusion_matrix'],
                                        index=[SENTIMENT_MAP[i] for i in SENTIMENT_MAP.keys()],
                                        columns=[SENTIMENT_MAP[i] for i in SENTIMENT_MAP.keys()]
                                    )
                                    fig = px.imshow(
                                        cm_df,
                                        text_auto=True,
                                        title="Confusion Matrix",
                                        labels=dict(x="Actual", y="Predicted", color="Count"),
                                        aspect="auto"
                                    )
                                    st.plotly_chart(fig, use_container_width=True)
                                    
                                    # Raw data
                                    st.subheader("📋 Raw Feedback Data")
                                    st.dataframe(df, use_container_width=True)
                                else:
                                    st.info("Waiting for feedback data...")
                        except Exception as e:
                            st.error(f"❌ Fetch Error: {str(e)}")
                            logger.error(f"Admin dashboard error: {str(e)}")
                else:
                    st.warning("⚠️ Incorrect password.")
            except KeyError:
                st.error("❌ Admin password not configured in secrets.")
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
                logger.error(f"Admin authentication error: {str(e)}")
