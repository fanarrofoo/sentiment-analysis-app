import streamlit as st
import joblib
import os
import pandas as pd
from st_supabase_connection import SupabaseConnection
import plotly.express as px
import numpy as np

# --- 1. App Configuration ---
st.set_page_config(page_title="Sentiment Analyser", page_icon="🪄", layout="wide")

# --- 2. Constants and Configuration ---
sentiment_map = {
    0: "Sadness", 1: "Happiness", 2: "Fear", 
    3: "Anger", 4: "Disgust", 5: "Surprise", 6: "Sarcastic"
}

color_map = {
    0: "#1f77b4", 1: "#2ca02c", 2: "#9467bd", 
    3: "#d62728", 4: "#8c564b", 5: "#ff7f0e", 6: "#e377c2"
}

# --- 3. Initialize Session State Variables ---
if 'prediction' not in st.session_state:
    st.session_state.prediction = None
if 'label' not in st.session_state:
    st.session_state.label = None
if 'user_input' not in st.session_state:
    st.session_state.user_input = ""
if 'vec' not in st.session_state:
    st.session_state.vec = None

# --- 4. Load Model & Assets ---
current_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(current_dir, 'sentiment_svc_model.pkl')
tfidf_path = os.path.join(current_dir, 'tfidf_vectorizer.pkl')

@st.cache_resource
def load_assets():
    """Load model and vectorizer with error handling."""
    try:
        if not os.path.exists(model_path):
            st.error(f"❌ Model file not found: {model_path}")
            st.stop()
        if not os.path.exists(tfidf_path):
            st.error(f"❌ Vectorizer file not found: {tfidf_path}")
            st.stop()
        
        model = joblib.load(model_path)
        tfidf = joblib.load(tfidf_path)
        return model, tfidf
    except Exception as e:
        st.error(f"❌ Error loading model assets: {str(e)}")
        st.stop()

try:
    model, tfidf = load_assets()
except Exception as e:
    st.error(f"❌ Failed to load models. Please check that model files exist.")
    st.stop()

# --- 5. Initialize Supabase Connection ---
try:
    conn = st.connection("supabase", type=SupabaseConnection)
except Exception as e:
    st.warning(f"⚠️ Could not connect to Supabase: {str(e)}. Feedback features may not work.")
    conn = None

# --- 6. Helper Functions ---
def render_user_guide():
    """Display user guide page."""
    st.title("📖 User Guide")
    st.markdown("---")
    
    st.header("How to Use the Sentiment Analyzer")
    st.markdown("""
    ### Step 1: Enter Text
    Type or paste your Kurdish text into the text area on the main page.
    
    ### Step 2: Analyze
    Click the "Analyze Sentiment" button to get the sentiment prediction.
    
    ### Step 3: Review Results
    The app will display:
    - **Sentiment Label**: The detected emotion (Sadness, Happiness, Fear, Anger, Disgust, Surprise, or Sarcastic)
    - **Confidence Score**: How confident the model is in its prediction
    - **Model ID**: The numeric classification ID for research purposes
    
    ### Step 4: Provide Feedback (Optional)
    If the prediction is incorrect, you can help improve the model by:
    1. Expanding the "Help Improve the AI" section
    2. Selecting the correct sentiment label
    3. Providing consent for anonymized storage
    4. Submitting your feedback
    """)
    
    st.header("Understanding Sentiment Labels")
    sentiment_info = pd.DataFrame([
        {"ID": 0, "Sentiment": "Sadness", "Description": "Text expressing sadness or melancholy"},
        {"ID": 1, "Sentiment": "Happiness", "Description": "Text expressing joy or positive emotions"},
        {"ID": 2, "Sentiment": "Fear", "Description": "Text expressing fear or anxiety"},
        {"ID": 3, "Sentiment": "Anger", "Description": "Text expressing anger or frustration"},
        {"ID": 4, "Sentiment": "Disgust", "Description": "Text expressing disgust or revulsion"},
        {"ID": 5, "Sentiment": "Surprise", "Description": "Text expressing surprise or shock"},
        {"ID": 6, "Sentiment": "Sarcastic", "Description": "Text with sarcastic or ironic tone"},
    ])
    st.dataframe(sentiment_info, use_container_width=True, hide_index=True)
    
    st.header("Model Information")
    st.markdown("""
    - **Model Type**: LinearSVC (Support Vector Classifier)
    - **Accuracy**: 86%
    - **Language**: Kurdish NLP
    - **Purpose**: PhD Research
    
    This model has been trained on Kurdish text and can classify emotions across 7 categories.
    """)

def calculate_confidence_scores(vec, prediction):
    """Calculate confidence scores from decision function."""
    try:
        decision_scores = model.decision_function(vec)[0]
        exp_scores = np.exp(decision_scores - np.max(decision_scores))
        probabilities = exp_scores / exp_scores.sum()
        confidence_pct = probabilities[prediction] * 100
        return confidence_pct
    except Exception as e:
        st.warning(f"Could not calculate confidence: {str(e)}")
        return None

def display_result(prediction, label, vec=None):
    """Display sentiment analysis result with confidence."""
    st.divider()
    
    # Get color for this sentiment
    sentiment_color = color_map.get(prediction, "#f0f2f6")
    
    # Calculate confidence if vector is available
    confidence_pct = None
    if vec is not None:
        confidence_pct = calculate_confidence_scores(vec, prediction)
    
    # Display the Visual Card
    sentiment_html = f"""
        <div style="
            background-color: #f8f9fb; 
            padding: 30px; 
            border-radius: 15px; 
            border-left: 15px solid {sentiment_color};
            text-align: center;
            box-shadow: 2px 2px 10px rgba(0,0,0,0.05);
            margin-bottom: 5px;">
            <p style="color: #555; font-size: 18px; margin-bottom: 5px; font-weight: bold;">The AI detected:</p>
            <h1 style="color: {sentiment_color}; margin: 0; font-size: 65px; text-transform: uppercase;">
                {label}
            </h1>
        </div>
    """
    st.markdown(sentiment_html, unsafe_allow_html=True)
    
    # Display confidence if available
    if confidence_pct is not None:
        st.write(f"**Model Confidence:** {confidence_pct:.1f}%")
        st.progress(confidence_pct / 100)
    
    # Keep the formal class ID for research records
    st.write(f"**Classification Metadata:** Model ID {prediction}")
    st.success("✅ Analysis completed successfully!")

def display_feedback_section(user_input):
    """Display feedback section for incorrect predictions."""
    st.divider()
    with st.expander("🛠️ Help Improve the AI (Report Incorrect Prediction)"):
        correct_label = st.selectbox(
            "What is the correct sentiment?", 
            options=list(sentiment_map.keys()),
            format_func=lambda x: f"{x} - {sentiment_map[x]}"
        )
        consent_given = st.checkbox("I consent to the anonymized storage of this text.")
        
        if consent_given:
            if st.button("Submit Feedback", type="primary"):
                if conn is None:
                    st.error("⚠️ Cannot submit feedback: Supabase connection unavailable.")
                else:
                    try:
                        feedback_data = {
                            "user_input": user_input,
                            "model_prediction": int(st.session_state.prediction),
                            "correct_label": int(correct_label)
                        }
                        conn.table("sentiment_feedback").insert(feedback_data).execute()
                        st.success("✅ Thank you! Feedback logged.")
                    except Exception as e:
                        st.error(f"❌ Error submitting feedback: {str(e)}")
        else:
            st.button("Submit Feedback", disabled=True, help="Check consent box first")

def display_admin_dashboard():
    """Display admin dashboard with performance metrics."""
    st.divider()
    with st.expander("🔐 Admin Access"):
        password = st.text_input("Enter Admin Password", type="password")
        if password and "ADMIN_PASSWORD" in st.secrets and password == st.secrets["ADMIN_PASSWORD"]:
            if conn is None:
                st.error("⚠️ Cannot access admin dashboard: Supabase connection unavailable.")
                return
                
            try:
                response = conn.table("sentiment_feedback").select("*").execute()
                if response.data:
                    df_admin = pd.DataFrame(response.data)
                    
                    # Performance Metrics
                    total_feedback = len(df_admin)
                    correct_matches = len(df_admin[df_admin['model_prediction'] == df_admin['correct_label']])
                    live_acc = (correct_matches / total_feedback) * 100 if total_feedback > 0 else 0
                    
                    st.subheader("📈 Performance Overview")
                    c1, c2 = st.columns(2)
                    c1.metric("Total Feedback", total_feedback)
                    c2.metric("Live Accuracy", f"{live_acc:.1f}%")
                    
                    # Confusion Matrix
                    st.subheader("🧠 Model Confusion Matrix")
                    df_admin['Predicted'] = df_admin['model_prediction'].map(sentiment_map)
                    df_admin['Actual'] = df_admin['correct_label'].map(sentiment_map)
                    conf_matrix = pd.crosstab(df_admin['Predicted'], df_admin['Actual'])
                    st.plotly_chart(px.imshow(conf_matrix, text_auto=True), use_container_width=True)
                    
                    st.dataframe(df_admin)
                else:
                    st.info("No feedback data yet.")
            except Exception as e:
                st.error(f"❌ Fetch error: {str(e)}")
        elif password:
            st.warning("⚠️ Incorrect password.")

# --- 7. Sidebar Navigation ---
st.sidebar.title("📌 Menu")
app_mode = st.sidebar.radio("Go to:", ["Sentiment Analyzer", "User Guide"])

# --- 8. Main App Logic ---
if app_mode == "User Guide":
    render_user_guide()
else:
    # --- Sentiment Analyzer Page ---
    st.title("Kurdish Sentiment Analysis")
    st.write("By Fanar Rofoo")
    
    with st.expander("📖 About this App"):
        st.write("This app is a part of PhD research, that employs a LinearSVC model (86% accuracy) to analyze Kurdish NLP.")
    
    with st.expander("🔐 Data Privacy & Ethics"):
        st.write("Data is anonymized. We only log the text and labels you provide for model improvement.")
    
    user_input = st.text_area(
        "Enter a sentence to analyse:", 
        placeholder="Kurdish text only",
        value=st.session_state.user_input,
        key="text_input"
    )
    
    if st.button("Analyze Sentiment"):
        if not user_input or user_input.strip() == "":
            st.warning("⚠️ Please enter some text first.")
        else:
            try:
                # Store user input in session state
                st.session_state.user_input = user_input
                
                # Transform and predict
                vec = tfidf.transform([user_input])
                prediction = model.predict(vec)[0]
                label = sentiment_map.get(prediction, "Unknown")
                
                # Store results in session state
                st.session_state.prediction = prediction
                st.session_state.label = label
                st.session_state.vec = vec
                
                # Force rerun to show results
                st.rerun()
            except Exception as e:
                st.error(f"❌ Error during analysis: {str(e)}")
    
    # --- Display Result ---
    if st.session_state.prediction is not None:
        display_result(
            st.session_state.prediction, 
            st.session_state.label,
            st.session_state.vec
        )
        display_feedback_section(st.session_state.user_input)
    
    # --- Admin Dashboard ---
    display_admin_dashboard()
