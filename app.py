import streamlit as st
import joblib
import os
import pandas as pd
from st_supabase_connection import SupabaseConnection
import plotly.express as px
import numpy as np

# --- 1. App Configuration ---
st.set_page_config(page_title="Sentiment Analyser", page_icon="🪄", layout="wide")

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
# --- Initialize Session State Variables ---
# This ensures the variables exist so line 37 doesn't crash
if 'prediction' not in st.session_state:
    st.session_state.prediction = None
if 'label' not in st.session_state:
    st.session_state.label = None
sentiment_map = {
    0: "Sadness", 1: "Happiness", 2: "Fear", 
    3: "Anger", 4: "Disgust", 5: "Surprise", 6: "Sarcastic"
}

conn = st.connection("supabase", type=SupabaseConnection)

# --- 3. Sidebar Navigation ---
st.sidebar.title("📌 Menu")
app_mode = st.sidebar.radio("Go to:", ["Sentiment Analyzer", "User Guide"])

# --- 4. Display Result (High Visibility + Confidence) ---
# Only run this if we have a prediction in state
if st.session_state.get('prediction') is not None:
    st.divider()

    # 1. Calculate Confidence Scores
    decision_scores = model.decision_function(vec)[0]
    exp_scores = np.exp(decision_scores - np.max(decision_scores))
    probabilities = exp_scores / exp_scores.sum()
    confidence_pct = probabilities[st.session_state.prediction] * 100

    # 2. Define colors
    color_map = {
        0: "#1f77b4", 1: "#2ca02c", 2: "#9467bd", 
        3: "#d62728", 4: "#8c564b", 5: "#ff7f0e", 6: "#e377c2"
    }
    sentiment_color = color_map.get(st.session_state.prediction, "#f0f2f6")

    # 3. Display the Visual Card
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
                {st.session_state.label}
            </h1>
        </div>
    """
    st.markdown(sentiment_html, unsafe_allow_html=True)
    
    # 4. The Confidence Meter
    st.write(f"**Model Confidence:** {confidence_pct:.1f}%")
    st.progress(confidence_pct / 100)
    st.success("✅ Analysis completed successfully!")
# --- 5. Main App Logic ---
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

    user_input = st.text_area("Enter a sentence to analyse:", placeholder="Kurdish text only")

    if 'prediction' not in st.session_state:
        st.session_state.prediction = None
    if 'label' not in st.session_state:
        st.session_state.label = None

    if st.button("Analyze Sentiment"):
        if user_input.strip() == "":
            st.warning("Please enter some text first.")
        else:
            vec = tfidf.transform([user_input])
            st.session_state.prediction = model.predict(vec)[0]
            st.session_state.label = sentiment_map.get(st.session_state.prediction, "Unknown")

    # --- 4. Display Result (High Visibility) ---
if st.session_state.prediction is not None:
    st.divider()
    
    # 1. Create a "Card" effect using a success box
    with st.container():
        st.markdown("### 🔍 Analysis Result")
        
        # This custom HTML makes the sentiment name very large and centered
        sentiment_html = f"""
            <div style="
                background-color: #f0f2f6; 
                padding: 20px; 
                border-radius: 10px; 
                border-left: 10px solid #00c0f2;
                text-align: center;
                margin-bottom: 20px;">
                <p style="color: #31333F; font-size: 20px; margin: 0;">The AI detected:</p>
                <h1 style="color: #ff4b4b; margin: 0; font-size: 60px;">{st.session_state.label}</h1>
            </div>
        """
        st.markdown(sentiment_html, unsafe_allow_html=True)
        
        # Keep the formal class ID for your research records
        st.write(f"**Classification Metadata:** Model ID {st.session_state.prediction}")
        st.success("✅ Analysis completed successfully!")
        # --- Feedback Section ---
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
                    try:
                        feedback_data = {
                            "user_input": user_input,
                            "model_prediction": int(st.session_state.prediction),
                            "correct_label": int(correct_label)
                        }
                        conn.table("sentiment_feedback").insert(feedback_data).execute()
                        st.success("✅ Thank you! Feedback logged.")
                    except Exception as e:
                        st.error(f"Error: {e}")
            else:
                st.button("Submit Feedback", disabled=True, help="Check consent box first")

    # --- 6. Admin Dashboard ---
    st.divider()
    with st.expander("🔐 Admin Access"):
        password = st.text_input("Enter Admin Password", type="password")
        if password == st.secrets["ADMIN_PASSWORD"]:
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
                    st.info("No data yet.")
            except Exception as e:
                st.error(f"Fetch error: {e}")
