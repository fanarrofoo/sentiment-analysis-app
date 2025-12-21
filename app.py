import streamlit as st
import joblib
import os
import pandas as pd
from st_supabase_connection import SupabaseConnection
import plotly.express as px

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
st.set_page_config(page_title="Sentiment Analyser", page_icon="🪄")
st.title("Kurdish Sentiment Analysis")
st.write("By Fanar Rofoo")
st.write("This sentiment analysis app is the culmination of PhD research, employing a LinearSVC model that achieves 86% accuracy.")
conn = st.connection("supabase", type=SupabaseConnection)

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
    
#if st.session_state.prediction >=0:
#    st.success("The operation was completed successfully.")
    
 # --- 5. Feedback Section ---
if st.session_state.prediction is not None:
    st.divider()
    st.subheader("🛠️ Help Improve the AI")
    
    with st.expander("Report an incorrect prediction"):
        # (Keep your selectbox logic here...)
        correct_label = st.selectbox(
            "What is the correct sentiment (0-6)?", 
            options=list(sentiment_map.keys()),
            format_func=lambda x: f"{x} - {sentiment_map[x]}"
        )
        
        if st.button("Submit Feedback"):
            # Prepare data for Supabase
            feedback_data = {
                "user_input": user_input,
                "model_prediction": int(st.session_state.prediction),
                "correct_label": int(correct_label)
            }
            
            try:
                # Insert into Supabase table
                conn.table("sentiment_feedback").insert(feedback_data).execute()
                st.success("✅ Thank you! Your feedback has been sent to the developer.")
            except Exception as e:
                st.error(f"Failed to save feedback: {e}")
# --- 6. Admin Dashboard (Hidden) ---
st.divider()
with st.expander("🔐 Admin Access"):
    password = st.text_input("Enter Admin Password", type="password")
    
    if password == st.secrets["ADMIN_PASSWORD"]:
        st.success("Access Granted")
        st.subheader("Recent Feedback Data")
        
        try:
            # Fetch data from Supabase
            response = conn.table("sentiment_feedback").select("*").execute()
            
            if response.data:
                df_admin = pd.DataFrame(response.data)
                # --- Charting Logic ---
    st.subheader("📊 Error Distribution")
    st.write("Which sentiments are users reporting as the 'Correct' label?")

    # Map the IDs to Names for the chart
    df_admin['Label Name'] = df_admin['correct_label'].map(sentiment_map)
    
    # Create a frequency count
    chart_data = df_admin['Label Name'].value_counts().reset_index()
    chart_data.columns = ['Sentiment', 'Count']

    # Create the Plotly Bar Chart
    fig = px.bar(
        chart_data, 
        x='Sentiment', 
        y='Count',
        color='Sentiment',
        labels={'Count': 'Number of Reports', 'Sentiment': 'Correct Category'},
        template="plotly_white"
    )
    
    st.plotly_chart(fig, use_container_width=True)

    # --- Data Table & Download ---
    st.subheader("📋 Raw Feedback Logs")
    st.dataframe(df_admin, use_container_width=True)
    
    csv = df_admin.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Download Feedback as CSV",
        data=csv,
        file_name="kurdish_sentiment_feedback.csv",
        mime="text/csv",
    )
                # Make the dataframe look nicer
                st.dataframe(df_admin, use_container_width=True)
                
                # Add a download button for your PhD research CSV
                csv = df_admin.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 Download Feedback as CSV",
                    data=csv,
                    file_name="kurdish_sentiment_feedback.csv",
                    mime="text/csv",
                )
            else:
                st.info("No feedback entries found yet.")
        
        except Exception as e:
            st.error(f"Error fetching data: {e}")
    elif password:
        st.error("Incorrect password")
