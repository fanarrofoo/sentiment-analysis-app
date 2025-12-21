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
# --- Research Methodology Section ---
with st.expander("📖 About this App"):
    st.markdown("""
    ### Methodology & Background
    This tool is part of a **PhD research project** focused on enhancing **Kurdish Natural Language Processing (NLP)**. 
    Kurdish is a "low-resource" language in the AI world, meaning there is less digital data available compared to English. 
    This research aims to bridge that gap.

    **How the Model Works:**
    1.  **Preprocessing:** Your text is cleaned and converted into numerical data using **TF-IDF Vectorization**.
    2.  **Classification:** We utilize a **Linear Support Vector Classifier (LinearSVC)**, which is highly effective for text classification tasks.
    3.  **Accuracy:** During the validation phase, this model achieved an accuracy of **86%** across 7 sentiment categories.

    **The 7 Sentiments Analyzed:**
    * **0-3:** Sadness, Happiness, Fear, Anger
    * **4-6:** Disgust, Surprise, Sarcastic

    **Your Role:**
    By providing feedback on incorrect predictions, you are helping to refine the dataset. This "Human-in-the-loop" approach is vital for capturing the linguistic nuances of Kurdish dialects and sarcasm.
    """)
    
# --- Data Privacy Section ---
with st.expander("🔐 Data Privacy & Ethics"):
    st.markdown("""
    ### Your Privacy Matters
    In accordance with academic research ethics and data protection principles (like GDPR), we are committed to protecting your privacy.
    **1. What data is collected?**
    We only collect the **Kurdish text** you provide and the **sentiment labels** (both AI-predicted and user-corrected). 
    **2. Is my identity tracked?**
    **No.** We do not collect names, email addresses, IP addresses, or any other personal identifiers. The data is completely **anonymized**.
    **3. How is the data used?**
    The logs are used exclusively for:
    * Evaluating the performance of the LinearSVC model.
    * Identifying linguistic patterns for PhD thesis analysis.
    * Improving the dataset for future Kurdish NLP research.
    **4. Data Security**
    Your feedback is stored securely in a private Supabase database and will not be sold or shared with third-party advertisers.
    **By using the "Submit Feedback" button, you consent to the storage of the entered text for the research purposes mentioned above.**
    """)

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
    
   if st.session_state.prediction >= 0: 
        st.success("✅ Analysis completed successfully!")

    # --- 5. Feedback Section ---
    st.divider()
    st.subheader("🛠️ Help Improve the AI")
    
    with st.expander("Report an incorrect prediction"):
        st.markdown("""
        **ID | Sentiment**
        0 | Sadness  
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

        # Consent mechanism for PhD Ethics
        st.info("💡 Your feedback helps improve Kurdish NLP research.")
        consent_given = st.checkbox("I consent to the anonymized storage of this text for research purposes.")

        # Submit button logic
        if consent_given:
            if st.button("Submit Feedback", type="primary"):
                try:
                    feedback_data = {
                        "user_input": user_input,
                        "model_prediction": int(st.session_state.prediction),
                        "correct_label": int(correct_label)
                    }
                    # Insert into Supabase
                    conn.table("sentiment_feedback").insert(feedback_data).execute()
                    st.success("✅ Thank you! Your feedback has been safely logged.")
                except Exception as e:
                    st.error(f"Failed to save feedback: {e}")
        else:
            st.button("Submit Feedback", help="Please check the consent box first", disabled=True)
# --- 6. Admin Dashboard (Hidden) ---
st.divider()
with st.expander("🔐 Admin Access"):
    password = st.text_input("Enter Admin Password", type="password")
    
    if password == st.secrets["ADMIN_PASSWORD"]:
        st.success("Access Granted")
        
# --- PART A: FETCH & VISUALIZE DATA ---
        try:
            response = conn.table("sentiment_feedback").select("*").execute()
            
            if response.data:
                df_admin = pd.DataFrame(response.data)
                
                # 1. Calculate Live Accuracy
                total_feedback = len(df_admin)
                # Count where prediction matches the user's correct label
                correct_matches = len(df_admin[df_admin['model_prediction'] == df_admin['correct_label']])
                live_accuracy = (correct_matches / total_feedback) * 100 if total_feedback > 0 else 0
                
                # 2. Display Metrics in Columns
                st.subheader("📈 Performance Overview")
                col1, col2, col3 = st.columns(3)
                col1.metric("Total Feedback", f"{total_feedback}")
                col2.metric("Live Accuracy", f"{live_accuracy:.1f}%")
                col3.metric("Training Accuracy", "86.0%") # Your fixed PhD benchmark

                # 3. Process names for charts
                df_admin['Predicted Name'] = df_admin['model_prediction'].map(sentiment_map)
                df_admin['Correct Name'] = df_admin['correct_label'].map(sentiment_map)

                # --- VISUALIZATION: Error Distribution ---
                st.subheader("📊 Error Distribution")
                chart_data = df_admin['Correct Name'].value_counts().reset_index()
                chart_data.columns = ['Sentiment', 'Count']
                st.plotly_chart(px.bar(chart_data, x='Sentiment', y='Count', color='Sentiment', template="plotly_white"), use_container_width=True)

                # --- VISUALIZATION: Confusion Matrix ---
                st.subheader("🧠 Model Confusion Matrix")
                st.write("This heatmap shows exactly which labels are being swapped.")
                
                # 
                
                confusion_matrix = pd.crosstab(
                    df_admin['Predicted Name'], 
                    df_admin['Correct Name'],
                    dropna=False
                )
                
                fig_heat = px.imshow(
                    confusion_matrix, 
                    text_auto=True, 
                    color_continuous_scale='RdBu_r',
                    labels=dict(x="User Said (Actual)", y="AI Said (Predicted)")
                )
                st.plotly_chart(fig_heat, use_container_width=True)

                # --- DATA TABLE & DOWNLOAD ---
                st.subheader("📋 Raw Feedback Logs")
                st.dataframe(df_admin, use_container_width=True)
                
                csv = df_admin.to_csv(index=False).encode('utf-8')
                st.download_button("📥 Download Feedback as CSV", data=csv, file_name="kurdish_sentiment_feedback.csv", mime="text/csv")
            
            else:
                st.info("No feedback entries found yet.")
                
        except Exception as e:
            st.error(f"Error fetching data: {e}")

        # --- PART B: DANGER ZONE (Separate from the fetch try/except) ---
        st.markdown("---")
        st.subheader("⚠️ Danger Zone")
        confirm_delete = st.checkbox("I want to permanently delete all feedback records.")
        
        if st.button("Delete All Logs", type="primary", disabled=not confirm_delete):
            try:
                # We use a filter that is always true to delete all rows
                conn.table("sentiment_feedback").delete().neq("id", 0).execute()
                st.success("💥 All records deleted!")
                st.rerun() 
            except Exception as e:
                st.error(f"Deletion failed: {e}")
            
    elif password:
        st.error("Incorrect password")
