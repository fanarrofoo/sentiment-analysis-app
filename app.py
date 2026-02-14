import streamlit as st
import pyodbc

st.title("🔌 Final Connection Test")

# --- 1. Load Credentials ---
try:
    # We look for the keys we just added to Secrets
    server = "140.82.39.222"
    port = "1744"
    database = "TestDB"
    username = st.secrets["DB_USER"]
    password = st.secrets["DB_PASS"]
except Exception as e:
    st.error(f"❌ Configuration Error: Could not find 'DB_USER' or 'DB_PASS' in secrets. {e}")
    st.stop()

# --- 2. Build Raw Connection String ---
# We force the driver and port syntax manually
conn_str = (
    f"DRIVER={{ODBC Driver 17 for SQL Server}};"
    f"SERVER={server},{port};"  # The comma syntax is vital for Linux
    f"DATABASE={database};"
    f"UID={username};"
    f"PWD={password};"
    "TIMEOUT=5;"
)

st.info(f"Testing connection to: `{server},{port}`")

# --- 3. Attempt Connection ---
if st.button("🚀 Test Connection Now"):
    try:
        conn = pyodbc.connect(conn_str)
        st.success("✅ SUCCESS! The server is reachable.")
        st.write("This confirms the Firewall is OPEN and credentials are CORRECT.")
        conn.close()
    except pyodbc.Error as ex:
        sqlstate = ex.args[0] if ex.args else "Unknown"
        st.error(f"❌ FAILED. Error State: {sqlstate}")
        st.code(str(ex), language="bash")
        
        if "HYT00" in str(ex):
            st.error("🚨 DIAGNOSIS: TIMEOUT.")
            st.markdown("""
            **What this means:** The request was sent, but the server ignored it.
            
            **The Only Fix:**
            You must log into the Windows Server at `140.82.39.222` and add an **Inbound Rule** to the **Windows Firewall** allowing traffic on Port **1744** from **Any IP**.
            """)
        elif "28000" in str(ex) or "18456" in str(ex):
            st.warning("🔑 DIAGNOSIS: BAD PASSWORD. The connection worked, but login failed.")
