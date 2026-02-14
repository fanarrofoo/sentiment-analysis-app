import streamlit as st
import pyodbc

st.title("🔌 Connection Debugger")

# 1. Get credentials safely
try:
    # Adjust these keys to match your actual secrets structure
    server = "140.82.39.222"
    port = "1744" 
    database = "TestDB"
    username = st.secrets["connections"]["sqlserver"]["username"]
    password = st.secrets["connections"]["sqlserver"]["password"]
except Exception as e:
    st.error(f"Secrets Error: {e}")
    st.stop()

# 2. Build the RAW connection string (The 'comma' method)
conn_str = (
    f"DRIVER={{ODBC Driver 17 for SQL Server}};"
    f"SERVER={server},{port};"  # Note the comma!
    f"DATABASE={database};"
    f"UID={username};"
    f"PWD={password};"
    "TIMEOUT=10;"
)

st.code(conn_str.replace(password, "******"), language="text")

# 3. Attempt Raw Connection
if st.button("Test Raw Connection"):
    try:
        conn = pyodbc.connect(conn_str, timeout=10)
        st.success("✅ SUCCESS! Connected to SQL Server.")
        conn.close()
    except pyodbc.Error as ex:
        sqlstate = ex.args[0] if ex.args else "Unknown"
        st.error(f"❌ FAILED. State: {sqlstate}")
        st.error(ex)
        
        if "HYT00" in str(ex):
            st.warning("👉 This confirms a FIREWALL issue. The server is dropping the packets from Streamlit Cloud.")
