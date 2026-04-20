import streamlit as st
import uuid
import os

# --- CRITICAL FIX FOR STREAMLIT CLOUD CHROMADB ---
try:
    __import__('pysqlite3')
    import sys
    sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
except ImportError:
    pass
# -------------------------------------------------

# Securely grab the API Key so it NEVER crashes
try:
    groq_key = st.secrets["GROQ_API_KEY"]
except Exception:
    groq_key = os.environ.get("GROQ_API_KEY", "gsk_placeholder_key_to_prevent_crash")

os.environ["GROQ_API_KEY"] = groq_key

@st.cache_resource
def init_agent():
    import agent
    return agent

agent_module = init_agent()

st.title("MediCare Hospital Assistant")

with st.sidebar:
    st.header("About")
    st.write("Domain: Hospital Patient Assistant")
    st.write("User: Patients and hospital callers")
    st.write("Capabilities: Answers questions about OPD timings, doctors, fees, insurance, and appointments using a ChromaDB Knowledge Base.")
    st.write("Tool used: Datetime tool to fetch current date/time.")
    
    if st.button("New Conversation"):
        st.session_state.thread_id = str(uuid.uuid4())
        st.session_state.messages = []
        st.rerun()

if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

user_input = st.chat_input("Ask about MediCare Hospital...")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)
        
    with st.spinner("Thinking..."):
        try:
            result = agent_module.ask(user_input, st.session_state.thread_id)
            answer = result["answer"]
        except Exception as e:
            if "gsk_placeholder" in os.environ.get("GROQ_API_KEY", ""):
                answer = "⚠️ **Action Required**: Please add your Groq API Key to the Streamlit Secrets (Settings -> Secrets) in the bottom right."
            else:
                answer = f"Error: {str(e)}"
            
    st.session_state.messages.append({"role": "assistant", "content": answer})
    with st.chat_message("assistant"):
        st.markdown(answer)
