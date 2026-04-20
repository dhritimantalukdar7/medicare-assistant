import streamlit as st
import uuid
import os

# Set dummy API key if not present, but user will need one for Groq
os.environ["GROQ_API_KEY"] = os.environ.get("GROQ_API_KEY") # Handled by Streamlit secrets in production

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
            answer = f"Error: {str(e)}"
            
    st.session_state.messages.append({"role": "assistant", "content": answer})
    with st.chat_message("assistant"):
        st.markdown(answer)
