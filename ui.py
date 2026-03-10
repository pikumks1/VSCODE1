import streamlit as st
from src.search import RAGSearch  # Import your class

# 1. Setup the page
st.set_page_config(page_title="My RAG Bot")
st.title("💬 My RAG Assistant")

# 2. Initialize the RAG engine (Use @st.cache_resource to keep it in memory)
@st.cache_resource
def get_rag():
    return RAGSearch()

rag_engine = get_rag()

# 3. Handle chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display previous messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 4. Handle new user input
if prompt := st.chat_input("Ask me anything about your documents!"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = rag_engine.search_and_summarize(prompt)
            st.markdown(response)
            
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})