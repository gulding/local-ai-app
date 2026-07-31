import streamlit as st
from langchain_community.llms import Ollama
import PyPDF2

# 1. UI Configuration
st.set_page_config(page_title="Local AI", page_icon="🦙")
st.title("Privacy-First Local AI 🦙")
st.caption("Running Meta's Llama 3.2 100% offline. Zero data leaves your machine.")

# 2. Sidebar for Secure Document Upload
with st.sidebar:
    st.header("📄 Secure Document Chat")
    st.write("Upload a private PDF. It will be processed entirely on your local RAM.")
    uploaded_file = st.file_uploader("Upload PDF", type=["pdf"])
    
    document_text = ""
    if uploaded_file is not None:
        pdf_reader = PyPDF2.PdfReader(uploaded_file)
        for page in pdf_reader.pages:
            text = page.extract_text()
            if text:
                document_text += text + "\n"
        st.success("Document loaded into local memory!")

# 3. Session State for Chat History
if "messages" not in st.session_state:
    st.session_state.messages = []

# Render previous messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 4. Handle User Input
if prompt := st.chat_input("Ask your local AI anything..."):
    # Display user message
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Display AI response with real-time streaming
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        
        # If a document is uploaded, inject it into the prompt context
        final_prompt = prompt
        if document_text:
            # We limit the text to ~3000 characters so we don't overflow the local model's context window
            final_prompt = f"Use the following document text to answer the question:\n\n{document_text[:3000]}\n\nQuestion: {prompt}"
        
        # Connect to local Ollama instance
        llm = Ollama(model="llama3.2")
        
        # Stream the response chunk by chunk
        for chunk in llm.stream(final_prompt):
            full_response += chunk
            # Add a blinking cursor to simulate typing
            message_placeholder.markdown(full_response + "▌")
        
        # Final output without the cursor
        message_placeholder.markdown(full_response)
        
    # Save to history
    st.session_state.messages.append({"role": "assistant", "content": full_response})