import streamlit as st
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
import PyPDF2

# --- 1. UI Configuration ---
st.set_page_config(page_title="Enterprise Local AI", page_icon="🦙", layout="wide")
st.title("Enterprise Local AI & RAG 🦙")
st.caption("100% Offline | FAISS Vector Search | Dynamic Personas")

# --- 2. Sidebar Controls ---
with st.sidebar:
    st.header("⚙️ Engine Configuration")
    
    # Allow model switching if you download more models (like Mistral)
    selected_model = st.selectbox("Local Model", ["llama3.2", "llama3.1", "mistral", "gemma2"])
    
    # Temperature slider for hallucination control
    temperature = st.slider("Temperature (Creativity vs Fact)", 0.0, 1.0, 0.3, 0.1)
    
    # System Prompt injection
    system_persona = st.selectbox(
        "AI Persona",
        [
            "Standard Assistant",
            "Senior Python Developer",
            "Balkan Legal & Compliance Analyst",
            "Executive Summarizer"
        ]
    )
    
    personas = {
        "Standard Assistant": "You are a highly capable and concise AI assistant.",
        "Senior Python Developer": "You are a senior software engineer. Always provide highly optimized, secure code.",
        "Balkan Legal & Compliance Analyst": "You are a strict legal analyst specialized in regional regulations. Focus on compliance, risks, and exact phrasing.",
        "Executive Summarizer": "You are a corporate executive assistant. Summarize everything into concise bullet points."
    }
    
    st.divider()
    
    # --- 3. The RAG Engine (Vector Database) ---
    st.header("📄 Secure Document Vault")
    st.write("Upload massive PDFs. Processed securely via FAISS.")
    uploaded_file = st.file_uploader("Upload PDF", type=["pdf"])
    
    if uploaded_file:
        # Only build the vector index if a new file is uploaded
        if "loaded_file" not in st.session_state or st.session_state.loaded_file != uploaded_file.name:
            with st.spinner("Chunking & Embedding Document..."):
                pdf_reader = PyPDF2.PdfReader(uploaded_file)
                document_text = "".join([page.extract_text() for page in pdf_reader.pages if page.extract_text()])
                
                text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
                chunks = text_splitter.split_text(document_text)
                
                embeddings = OllamaEmbeddings(model="nomic-embed-text")
                st.session_state.vector_store = FAISS.from_texts(chunks, embedding=embeddings)
                st.session_state.loaded_file = uploaded_file.name
                
            st.success("✅ FAISS Vector Index Created in RAM!")

# --- 4. Chat Interface ---
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Query your local model or document..."):
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        
        # --- 5. Semantic Search Retrieval ---
        # --- 5. Semantic Search Retrieval ---
        context = ""
        if "vector_store" in st.session_state:
            retriever = st.session_state.vector_store.as_retriever(search_kwargs={"k": 3})
            docs = retriever.invoke(prompt)
            context_text = "\n\n".join([d.page_content for d in docs])
            context = f"\n\n--- DOCUMENT CONTEXT ---\n{context_text}"
            
        # 1. Build the exact conversation history
        messages = [SystemMessage(content=personas[system_persona])]
        
        for msg in st.session_state.messages[:-1]: # Load all past messages except the current one
            if msg["role"] == "user":
                messages.append(HumanMessage(content=msg["content"]))
            else:
                messages.append(AIMessage(content=msg["content"]))
                
        # 2. Append the CURRENT question combined with the FAISS context
        messages.append(HumanMessage(content=f"Current Question: {prompt}\n{context}"))
        
        # 3. Connect to Ollama with strict Anti-Loop Safeguards
        llm = ChatOllama(
            model=selected_model, 
            temperature=temperature,
            num_predict=800,       # Failsafe limit: mathematically forces the server to stop generating
            repeat_penalty=1.2     # High penalty (standard is 1.1+) to prevent repetitive stuttering
        )
        
        # Stream response word-by-word
        full_response = ""
        for chunk in llm.stream(messages):
            full_response += chunk.content
            message_placeholder.markdown(full_response + "▌")
            
        message_placeholder.markdown(full_response)
        
    st.session_state.messages.append({"role": "assistant", "content": full_response})