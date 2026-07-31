import streamlit as st
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_core.documents import Document
import PyPDF2

# --- 1. UI Configuration ---
st.set_page_config(page_title="Enterprise Local RAG", page_icon="🦙", layout="wide")
st.title("Enterprise Local RAG & Source Citation Engine 🦙")
st.caption("100% Offline | Page-Level Attribution | Multi-Doc Vault")

# --- 2. Sidebar Controls ---
with st.sidebar:
    st.header("⚙️ Engine Configuration")
    selected_model = st.selectbox("Local Model", ["llama3.2", "llama3.1", "mistral", "gemma2"])
    temperature = st.slider("Temperature (Creativity vs Fact)", 0.0, 1.0, 0.2, 0.1)
    
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
    
    # --- 3. Multi-Document Vault & Metadata Processing ---
    st.header("📄 Secure Document Vault")
    st.write("Upload multiple PDFs. Metadata will track page numbers for verified citations.")
    uploaded_files = st.file_uploader("Upload PDFs", type=["pdf"], accept_multiple_files=True)
    
    # Clear Vault Button
    if st.button("🗑️ Reset Vault & Clear Memory"):
        st.session_state.pop("vector_store", None)
        st.session_state.pop("processed_files", None)
        st.session_state.messages = []
        st.success("Vault reset successfully!")
        st.rerun()

    # --- Export Chat History ---
    if st.session_state.get("messages"):
        st.divider()
        st.subheader("💾 Save Session")
        
        # Compile the chat history into a Markdown string
        export_text = "# 🦙 Local RAG Session Export\n\n"
        for msg in st.session_state.messages:
            role = "👤 **User**" if msg["role"] == "user" else "🤖 **AI Assistant**"
            export_text += f"{role}\n\n{msg['content']}\n\n"
            
            # Append citations if they exist
            if "sources" in msg and msg["sources"]:
                export_text += "*Citations:*\n"
                for src in msg["sources"]:
                    export_text += f"- `{src['source']}` (Page {src['page']})\n"
                export_text += "\n"
            
            export_text += "---\n\n"
            
        # Streamlit Download Button
        st.download_button(
            label="📥 Download as Markdown",
            data=export_text,
            file_name="legal_analysis_export.md",
            mime="text/markdown",
            use_container_width=True
        )

    if uploaded_files:
        current_filenames = [f.name for f in uploaded_files]
        if "processed_files" not in st.session_state or st.session_state.processed_files != current_filenames:
            with st.spinner("Chunking & Vectorizing Documents with Metadata..."):
                raw_documents = []
                
                # Extract text while attaching File Name & Page Number metadata
                for pdf_file in uploaded_files:
                    pdf_reader = PyPDF2.PdfReader(pdf_file)
                    for page_idx, page in enumerate(pdf_reader.pages):
                        page_text = page.extract_text()
                        if page_text:
                            raw_documents.append(
                                Document(
                                    page_content=page_text,
                                    metadata={"source": pdf_file.name, "page": page_idx + 1}
                                )
                            )
                
                # Split documents keeping metadata intact
                text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
                split_docs = text_splitter.split_documents(raw_documents)
                
                # Build FAISS Vector Index using lightweight embeddings
                embeddings = OllamaEmbeddings(model="nomic-embed-text")
                st.session_state.vector_store = FAISS.from_documents(split_docs, embedding=embeddings)
                st.session_state.processed_files = current_filenames
                
            st.success(f"✅ Indexed {len(uploaded_files)} Document(s) across {len(split_docs)} chunks!")

# --- 4. Chat History Rendering ---
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "sources" in message and message["sources"]:
            with st.expander("📚 Verified Source Passages"):
                for src in message["sources"]:
                    st.markdown(f"**Document:** `{src['source']}` | **Page:** `{src['page']}`")
                    st.caption(f"\"{src['snippet']}...\"")

# --- 5. Execution Loop & Citation Injection ---
if prompt := st.chat_input("Query your local vault..."):
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        retrieved_docs = []
        context_str = ""
        sources_meta = []
        
        # Retrieval with Metadata
        if "vector_store" in st.session_state:
            retriever = st.session_state.vector_store.as_retriever(search_kwargs={"k": 3})
            retrieved_docs = retriever.invoke(prompt)
            
            context_blocks = []
            for doc in retrieved_docs:
                src_name = doc.metadata.get("source", "Unknown")
                pg_num = doc.metadata.get("page", "?")
                context_blocks.append(f"[Source: {src_name}, Page {pg_num}]\n{doc.page_content}")
                
                sources_meta.append({
                    "source": src_name,
                    "page": pg_num,
                    "snippet": doc.page_content[:250].replace("\n", " ")
                })
                
            context_str = f"\n\n--- RETRIEVED CONTEXT ---\n" + "\n\n".join(context_blocks)

        # Assemble Messages
        messages = [SystemMessage(content=personas[system_persona])]
        for msg in st.session_state.messages[:-1]:
            if msg["role"] == "user":
                messages.append(HumanMessage(content=msg["content"]))
            elif msg["role"] == "assistant":
                messages.append(AIMessage(content=msg["content"]))
                
        messages.append(HumanMessage(content=f"Current Question: {prompt}{context_str}"))

        # Local LLM Stream
        llm = ChatOllama(
            model=selected_model,
            temperature=temperature,
            num_predict=800,
            repeat_penalty=1.2
        )
        
        full_response = ""
        for chunk in llm.stream(messages):
            full_response += chunk.content
            message_placeholder.markdown(full_response + "▌")
            
        message_placeholder.markdown(full_response)
        
        # Render Citations Container immediately
        if sources_meta:
            with st.expander("📚 Verified Source Passages"):
                for src in sources_meta:
                    st.markdown(f"**Document:** `{src['source']}` | **Page:** `{src['page']}`")
                    st.caption(f"\"{src['snippet']}...\"")

    # Store full state including sources
    st.session_state.messages.append({
        "role": "assistant",
        "content": full_response,
        "sources": sources_meta
    })