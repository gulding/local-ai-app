# 🦙 Enterprise Local RAG Assistant & Document Intelligence

An offline, privacy-first Retrieval-Augmented Generation (RAG) platform built in Python using **LangChain**, **FAISS**, **Ollama**, and **Streamlit**. Designed for local semantic search, document ingestion, hyperparameter tuning, and dynamic persona injection on confidential files with zero cloud API dependencies.

---

## 🌟 Key Features

* **🧠 Semantic FAISS Vector Search:** Slices massive PDFs using `RecursiveCharacterTextSplitter` and indexes text into a high-performance local vector store using `nomic-embed-text`.
* **🔒 100% Offline & Zero Latency Privacy:** Runs Meta's `llama3.2` locally via Ollama to query sensitive corporate, legal, or financial documents with complete data isolation.
* **🎭 Dynamic System Personas:** Instant system prompt switching across custom personas (e.g., *Balkan Legal & Compliance Analyst*, *Senior Python Developer*, *Executive Summarizer*).
* **🎛️ Hyperparameter Controls & Anti-Looping:** Interactive sidebar sliders for temperature tuning, token context buffers, and custom repetition penalties to eliminate text degeneration loops.
* **⚡ Real-Time Token Streaming:** Chunk-by-chunk output generation via LangChain execution chains for responsive user UX.

---

## 🛠️ Tech Stack

* **UI Framework:** [Streamlit](https://streamlit.io/)
* **Vector Database:** [FAISS (Facebook AI Similarity Search)](https://github.com/facebookresearch/faiss)
* **Embedding Model:** `nomic-embed-text` (via Ollama)
* **LLM Engine:** Meta `llama3.2` (via Ollama)
* **AI Orchestration:** LangChain (`langchain-ollama`, `langchain-community`)
* **Document Parser:** PyPDF2

---

## 🚀 Getting Started

### 1. Prerequisites
Install [Ollama](https://ollama.com/) and pull both the generation and embedding models:
```bash
ollama run llama3.2
ollama pull nomic-embed-text
2. Setup Project & Environment
Bash
git clone [https://github.com/gulding/local-ai-app.git](https://github.com/gulding/local-ai-app.git)
cd local-ai-app

python -m venv venv
# On Windows:
.\venv\Scripts\activate
3. Install Dependencies
Bash
pip install langchain-ollama langchain-community langchain-text-splitters faiss-cpu streamlit PyPDF2
4. Run the Application
Bash
streamlit run app.py
Open http://localhost:8501 to upload private PDFs, select your AI persona, and perform vector-backed document analysis.
