# 🦙 Privacy-First Local AI Document Assistant

An offline, secure AI web application built in Python that runs Large Language Models locally on your hardware using Ollama, LangChain, and Streamlit. Designed for processing confidential documents without data ever leaving your machine.

---

## 🌟 Key Features

* **🔒 100% Offline & Private:** Operates entirely locally via Ollama with zero external API calls, ensuring absolute data privacy for sensitive documents.
* **📄 Secure Document Ingestion:** Built-in PDF reader (`PyPDF2`) that injects document context directly into the local model's prompt pipeline via the sidebar.
* **⚡ Real-Time Token Streaming:** Features live, chunk-by-chunk streaming responses with a dynamic typing indicator for a smooth user experience.
* **🐍 Pure Python Architecture:** Built using Streamlit to deliver a reactive, enterprise-grade user interface without requiring complex frontend frameworks.

---

## 🛠️ Tech Stack

* **UI Framework:** [Streamlit](https://streamlit.io/)
* **AI Orchestration:** [LangChain](https://www.langchain.com/) (`langchain-community`)
* **Local Inference Engine:** [Ollama](https://ollama.com/) running **Meta Llama 3.2 (3B)**
* **Document Processing:** PyPDF2

---

## 🚀 Getting Started

### 1. Prerequisites
Ensure you have [Ollama](https://ollama.com/) installed and running on your system with the Llama 3.2 model pulled:
```bash
ollama run llama3.2
2. Clone and Setup Environment
Bash
git clone [https://github.com/gulding/local-ai-app.git](https://github.com/gulding/local-ai-app.git)
cd local-ai-app

python -m venv venv
# On Windows:
.\venv\Scripts\activate
3. Install Dependencies
pip install langchain langchain-community streamlit PyPDF2

4. Run the Application
Bash
streamlit run app.py
Open http://localhost:8501 in your browser to interact with your local offline assistant.
