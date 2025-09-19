# 🤖 Chat With Gemma

A **feature-rich, production-ready** Streamlit application that lets you chat with **Google’s Gemma** models (or any Ollama-compatible LLM) in two modes:


| Mode | Description |
|------|-------------|
| **Chat** | Classic back-and-forth conversation with memory |
| **RAG** | Upload documents (PDF, TXT, MD) and ask questions *about* them |

---

## 🚀 Quick Start

1. **Install Ollama**  
   [https://ollama.ai](https://ollama.ai)  
   ```bash
   ollama pull gemma3:1b
   ```

2. **Clone & Install**
   ```bash
   git clone https://github.com/your-org/chat-with-gemma.git
   cd chat-with-gemma
   pip install -r requirements.txt
   ```

3. **Launch**
   ```bash
   streamlit run app.py
   ```
   Browser opens at [http://localhost:8501](http://localhost:8501)

---

## 🔧 Requirements

| Package | Purpose |
|---------|---------|
| `streamlit` | Web UI |
| `langchain-ollama` | LLM integration |
| `langchain-community` | Memory, loaders, FAISS |
| `sentence-transformers` | Embeddings |
| `pypdf` | PDF parsing |
| `nltk` | Text utilities |
| `scikit-learn` | Similarity metrics |
| `plotly` | Analytics charts |
| `streamlit-ace` | Code editor (system prompt) |

Install all at once:
```bash
pip install -r requirements.txt
```

---

## 📁 Project Structure

```
chat-with-gemma/
├── app.py                  # Main entry point
├── requirements.txt
├── README.md
└── chat_data/              # Auto-created at runtime
    ├── faiss_index/        # Vector store snapshots
    ├── histories/          # JSON chat exports
    ├── cache/              # Temporary downloads
    ├── logs/               # chat_app.log
    └── chat_analytics.db   # SQLite analytics DB
```

---

## ⚡ Key Features

| Feature | Description |
|---------|-------------|
| **Multi-doc RAG** | Upload many PDF/TXT/MD files; chunking + FAISS retrieval |
| **Confidence Score** | Real-time response quality indicator |
| **Response Metrics** | Latency & token-speed tracking |
| **Conversation Memory** | Buffer memory with conversation ID isolation |
| **Analytics Dashboard** | 7-day rolling stats (messages, avg confidence, response time) |
| **Modern UI** | Glass-morphism cards, dark/light aware, mobile-friendly |
| **System Prompt Presets** | Academic, Creative, Professional one-click personas |
| **Import / Export** | Full conversation JSON download & replay |
| **Graceful Errors** | User-friendly messages + collapsible stack-traces |
| **SQLite Logging** | All messages + metadata for downstream BI |
| **Hot-reload Safe** | Session-state hygiene, no duplicate downloads |


## ⚙️ Configuration

All constants live at the top of `app.py`:

```python
OLLAMA_DEFAULT_URL = "http://localhost:11434"
DEFAULT_MODEL        = "gemma3:1b"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
CHUNK_SIZE           = 1000
CHUNK_OVERLAP        = 200
```

Override with env-vars if desired:
```bash
export OLLAMA_URL="http://remote-ollama:11434"
streamlit run app.py
```

## 🔒 Security Notes

* Uploaded documents are **stored only in RAM** and **deleted** after FAISS index creation (temp-file lifecycle).  
* No analytics are sent to third parties; everything stays local.  
* SQL parameters are used everywhere → no injection risk.  
* File-size limit: 10 MB per document (changeable in `process_document`).

## 🐛 Troubleshooting

| Symptom | Fix |
|---------|-----|
| `ImportError: langchain_ollama` | `pip install langchain-ollama` |
| `nltk punkt not found` | Auto-downloaded on first run (quiet mode) |
| `Connection refused` | Ensure Ollama service is running on the configured URL |
| `PDF fails to load` | `pip install pypdf` and check encoding |
| `FAISS index empty` | Upload docs → **Process Documents** button must be clicked |

Logs are verbose: `chat_data/logs/chat_app.log`

## License 📜
This project is licensed under the **Apache License 2.0**.

## Contact 📬
For questions, bug reports, or feature requests, please open an issue on GitHub or contact me at [ketanedumail@gmail.com]. 📧

*Built with ❤️ using Streamlit.*
