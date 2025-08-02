# 📄 PDF AI Chatbot (FastAPI + Hugging Face)

Upload a PDF and ask questions like a human would.  
This app extracts PDF content, indexes it with embeddings, and answers natural language queries.

---

## 🚀 Features

- Upload PDFs
- Chunk + embed content using `sentence-transformers`
- Store and search with FAISS vector store
- Answer questions using HuggingFace QA (`deepset/roberta-base-squad2`)
- Powered by FastAPI

---

## 🧠 How it works

1. **PDF Upload**  
   → PDF is parsed into text pages using PyMuPDF

2. **Embedding**  
   → Text is split into small chunks and converted to vector embeddings using `all-MiniLM-L6-v2`

3. **Indexing**  
   → Embeddings are stored in FAISS for fast similarity search

4. **Question Answering**  
   → When a question is asked, top 5 relevant chunks are retrieved and used as context for a QA model

---

## 🛠️ Setup

```bash
# Clone the repo
git clone https://github.com/your-username/pdf-ai-chat.git
cd pdf-ai-chat

# Install dependencies
pip install -r requirements.txt

# Run the API
uvicorn app.main:app --reload
