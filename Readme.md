# RAG Chatbot 🤖

A Retrieval-Augmented Generation (RAG) chatbot that answers questions about your documents and web pages using AI.

## What is RAG?

Normal AI answers from memory and may hallucinate.
RAG = AI reads YOUR documents first, then answers.
Like an open-book exam instead of closed-book!

**Your Document → Chunks → Embeddings → FAISS → Answer**
## Features

- ✅ Supports PDF, DOCX, XLSX, PPTX, TXT files
- ✅ Scrapes up to 5 web URLs
- ✅ Semantic search using FAISS vector database
- ✅ Answers grounded in your documents (no hallucination)
- ✅ Shows source links with every answer
- ✅ Adjustable chunk size and overlap settings
- ✅ Full logging to terminal and rag_app.log file
- ✅ FastAPI backend with auto-generated API docs
- ✅ Streamlit chat interface

## Project Structure
rag_project/
├── rag_pipeline.py   → Core RAG logic (load, chunk, embed, retrieve, answer)
├── app.py            → Streamlit chat UI
├── main.py           → FastAPI REST API backend
├── .env              → API keys (never share this!)
├── rag_app.log       → Application logs
└── requirements.txt  → All Python dependencies
## Tech Stack

| Layer | Technology |
|-------|-----------|
| Embeddings | sentence-transformers (all-MiniLM-L6-v2) |
| Vector DB | FAISS (Facebook AI Similarity Search) |
| LLM | Groq API (Llama 3.3 70B) |
| UI | Streamlit |
| API | FastAPI + Uvicorn |
| PDF | PyMuPDF (fitz) |
| Word | python-docx |
| Excel | openpyxl |
| PowerPoint | python-pptx |
| Web Scraping | requests + BeautifulSoup4 |

## Setup Instructions

### Step 1 — Clone the repository
```bash
git clone https://github.com/YOUR_USERNAME/rag-chatbot.git
cd rag-chatbot
```

### Step 2 — Create virtual environment
```bash
python -m venv venv
```

Activate it:
```bash
# Windows:
venv\Scripts\activate

# Mac/Linux:
source venv/bin/activate
```

### Step 3 — Install all dependencies
```bash
pip install -r requirements.txt
```

### Step 4 — Get your free Groq API key
1. Go to https://console.groq.com
2. Sign in with Google
3. Click API Keys → Create API Key
4. Copy the key (starts with gsk_...)

### Step 5 — Create .env file
Create a file named `.env` in the project root:

GROQ_API_KEY=gsk_your_key_here
### Step 6 — Run the Streamlit UI
```bash
streamlit run app.py --server.fileWatcherType none
```
Open browser at: **http://localhost:8501**

### Step 7 — Run the FastAPI backend (optional, separate terminal)
```bash
uvicorn main:app --reload
```
Open API docs at: **http://localhost:8000/docs**

## How to Use

### Option A — Upload a File
1. Open http://localhost:8501
2. In the sidebar under "Upload Document"
3. Upload any PDF, DOCX, XLSX, PPTX or TXT file
4. Click "Process Document"
5. Ask questions in the chat box at the bottom

### Option B — Enter Web URLs
1. In the sidebar under "Or Enter Web URLs"
2. Enter up to 5 URLs, one per line:

INDEXING PHASE — runs once when document is uploaded:
Document (PDF/DOCX/URL)
↓
load_document()      → extract raw text
↓
chunk_text()         → split into 200-word overlapping pieces
↓
create_embeddings()  → convert each chunk to 384 numbers (vector)
↓
build_faiss_index()  → store all vectors for fast search
QUERY PHASE — runs every time user asks a question:
User Question
↓
embed question       → convert to 384 numbers
↓
FAISS search         → find top-3 most similar chunk vectors
↓
retrieve chunks      → get the actual text of those chunks
↓
ask_llm()            → send chunks + question to Groq
↓
Answer               → grounded in YOUR document ✅

## Logging

All operations are logged to two places simultaneously:
- **Terminal** — real-time while app runs
- **rag_app.log** — permanent log file in project folder

Log format:
2024-01-15 14:23:01 INFO     Reading PDF: resume.pdf
2024-01-15 14:23:02 INFO     Created 42 chunks
2024-01-15 14:23:05 WARNING  Low relevance score (45%) — answer may be inaccurate
2024-01-15 14:23:06 ERROR    URL timeout: https://example.com
## Chunk Settings Guide

| Document Size | Recommended chunk_size | Recommended overlap |
|---------------|----------------------|-------------------|
| 1-2 pages | 100-150 words | 20-30 words |
| 5-20 pages | 200-300 words | 50 words |
| 50-100 pages | 300-400 words | 75-100 words |
| Books (100+ pages) | 400-500 words | 100-150 words |

## Troubleshooting

**Problem: Groq quota exceeded**
Solution: Create a new API key at https://console.groq.com

**Problem: ModuleNotFoundError**
```bash
Solution: pip install -r requirements.txt
Make sure venv is activated (you should see (venv) in terminal)
```

**Problem: PDF not loading**
Solution: Make sure pymupdf is installed: pip install pymupdf

**Problem: URL not loading**
Solution: Check internet connection
Some websites block scraping — try a different URL

**Problem: Indentation errors**
Solution: In VS Code press Shift+Alt+F to auto-format the file

## Generate requirements.txt

```bash
pip freeze > requirements.txt
```

## Author

**Vinit Kumar**
Heritage Institute of Technology, Kolkata
B.Tech (ECE) |  Software developer at TCS

## License

MIT License — free to use, modify and distribute.
