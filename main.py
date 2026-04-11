# ============================================================
#  RAG API BACKEND - main.py
#  Built with FastAPI
#
#  WHAT IS FASTAPI?
#  A modern Python web framework for building APIs.
#  API = Application Programming Interface
#  It's a way for programs to talk to each other via HTTP.
#
#  WHAT IS AN API ENDPOINT?
#  A URL that accepts requests and returns responses.
#  Like a function, but accessible over the internet.
#
#  EXAMPLE:
#  Your browser visits: GET http://localhost:8000/health
#  FastAPI responds:    {"status": "ok"}
#
#  OUR ENDPOINTS:
#  POST /upload    → upload a PDF, get it indexed
#  POST /ask       → ask a question, get an answer
#  GET  /health    → check if server is running
#  GET  /chunks    → see all stored chunks
#
#  HOW TO RUN:
#  uvicorn main:app --reload
#
#  HOW TO TEST:
#  Visit http://localhost:8000/docs
#  FastAPI auto-generates a beautiful test UI!
# ============================================================


# ── IMPORTS ─────────────────────────────────────────────────

from fastapi import FastAPI, UploadFile, File, HTTPException
# FastAPI   → the main framework class
# UploadFile → handles file uploads (PDFs)
# File       → declares a parameter as a file
# HTTPException → for sending error responses with status codes

from fastapi.middleware.cors import CORSMiddleware
# CORS = Cross-Origin Resource Sharing
# Allows your Streamlit UI (port 8501) to call this API (port 8000)
# Without CORS, browsers BLOCK requests between different ports/domains

from pydantic import BaseModel
# Pydantic = data validation library
# BaseModel = base class for defining request/response shapes
# Like a blueprint for what data must look like

import uvicorn
# The server that runs our FastAPI app
# Similar to how Apache/Nginx runs websites

import tempfile
import os
from dotenv import load_dotenv

# Import our RAG pipeline functions
from rag_pipeline import (
    load_embedding_model,
    load_document,
    chunk_text,
    create_embeddings,
    build_faiss_index,
    retrieve_chunks,
    ask_llm
)

# Load environment variables from .env
load_dotenv()


# ============================================================
#  CONCEPT: WHAT IS A PYDANTIC MODEL?
#
#  When your API receives a request, it needs to know
#  exactly what data to expect. Pydantic models define this.
#
#  Example without Pydantic (bad):
#  def ask(data):
#      question = data["question"]  # what if key is missing? CRASH!
#
#  Example with Pydantic (good):
#  class AskRequest(BaseModel):
#      question: str    # FastAPI automatically validates this
#      top_k: int = 3   # default value if not provided
#
#  If request is missing 'question', FastAPI auto-returns:
#  {"detail": "field required"} with status 422
#  No crash, clean error message!
# ============================================================

class AskRequest(BaseModel):
    """
    Shape of the request body for POST /ask endpoint.

    Client must send JSON like:
    {
        "question": "What is photosynthesis?",
        "top_k": 3
    }
    """
    question: str        # required — must be a string
    top_k: int = 3       # optional — defaults to 3


class AskResponse(BaseModel):
    """
    Shape of the response from POST /ask endpoint.

    API will return JSON like:
    {
        "question": "What is photosynthesis?",
        "answer": "Photosynthesis is...",
        "retrieved_chunks": ["chunk1...", "chunk2..."],
        "num_chunks_searched": 4
    }
    """
    question: str
    answer: str
    retrieved_chunks: list[str]
    num_chunks_searched: int


class HealthResponse(BaseModel):
    """Response shape for GET /health"""
    status: str
    rag_ready: bool
    num_chunks: int
    message: str


# ============================================================
#  APP STATE — storing pipeline between requests
#
#  PROBLEM: HTTP is stateless — each request is independent.
#  If we process a PDF in /upload, the next /ask request
#  has no memory of it!
#
#  SOLUTION: Store pipeline objects in app.state
#  app.state is a simple object that persists as long as
#  the server is running.
#
#  Think of it like:
#  app.state = the server's long-term memory
#  request    = the server's short-term memory (gone after response)
# ============================================================

# Create the FastAPI application
app = FastAPI(
    title="RAG Chatbot API",
    description="A RAG pipeline that answers questions about your documents",
    version="1.0.0"
    # These fields appear in the auto-generated /docs UI!
)

# ── CORS MIDDLEWARE ──────────────────────────────────────────
# Middleware = code that runs on EVERY request before/after handler
# CORS middleware adds headers that tell browsers:
# "Yes, it's OK for other origins to call this API"
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],      # allow ALL origins (ok for development)
    allow_methods=["*"],      # allow GET, POST, PUT, DELETE etc
    allow_headers=["*"],      # allow all headers
)

# ── INITIALIZE APP STATE ─────────────────────────────────────
# These will be populated when a document is uploaded
app.state.chunks      = []      # text chunks from document
app.state.faiss_index = None    # FAISS vector index
app.state.embed_model = None    # sentence transformer model
app.state.rag_ready   = False   # True once document is indexed
app.state.api_key     = os.getenv("GROQ_API_KEY", "")


# ============================================================
#  STARTUP EVENT
#
#  @app.on_event("startup") runs ONCE when server starts.
#  Perfect for loading heavy things like ML models.
#
#  WHY HERE AND NOT IN EACH REQUEST?
#  Loading the embedding model takes 2-3 seconds.
#  If we loaded it in every /ask request:
#  → each question takes 3+ seconds just to load the model!
#
#  Loading once at startup:
#  → server starts a bit slow (3 seconds)
#  → every subsequent request is instant ✅
# ============================================================

@app.on_event("startup")
async def startup_event():
    """
    Runs once when FastAPI server starts.
    Pre-loads the embedding model so first request is fast.
    """
    print("🚀 RAG API starting up...")
    print("📦 Pre-loading embedding model...")
    app.state.embed_model = load_embedding_model()
    print("✅ Server ready!")

    if not app.state.api_key:
        print("⚠️  Warning: GROQ_API_KEY not found in .env")
    else:
        print("✅ Groq API key loaded")


# ============================================================
#  ENDPOINTS
#
#  WHAT IS AN ENDPOINT?
#  A URL + HTTP method combination that your API responds to.
#
#  HTTP METHODS:
#  GET    → retrieve data    (read-only, no body)
#  POST   → send data        (creates/processes something)
#  PUT    → update data      (replaces existing)
#  DELETE → remove data
#
#  STATUS CODES (what the server returns):
#  200 → OK (success)
#  201 → Created (something was made)
#  400 → Bad Request (client sent wrong data)
#  404 → Not Found
#  422 → Validation Error (Pydantic caught bad data)
#  500 → Internal Server Error (our code crashed)
# ============================================================


# ── ENDPOINT 1: HEALTH CHECK ─────────────────────────────────
@app.get("/health", response_model=HealthResponse)
async def health_check():
    """
    GET /health
    Returns server status and whether RAG pipeline is ready.

    Use this to check if:
    - Server is running
    - Document has been uploaded and indexed
    - How many chunks are stored

    Example response:
    {
        "status": "ok",
        "rag_ready": true,
        "num_chunks": 42,
        "message": "RAG pipeline ready with 42 chunks"
    }
    """
    num_chunks = len(app.state.chunks)

    if app.state.rag_ready:
        message = f"RAG pipeline ready with {num_chunks} chunks"
    else:
        message = "No document uploaded yet. POST to /upload first."

    return HealthResponse(
        status="ok",
        rag_ready=app.state.rag_ready,
        num_chunks=num_chunks,
        message=message
    )


# ── ENDPOINT 2: UPLOAD DOCUMENT ──────────────────────────────
@app.post("/upload")
async def upload_document(
    file: UploadFile = File(...),
    chunk_size: int = 200,
    overlap: int = 50
):
    """
    POST /upload
    Upload a PDF or TXT file to index for RAG.

    WHAT HAPPENS HERE:
    1. Receive the uploaded file (as bytes)
    2. Save to a temp file on disk
    3. Run it through RAG pipeline (chunk → embed → index)
    4. Store results in app.state
    5. Return success message

    Parameters:
    - file: the PDF or TXT file (sent as form data)
    - chunk_size: words per chunk (query param, default 200)
    - overlap: overlap words (query param, default 50)

    Example with curl:
    curl -X POST "http://localhost:8000/upload" \
         -F "file=@myresume.pdf"
    """

    # ── VALIDATE FILE TYPE ────────────────────────────────
    # file.filename = original filename e.g. "resume.pdf"
    allowed = ['.pdf', '.txt', '.docx']   # ← added .docx
    file_ext = os.path.splitext(file.filename)[1].lower()

    if file_ext not in allowed:
        # HTTPException sends a proper error response
        # status_code=400 means "Bad Request — client's fault"
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{file_ext}'. Use .pdf or .txt"
        )

    # ── READ FILE CONTENTS ────────────────────────────────
    # file.read() returns bytes (raw binary data)
    # We need to save to disk because load_document() needs a path
    try:
        contents = await file.read()
        # 'await' = this is async code
        # file.read() is an async operation (non-blocking)
        # 'await' says "wait for this to finish before continuing"

        # Save to temporary file
        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=file_ext
        ) as tmp:
            tmp.write(contents)
            tmp_path = tmp.name
        # tmp_path = something like "C:\Temp\tmpXXXX.pdf"

        # ── RUN RAG PIPELINE ──────────────────────────────
        print(f"\n📄 Processing: {file.filename}")

        # Step 1: Extract text
        text = load_document(tmp_path)

        # Step 2: Chunk
        chunks = chunk_text(text, chunk_size=chunk_size, overlap=overlap)

        # Step 3: Embed (model already loaded at startup)
        embeddings = create_embeddings(chunks, app.state.embed_model)

        # Step 4: Build FAISS index
        index = build_faiss_index(embeddings)

        # Step 5: Save to app state for future /ask requests
        app.state.chunks      = chunks
        app.state.faiss_index = index
        app.state.rag_ready   = True

        # Clean up temp file
        os.unlink(tmp_path)

        return {
            "message"   : f"✅ Document indexed successfully!",
            "filename"  : file.filename,
            "num_chunks": len(chunks),
            "chunk_size": chunk_size,
            "overlap"   : overlap
        }

    except Exception as e:
        # If anything goes wrong, return a 500 error
        raise HTTPException(
            status_code=500,
            detail=f"Processing failed: {str(e)}"
        )


# ── ENDPOINT 3: ASK A QUESTION ───────────────────────────────
@app.post("/ask", response_model=AskResponse)
async def ask_question(request: AskRequest):
    """
    POST /ask
    Ask a question about the uploaded document.

    WHAT HAPPENS HERE:
    1. Check that a document has been uploaded first
    2. Run retrieval — find top-k relevant chunks
    3. Send chunks + question to Groq
    4. Return answer + retrieved chunks

    Request body (JSON):
    {
        "question": "What is photosynthesis?",
        "top_k": 3
    }

    Example with curl:
    curl -X POST "http://localhost:8000/ask" \
         -H "Content-Type: application/json" \
         -d '{"question": "What is photosynthesis?"}'
    """

    # ── CHECK RAG IS READY ────────────────────────────────
    if not app.state.rag_ready:
        raise HTTPException(
            status_code=400,
            detail="No document uploaded yet! POST a file to /upload first."
        )

    # ── CHECK API KEY ─────────────────────────────────────
    if not app.state.api_key:
        raise HTTPException(
            status_code=500,
            detail="GROQ_API_KEY not configured. Add it to .env file."
        )

    try:
        # Step 5: Retrieve relevant chunks
        relevant_chunks = retrieve_chunks(
            request.question,
            app.state.embed_model,
            app.state.faiss_index,
            app.state.chunks,
            top_k=request.top_k
        )

        # Step 6: Generate answer
        answer = ask_llm(
            request.question,
            relevant_chunks,
            app.state.api_key
        )

        # Return structured response
        return AskResponse(
            question=request.question,
            answer=answer,
            retrieved_chunks=relevant_chunks,
            num_chunks_searched=len(app.state.chunks)
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error generating answer: {str(e)}"
        )


# ── ENDPOINT 4: VIEW ALL CHUNKS ───────────────────────────────
@app.get("/chunks")
async def get_chunks():
    """
    GET /chunks
    Returns all stored chunks — useful for debugging.

    Shows you exactly how your document was split.
    """
    if not app.state.rag_ready:
        raise HTTPException(
            status_code=400,
            detail="No document uploaded yet!"
        )

    return {
        "total_chunks": len(app.state.chunks),
        "chunks": [
            {
                "index"  : i,
                "preview": chunk[:150] + "...",
                "length" : len(chunk.split())
            }
            for i, chunk in enumerate(app.state.chunks)
        ]
    }


# ── ENDPOINT 5: ROOT ─────────────────────────────────────────
@app.get("/")
async def root():
    """
    GET /
    Welcome message — confirms API is running.
    """
    return {
        "message": "RAG Chatbot API is running!",
        "docs"   : "Visit http://localhost:8000/docs for API documentation",
        "endpoints": {
            "GET  /health" : "Check server status",
            "POST /upload" : "Upload a PDF or TXT file",
            "POST /ask"    : "Ask a question",
            "GET  /chunks" : "View all stored chunks"
        }
    }


# ── RUN SERVER ───────────────────────────────────────────────
# This only runs when you execute: python main.py
# (not when imported by another file)
if __name__ == "__main__":
    uvicorn.run(
        "main:app",   # "filename:FastAPI_instance_name"
        host="0.0.0.0",
        port=8000,
        reload=True   # auto-restarts when you save the file
                      # Great for development!
    )