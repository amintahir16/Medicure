"""
FastAPI Server for Medicure MBBS AI Chatbot
Serves API endpoints for grounded retrieval, multi-model generation,
PDF book library management, and static frontend.
"""

import os
import shutil
import uuid
from pathlib import Path
from typing import Optional, List, Dict, Any
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Query
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from backend.config import BASE_DIR, BOOKS_DIR, DEFAULT_LLM_PROVIDER, GEMINI_MODEL, GROQ_MODEL
from backend.engine.retriever import MBBSHybridRetriever
from backend.engine.generator import MBBSGenerator
from backend.engine.indexer import MBBSIndexer
from backend.engine.memory import MBBSMemoryManager
from backend.seed.sample_cases import SAMPLE_PROMPTS, STUDY_VIVA_BANK

import asyncio
from contextlib import asynccontextmanager

# Core engine singletons
indexer = MBBSIndexer()
retriever = MBBSHybridRetriever()
generator = MBBSGenerator(retriever=retriever)
memory = MBBSMemoryManager()

# Modern FastAPI Lifespan Manager
@asynccontextmanager
async def lifespan(app: FastAPI):
    new_books = await asyncio.to_thread(indexer.sync_books_dir)
    if new_books:
        print(f"Auto-indexed new textbooks: {list(new_books.keys())}")
    retriever.invalidate_cache()
    # Pre-warm local biomedical embeddings so user queries experience zero cold-start delay
    print("[*] Pre-warming local PubMedBERT biomedical embedding model...")
    await asyncio.to_thread(retriever.embedder.embed_query, "Medicure curriculum warmup")
    print(f"Medicure Server ready. {len(indexer.get_books_summary())} MBBS books indexed and embedding engine warm.")
    yield

app = FastAPI(
    title="Medicure MBBS AI Chatbot",
    description="Ground-truth medical consultation companion for MBBS students with exact book, chapter, topic and page referencing.",
    version="2.0.0",
    lifespan=lifespan
)

# Enable CORS for development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.middleware("http")
async def add_no_cache_headers(request, call_next):
    response = await call_next(request)
    if any(request.url.path.endswith(ext) for ext in [".js", ".css", ".html"]) or request.url.path == "/":
        response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
        response.headers["Pragma"] = "no-cache"
        response.headers["Expires"] = "0"
    return response


# --- Pydantic Request Models ---
class ChatRequest(BaseModel):
    query: str
    session_id: Optional[str] = None
    provider: Optional[str] = "offline"  # 'offline', 'gemini', 'groq'
    api_key: Optional[str] = None
    model_name: Optional[str] = None
    year_filter: Optional[str] = "all"
    subject_filter: Optional[str] = "all"
    study_mode: Optional[str] = "standard"  # 'standard', 'case_vignette', 'viva_quiz'

class SessionCreateRequest(BaseModel):
    title: Optional[str] = "New Medical Consultation"
    mbbs_year: Optional[str] = "all"
    subject: Optional[str] = "all"


# --- API Routes ---

@app.get("/api/health")
async def health_check():
    books = indexer.get_books_summary()
    total_pages = sum(b.get("total_pages", 0) for b in books)
    return {
        "status": "healthy",
        "system": "Medicure MBBS AI 2.0 (Hybrid Dense-Sparse RAG)",
        "indexed_books_count": len(books),
        "total_curriculum_pages": total_pages,
        "embedding_engine": "PubMedBERT (NeuML/pubmedbert-base-embeddings)",
        "embedding_dimensions": 768,
        "retrieval_architecture": "Dense Vector KNN (sqlite-vec) + Sparse BM25 (FTS5) via RRF",
        "default_provider": DEFAULT_LLM_PROVIDER,
        "active_models": {
            "gemini": GEMINI_MODEL,
            "groq": GROQ_MODEL,
            "offline": "Medicure Built-in Grounded Engine"
        }
    }

@app.post("/api/chat")
async def chat_endpoint(req: ChatRequest):
    if not req.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    session_id = req.session_id or str(uuid.uuid4())
    prev_messages = memory.get_messages(session_id)
    
    # Store user message
    memory.add_message(
        session_id=session_id,
        role="user",
        content=req.query
    )

    # Generate response with grounded citations and multi-turn context
    try:
        result = await generator.generate_response(
            query=req.query,
            provider=req.provider or "offline",
            api_key=req.api_key,
            model_name=req.model_name,
            year_filter=req.year_filter,
            subject_filter=req.subject_filter,
            study_mode=req.study_mode or "standard",
            history=prev_messages
        )

        # Store assistant message
        memory.add_message(
            session_id=session_id,
            role="assistant",
            content=result.get("answer", ""),
            citations=result.get("citations", []),
            provider_used=result.get("provider_used", "offline")
        )

        return {
            "session_id": session_id,
            "answer": result.get("answer", "No response generated."),
            "citations": result.get("citations", []),
            "evidence": result.get("evidence", []),
            "provider_used": result.get("provider_used", "Medicure"),
            "total_references": result.get("total_references", len(result.get("citations", [])))
        }
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Engine error: {str(e)}")

@app.get("/api/sessions")
async def get_sessions():
    return memory.list_sessions()

@app.get("/api/sessions/{session_id}")
async def get_session_messages(session_id: str):
    messages = memory.get_messages(session_id)
    return {"session_id": session_id, "messages": messages}

@app.post("/api/sessions")
async def create_session(req: SessionCreateRequest):
    new_id = str(uuid.uuid4())
    memory.create_session(
        session_id=new_id,
        title=req.title or "New Medical Consultation",
        mbbs_year=req.mbbs_year or "all",
        subject=req.subject or "all"
    )
    return {"session_id": new_id, "title": req.title}

@app.delete("/api/sessions/{session_id}")
async def delete_session(session_id: str):
    memory.delete_session(session_id)
    return {"status": "deleted", "session_id": session_id}

@app.get("/api/books")
async def get_books():
    indexer.sync_books_dir()
    books = indexer.get_books_summary()
    return {"books": books, "total_count": len(books)}

@app.get("/api/books/page")
async def get_page_view(book_title: str, page_number: int):
    page_data = indexer.get_page_content(book_title, page_number)
    if not page_data:
        raise HTTPException(status_code=404, detail="Page not found in indexed records.")
    return page_data

@app.post("/api/books/upload")
async def upload_book(file: UploadFile = File(...)):
    # 1. Sanitize filename to prevent directory traversal
    safe_filename = Path(file.filename).name.strip()
    if not safe_filename or not safe_filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Invalid filename. Only standard .pdf files are supported.")

    target_path = BOOKS_DIR / safe_filename

    # 2. Validate PDF magic bytes and bounded file size (200MB limit)
    MAX_FILE_SIZE = 200 * 1024 * 1024  # 200MB
    chunk_size = 1024 * 1024  # 1MB
    bytes_read = 0

    header = await file.read(4)
    if header != b"%PDF":
        raise HTTPException(status_code=400, detail="Uploaded file is not a valid PDF document (magic header mismatch).")
    await file.seek(0)

    def _save_file():
        nonlocal bytes_read
        with open(target_path, "wb") as buffer:
            while True:
                chunk = file.file.read(chunk_size)
                if not chunk:
                    break
                bytes_read += len(chunk)
                if bytes_read > MAX_FILE_SIZE:
                    raise ValueError("File exceeds maximum allowed size of 200MB.")
                buffer.write(chunk)

    try:
        await asyncio.to_thread(_save_file)
    except ValueError as e:
        if target_path.exists():
            target_path.unlink(missing_ok=True)
        raise HTTPException(status_code=413, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to securely save file: {e}")

    # 3. Offload CPU-heavy parsing and vector embeddings to worker thread without blocking asyncio loop
    try:
        chunks_count = await asyncio.to_thread(indexer.index_book, target_path)
        retriever.invalidate_cache()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to parse and index PDF: {e}")

    return {
        "message": f"Successfully uploaded and indexed '{safe_filename}'.",
        "filename": safe_filename,
        "indexed_chunks": chunks_count
    }

@app.get("/api/sample-prompts")
async def get_sample_prompts():
    return {"prompts": SAMPLE_PROMPTS}

@app.get("/api/viva-quiz")
async def get_viva_bank():
    return {"viva_questions": STUDY_VIVA_BANK}


# --- Mount Frontend Static Files ---
frontend_dir = BASE_DIR / "frontend"
if frontend_dir.exists():
    app.mount("/", StaticFiles(directory=str(frontend_dir), html=True), name="frontend")
