import os
import shutil
import asyncio
from pathlib import Path
from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel

from app.session import session_manager
from app.queue import request_queue
from src.ingestion import chunk_document
from src.retrieval import VectorStore, BM25Retriever, reciprocal_rank_fusion
from src.reranker import Reranker
from src.generator import get_llm, generate_response
from src.figures import extract_figures

UPLOAD_DIR = Path("uploads")
FIGURES_DIR = Path("figures")
UPLOAD_DIR.mkdir(exist_ok=True)
FIGURES_DIR.mkdir(exist_ok=True)

SUPPORTED_FORMATS = {".pdf", ".docx", ".txt", ".pptx", ".csv"}
MAX_FILE_SIZE = 50 * 1024 * 1024  # 50MB

app = FastAPI(title="DocuMind API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/figures", StaticFiles(directory="figures"), name="figures")

# Serve React frontend static assets
if os.path.exists("frontend/assets"):
    app.mount("/assets", StaticFiles(directory="frontend/assets"), name="static-assets")

reranker = Reranker()
llm = get_llm()

class QueryRequest(BaseModel):
    session_id: str
    query: str
    notes_mode: str = "medium"

@app.get("/health")
def health():
    return {"status": "ok", "queue": request_queue.waiting}

@app.get("/debug/sessions")
def debug_sessions():
    return {"sessions": list(session_manager._sessions.keys())}

@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    ext = Path(file.filename).suffix.lower()
    if ext not in SUPPORTED_FORMATS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported format '{ext}'. Supported: {list(SUPPORTED_FORMATS)}"
        )

    contents = await file.read()
    if len(contents) > MAX_FILE_SIZE:
        raise HTTPException(status_code=400, detail="File too large. Maximum size is 50MB.")

    session = session_manager.create()
    file_path = UPLOAD_DIR / f"{session.session_id}{ext}"
    with open(file_path, "wb") as f:
        f.write(contents)

    async def process():
        chunks = chunk_document(str(file_path))
        if not chunks:
            raise HTTPException(status_code=422, detail="Could not extract text from document.")

        vs = VectorStore()
        vs.add_chunks(chunks)
        bm25 = BM25Retriever(chunks)
        figures = extract_figures(str(file_path))

        session.vector_store = vs
        session.bm25 = bm25
        session.chunks = chunks
        session.filename = file.filename
        session.total_pages = max(c["page"] for c in chunks)
        session.figures = figures

    await request_queue.run(process)

    return {
        "session_id": session.session_id,
        "filename": file.filename,
        "chunks": len(session.chunks),
        "pages": session.total_pages,
        "figures": len(session.figures),
        "ocr": any(c.get("ocr") for c in session.chunks)
    }

@app.post("/query")
async def query(req: QueryRequest):
    session = session_manager.get(req.session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found. Please upload a document first.")
    if not session.vector_store:
        raise HTTPException(status_code=422, detail="Document not indexed yet.")

    top_k_map = {"brief": 5, "medium": 8, "detailed": 12, "qa": 8, "qa_brief": 5, "qa_detailed": 12}
    top_k = top_k_map.get(req.notes_mode, 8)

    async def process():
        vector_results = session.vector_store.search(req.query, k=10)
        bm25_results = session.bm25.search(req.query, k=10)
        fused = reciprocal_rank_fusion(vector_results, bm25_results)
        top_chunks, top_score = reranker.rerank_with_scores(req.query, fused, top_k=top_k)
        return generate_response(req.query, top_chunks, llm, notes_mode=req.notes_mode, top_reranker_score=top_score)
    
        result = await request_queue.run(process)
    
        return {
            "answer": result["answer"],
            "sources": result["sources"],
            "mode": result["mode"],
            "source_type": result["source_type"],
            "queue_position": request_queue.waiting
        }

@app.get("/session-figures/{session_id}")
def get_figures(session_id: str):
    session = session_manager.get(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found.")
    return {"figures": session.figures}

@app.delete("/session/{session_id}")
def delete_session(session_id: str):
    if not session_manager.exists(session_id):
        raise HTTPException(status_code=404, detail="Session not found.")
    session_manager.delete(session_id)
    return {"message": "Session deleted."}

# Serve React frontend — must be last
@app.get("/{full_path:path}")
def serve_frontend(full_path: str):
    if os.path.exists("frontend/index.html"):
        return FileResponse("frontend/index.html")
    return JSONResponse({"detail": "Frontend not found"}, status_code=404)
