import uuid
import faiss
import numpy as np
from typing import Optional
from src.retrieval import VectorStore, BM25Retriever

class Session:
    def __init__(self):
        self.session_id = str(uuid.uuid4())
        self.vector_store: Optional[VectorStore] = None
        self.bm25: Optional[BM25Retriever] = None
        self.chunks: list = []
        self.filename: str = ""
        self.total_pages: int = 0
        self.figures: list = []

class SessionManager:
    def __init__(self):
        self._sessions: dict[str, Session] = {}

    def create(self) -> Session:
        session = Session()
        self._sessions[session.session_id] = session
        return session

    def get(self, session_id: str) -> Optional[Session]:
        return self._sessions.get(session_id)

    def delete(self, session_id: str) -> None:
        if session_id in self._sessions:
            del self._sessions[session_id]

    def exists(self, session_id: str) -> bool:
        return session_id in self._sessions

session_manager = SessionManager()
