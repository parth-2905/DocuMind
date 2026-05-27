import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
from typing import List

class VectorStore:
    def __init__(self):
        self.embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        self.index = None
        self.chunks = []

    def add_chunks(self, chunks: List[dict]) -> None:
        self.chunks = chunks
        texts = [c["text"] for c in chunks]
        embeddings = self.embedder.encode(texts, show_progress_bar=False, convert_to_numpy=True).astype("float32")
        faiss.normalize_L2(embeddings)
        self.index = faiss.IndexFlatIP(embeddings.shape[1])
        self.index.add(embeddings)

    def search(self, query: str, k: int = 10) -> List[dict]:
        q_emb = self.embedder.encode([query], convert_to_numpy=True).astype("float32")
        faiss.normalize_L2(q_emb)
        k = min(k, len(self.chunks))
        scores, indices = self.index.search(q_emb, k)
        return [
            {**self.chunks[i], "score": float(scores[0][rank])}
            for rank, i in enumerate(indices[0]) if i != -1
        ]

class BM25Retriever:
    def __init__(self, chunks: List[dict]):
        self.chunks = chunks
        tokenised = [c["text"].lower().split() for c in chunks]
        self.bm25 = BM25Okapi(tokenised)

    def search(self, query: str, k: int = 10) -> List[dict]:
        scores = self.bm25.get_scores(query.lower().split())
        top_k = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
        return [self.chunks[i] for i in top_k]

def reciprocal_rank_fusion(
    vector_results: List[dict],
    bm25_results: List[dict],
    k: int = 60
) -> List[dict]:
    scores = {}
    chunk_map = {}

    for rank, chunk in enumerate(vector_results):
        cid = chunk["chunk_id"]
        scores[cid] = scores.get(cid, 0) + 1 / (k + rank + 1)
        chunk_map[cid] = chunk

    for rank, chunk in enumerate(bm25_results):
        cid = chunk["chunk_id"]
        scores[cid] = scores.get(cid, 0) + 1 / (k + rank + 1)
        chunk_map[cid] = chunk

    sorted_ids = sorted(scores, key=lambda x: scores[x], reverse=True)
    return [chunk_map[cid] for cid in sorted_ids]
