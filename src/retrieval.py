import chromadb
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
from typing import List

class VectorStore:
    def __init__(self, collection_name: str = "documind"):
        self.client = chromadb.PersistentClient(path="./chroma_db")
        self.collection = self.client.get_or_create_collection(collection_name)
        self.embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    def add_chunks(self, chunks: List[dict]) -> None:
        texts = [c["text"] for c in chunks]
        embeddings = self.embedder.encode(texts).tolist()
        self.collection.add(
            documents=texts,
            embeddings=embeddings,
            ids=[c["chunk_id"] for c in chunks],
            metadatas=[{"page": c["page"]} for c in chunks]
        )

    def search(self, query: str, k: int = 10) -> List[dict]:
        q_emb = self.embedder.encode([query]).tolist()
        results = self.collection.query(query_embeddings=q_emb, n_results=k)
        return [{"text": d, "page": m["page"], "id": i}
                for d, m, i in zip(results["documents"][0],
                                   results["metadatas"][0],
                                   results["ids"][0])]

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
        cid = chunk["id"]
        scores[cid] = scores.get(cid, 0) + 1 / (k + rank + 1)
        chunk_map[cid] = chunk
    for rank, chunk in enumerate(bm25_results):
        cid = chunk["chunk_id"]
        scores[cid] = scores.get(cid, 0) + 1 / (k + rank + 1)
        chunk_map[cid] = chunk
    sorted_ids = sorted(scores, key=lambda x: scores[x], reverse=True)
    return [chunk_map[cid] for cid in sorted_ids]
