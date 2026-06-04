from sentence_transformers import CrossEncoder
from typing import List, Tuple

class Reranker:
    def __init__(self):
        self.model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

    def rerank(self, query: str, chunks: List[dict], top_k: int = 5) -> List[dict]:
        pairs = [(query, c["text"]) for c in chunks]
        scores = self.model.predict(pairs)
        ranked = sorted(zip(scores, chunks), key=lambda x: x[0], reverse=True)
        return [chunk for _, chunk in ranked[:top_k]]

    def rerank_with_scores(self, query: str, chunks: List[dict], top_k: int = 5) -> Tuple[List[dict], float]:
        """Returns (top_chunks, top_score) for fallback threshold decision."""
        pairs = [(query, c["text"]) for c in chunks]
        scores = self.model.predict(pairs)
        ranked = sorted(zip(scores, chunks), key=lambda x: x[0], reverse=True)
        top_chunks = [chunk for _, chunk in ranked[:top_k]]
        top_score = float(ranked[0][0]) if ranked else -999.0
        return top_chunks, top_score
