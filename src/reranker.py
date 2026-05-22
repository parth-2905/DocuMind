from sentence_transformers import CrossEncoder
from typing import List

class Reranker:
    def __init__(self):
        self.model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

    def rerank(self, query: str, chunks: List[dict], top_k: int = 5) -> List[dict]:
        pairs = [(query, c["text"]) for c in chunks]
        scores = self.model.predict(pairs)
        ranked = sorted(zip(scores, chunks), key=lambda x: x[0], reverse=True)
        return [chunk for _, chunk in ranked[:top_k]]
