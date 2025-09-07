# pip install sentence-transformers numpy
from typing import List
import numpy as np
from sentence_transformers import SentenceTransformer

class EmbeddingScorer:
    """
    Fast semantic similarity vs gold answer, mapped to 0..10.
    """
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2", device: str | None = None):
        self.model = SentenceTransformer(model_name, device=device)

    @staticmethod
    def _cosine(a: np.ndarray, b: np.ndarray) -> float:
        denom = float(np.linalg.norm(a) * np.linalg.norm(b) + 1e-12)
        return float(np.dot(a, b) / denom)

    def score(self, response: str, gold: str) -> float:
        vecs = self.model.encode([response, gold], normalize_embeddings=True)
        sim = self._cosine(vecs[0], vecs[1])  # 0..1
        return round(sim * 10.0, 2)
