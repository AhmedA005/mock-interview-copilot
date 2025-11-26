from __future__ import annotations

from typing import List, Sequence, Tuple

import faiss
from sentence_transformers import SentenceTransformer

EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def chunk_text(text: str, chunk_size: int = 220, overlap: int = 40) -> List[str]:
    words = text.split()
    chunks: List[str] = []
    for i in range(0, len(words), max(1, chunk_size - overlap)):
        chunk = " ".join(words[i : i + chunk_size])
        if chunk.strip():
            chunks.append(chunk)
    return chunks


class Embedder:
    def __init__(self, model_name: str = EMBED_MODEL_NAME) -> None:
        self.model = SentenceTransformer(model_name)
        self.index = None
        self.chunks: List[str] = []

    def fit(self, chunks: Sequence[str]) -> None:
        self.chunks = list(chunks)
        if not self.chunks:
            raise ValueError("No chunks provided for embedding.")
        embeddings = self.model.encode(self.chunks, convert_to_numpy=True)
        dim = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dim)
        self.index.add(embeddings)

    def relevance(self, text: str, k: int = 3) -> float:
        if self.index is None:
            raise RuntimeError("Embedder not fitted.")
        query_emb = self.model.encode([text], convert_to_numpy=True)
        distances, _ = self.index.search(query_emb, k)
        avg_dist = float(distances.mean())
        score = 1 / (1 + avg_dist)
        return max(0.0, min(1.0, score))

