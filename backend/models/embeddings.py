"""
Embedding model management module.
Handles text embeddings using sentence transformers.
"""

from typing import List, Optional, Tuple

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from ..config import settings


class EmbeddingManager:
    """Manages embedding model and FAISS index operations."""

    _instance: Optional["EmbeddingManager"] = None
    _model: Optional[SentenceTransformer] = None

    def __new__(cls) -> "EmbeddingManager":
        """Singleton pattern for the embedding model."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    @property
    def is_loaded(self) -> bool:
        """Check if the embedding model is loaded."""
        return self._model is not None

    def load(self) -> SentenceTransformer:
        """Load or return the cached embedding model."""
        if self._model is None:
            print(f"🔎 Loading embedding model: {settings.EMBED_MODEL_NAME}...")
            self._model = SentenceTransformer(settings.EMBED_MODEL_NAME)
            print("✅ Embedding model loaded")
        return self._model

    def encode(self, texts: List[str]) -> np.ndarray:
        """Encode texts into embeddings."""
        model = self.load()
        return model.encode(texts, convert_to_numpy=True)

    def create_index(self, embeddings: np.ndarray) -> faiss.IndexFlatL2:
        """Create a FAISS index from embeddings."""
        dim = embeddings.shape[1]
        index = faiss.IndexFlatL2(dim)
        index.add(embeddings)
        return index

    def search(
        self,
        query: str,
        index: faiss.IndexFlatL2,
        chunks: List[str],
        k: int = 5,
    ) -> Tuple[np.ndarray, List[str]]:
        """Search for similar chunks using the FAISS index."""
        model = self.load()
        query_embedding = model.encode([query], convert_to_numpy=True)
        distances, indices = index.search(query_embedding, k)
        matched_chunks = [chunks[i] for i in indices[0]]
        return distances[0], matched_chunks

    def compute_relevance(
        self,
        question: str,
        index: faiss.IndexFlatL2,
        chunks: List[str],
        k: int = 3,
    ) -> float:
        """Compute relevance score for a question based on resume chunks."""
        distances, _ = self.search(question, index, chunks, k=k)
        avg_dist = float(distances.mean())
        score = 1 / (1 + avg_dist)
        return min(max(score, 0.0), 1.0)


# Global instance for easy access
embedding_manager = EmbeddingManager()
