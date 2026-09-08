"""
Local Embedding Engine for MedGemini MBBS
Uses FastEmbed (ONNX Runtime) with BAAI/bge-small-en-v1.5 (384 dimensions).
Completely local, offline-capable, and fast on standard CPU.
"""

import os
from typing import List, Union
import numpy as np
from fastembed import TextEmbedding

EMBEDDING_MODEL_NAME = "BAAI/bge-small-en-v1.5"
EMBEDDING_DIM = 384

class MBBSEmbeddingEngine:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(MBBSEmbeddingEngine, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self, model_name: str = EMBEDDING_MODEL_NAME):
        if self._initialized:
            return
        self.model_name = model_name
        self.dim = EMBEDDING_DIM
        self.threads = min(8, os.cpu_count() or 4)
        self._model = None
        self._initialized = True

    @property
    def model(self) -> TextEmbedding:
        if self._model is None:
            print(f"[*] Initializing local FastEmbed engine ({self.model_name}, threads={self.threads})...")
            self._model = TextEmbedding(model_name=self.model_name, threads=self.threads)
        return self._model

    @staticmethod
    def prepare_passage_for_embedding(book_title: str, chapter: str, topic: str, content: str, max_chars: int = 700) -> str:
        """
        Structures the text for semantic embedding with high topic density
        and bounded length for optimal CPU inference throughput.
        """
        clean_content = (content or "").strip()
        if len(clean_content) > max_chars:
            clean_content = clean_content[:max_chars]
        return f"{book_title} | {chapter} | {topic}\n{clean_content}"

    def embed_documents(self, texts: List[str], batch_size: int = 32) -> List[np.ndarray]:
        """
        Embeds a list of document passages into 384-dimensional normalized float32 vectors.
        """
        if not texts:
            return []
        embeddings = list(self.model.embed(texts, batch_size=batch_size))
        return [np.array(e, dtype=np.float32) for e in embeddings]

    def embed_query(self, query: str) -> np.ndarray:
        """
        Embeds a search query into a 384-dimensional normalized float32 vector.
        """
        query = query.strip()
        if not query:
            return np.zeros(self.dim, dtype=np.float32)
        q_emb = list(self.model.query_embed(query))[0]
        return np.array(q_emb, dtype=np.float32)

    @staticmethod
    def serialize_vector(vec: Union[np.ndarray, List[float]]) -> bytes:
        """Converts vector to raw float32 bytes for SQLite BLOB storage."""
        if isinstance(vec, list):
            vec = np.array(vec, dtype=np.float32)
        elif vec.dtype != np.float32:
            vec = vec.astype(np.float32)
        return vec.tobytes()

    @staticmethod
    def deserialize_vector(blob: bytes) -> np.ndarray:
        """Converts raw float32 bytes from SQLite BLOB back to numpy array."""
        return np.frombuffer(blob, dtype=np.float32)
