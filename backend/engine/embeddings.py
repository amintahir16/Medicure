"""
Local Biomedical Embedding Engine for Medicure MBBS
Uses PubMedBERT (NeuML/pubmedbert-base-embeddings, 768 dimensions) via SentenceTransformers.
Completely local, offline-capable, and optimized for CPU inference with native MeSH/UMLS biomedical vocabulary.
"""

import os
from typing import List, Union
try:
    from sentence_transformers import SentenceTransformer
    HAS_SENTENCE_TRANSFORMERS = True
except (ImportError, Exception):
    SentenceTransformer = None
    HAS_SENTENCE_TRANSFORMERS = False

EMBEDDING_MODEL_NAME = "NeuML/pubmedbert-base-embeddings"
EMBEDDING_DIM = 768

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
        self._model = None
        self._initialized = True

    @property
    def model(self):
        if not HAS_SENTENCE_TRANSFORMERS:
            return None
        if self._model is None:
            print(f"[*] Initializing local PubMedBERT biomedical engine ({self.model_name}, device=cpu)...")
            self._model = SentenceTransformer(self.model_name, device="cpu")
        return self._model

    @staticmethod
    def prepare_passage_for_embedding(book_title: str, chapter: str, topic: str, content: str, max_chars: int = 2200) -> str:
        """
        Structures the text for semantic embedding with high topic density
        and bounded length utilizing the full 512-token (~2000-2200 chars) context window
        of PubMedBERT without prematurely truncating clinical passages.
        """
        clean_content = (content or "").strip()
        if len(clean_content) > max_chars:
            # Cut at sentence boundary if possible
            truncated = clean_content[:max_chars]
            last_period = truncated.rfind('.')
            if last_period > max_chars * 0.7:
                clean_content = truncated[:last_period + 1]
            else:
                clean_content = truncated
        return f"{book_title} | {chapter} | {topic}\n{clean_content}"

    def embed_documents(self, texts: List[str], batch_size: int = 32) -> List[np.ndarray]:
        """
        Embeds a list of document passages into 768-dimensional normalized float32 vectors.
        """
        if not texts:
            return []
        if self.model is None:
            return [np.zeros(self.dim, dtype=np.float32) for _ in texts]
        embeddings = self.model.encode(
            texts, 
            batch_size=batch_size, 
            show_progress_bar=False, 
            normalize_embeddings=True,
            convert_to_numpy=True
        )
        return [np.array(e, dtype=np.float32) for e in embeddings]

    def embed_query(self, query: str) -> np.ndarray:
        """
        Embeds a search query into a 768-dimensional normalized float32 vector.
        """
        query = query.strip()
        if not query or self.model is None:
            return np.zeros(self.dim, dtype=np.float32)
        vec = self.model.encode(
            query, 
            show_progress_bar=False, 
            normalize_embeddings=True, 
            convert_to_numpy=True
        )
        return np.array(vec, dtype=np.float32)

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
