"""
Embedder — abstracts OpenAI and local HuggingFace embedding models.

Design principle: the rest of the codebase never calls an embedding API directly.
Everything goes through this module. This means swapping embedding providers
only requires changing EMBEDDING_PROVIDER in .env — zero code changes elsewhere.

CRITICAL: Once you index data with a given model, you MUST use the same model
for queries. Mixing models produces meaningless similarity scores.
"""

import numpy as np
import logging
from functools import lru_cache
from app.config import settings

logger = logging.getLogger(__name__)


class Embedder:
    """
    Unified embedding interface supporting:
      - OpenAI text-embedding-3-small / text-embedding-3-large
      - Local sentence-transformers models (free, no API key)
    """

    def __init__(self):
        self.provider  = settings.EMBEDDING_PROVIDER.lower()
        self.model     = settings.EMBEDDING_MODEL
        self._client   = None
        self._st_model = None
        logger.info(f"Embedder initialised: provider={self.provider}, model={self.model}")

    # ── lazy initialisation ───────────────────────────────────────────────────

    def _get_openai_client(self):
        if self._client is None:
            from openai import OpenAI
            if not settings.OPENAI_API_KEY:
                raise RuntimeError("OPENAI_API_KEY not set.")
            self._client = OpenAI(api_key=settings.OPENAI_API_KEY)
        return self._client

    def _get_st_model(self):
        if self._st_model is None:
            from sentence_transformers import SentenceTransformer
            logger.info(f"Loading local model: {self.model}")
            self._st_model = SentenceTransformer(self.model)
        return self._st_model

    # ── core embed methods ────────────────────────────────────────────────────

    def embed_texts(self, texts: list[str]) -> np.ndarray:
        """
        Embed a list of strings.
        Returns: np.ndarray of shape (N, embedding_dim), float32, L2-normalised.
        """
        if not texts:
            return np.array([])

        if self.provider == "openai":
            return self._embed_openai(texts)
        elif self.provider == "local":
            return self._embed_local(texts)
        else:
            raise ValueError(f"Unknown embedding provider: {self.provider}")

    def embed_single(self, text: str) -> np.ndarray:
        """Embed a single string. Returns shape (embedding_dim,)."""
        result = self.embed_texts([text])
        return result[0]

    def _embed_openai(self, texts: list[str]) -> np.ndarray:
        """Call OpenAI embeddings API. Handles batching for large lists."""
        client = self._get_openai_client()
        BATCH_SIZE = 100  # OpenAI allows up to 2048, but smaller batches are safer

        all_embeddings = []
        for i in range(0, len(texts), BATCH_SIZE):
            batch = texts[i : i + BATCH_SIZE]
            # Replace empty strings — OpenAI rejects them
            batch = [t if t.strip() else "empty" for t in batch]
            response = client.embeddings.create(
                input=batch,
                model=self.model,
            )
            batch_embeddings = [d.embedding for d in response.data]
            all_embeddings.extend(batch_embeddings)

        arr = np.array(all_embeddings, dtype=np.float32)
        # L2 normalise so dot product == cosine similarity
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        return arr / np.maximum(norms, 1e-10)

    def _embed_local(self, texts: list[str]) -> np.ndarray:
        """Use sentence-transformers for local, free embedding."""
        model = self._get_st_model()
        embeddings = model.encode(
            texts,
            normalize_embeddings=True,   # L2 normalise
            show_progress_bar=len(texts) > 50,
            batch_size=32,
        )
        return embeddings.astype(np.float32)

    # ── utility ───────────────────────────────────────────────────────────────

    def cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Cosine similarity between two normalised vectors (dot product shortcut)."""
        return float(np.dot(a, b))

    def batch_cosine_similarity(
        self, query: np.ndarray, corpus: np.ndarray
    ) -> np.ndarray:
        """
        Compute cosine similarities between one query vector and N corpus vectors.
        query:  shape (dim,)
        corpus: shape (N, dim)
        Returns: shape (N,)
        """
        return corpus @ query  # equivalent to cosine_sim when both are normalised

    @property
    def dimension(self) -> int:
        """Return embedding dimensionality."""
        dims = {
            "text-embedding-3-small": 1536,
            "text-embedding-3-large": 3072,
            "all-MiniLM-L6-v2": 384,
            "all-mpnet-base-v2": 768,
            "bge-large-en-v1.5": 1024,
        }
        return dims.get(self.model, settings.EMBEDDING_DIM)


# Singleton — re-use across requests to avoid reloading models
_embedder_instance: Embedder = None

def get_embedder() -> Embedder:
    global _embedder_instance
    if _embedder_instance is None:
        _embedder_instance = Embedder()
    return _embedder_instance
