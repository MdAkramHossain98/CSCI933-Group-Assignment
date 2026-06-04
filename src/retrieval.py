"""
Embedding and retrieval utilities.

The default implementation uses sentence-transformers for embeddings
and scikit-learn cosine similarity for retrieval.
"""

from __future__ import annotations

import hashlib
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

Chunk = Dict[str, Any]

# Default cache directory — lives at <project_root>/.cache/embeddings/
_DEFAULT_CACHE_DIR = Path(__file__).resolve().parents[1] / ".cache" / "embeddings"


def _corpus_hash(chunks: List[Chunk]) -> str:
    """Short MD5 hash of all chunk texts — used to name cache files."""
    combined = "".join(c["text"] for c in chunks)
    return hashlib.md5(combined.encode("utf-8")).hexdigest()[:16]


class EmbeddingRetriever:
    """
    Embedding-based retriever with optional disk caching.

    Parameters
    ----------
    embedding_model_name : str
        Any sentence-transformers model identifier.
    cache_dir : Path or None
        Directory for cached embeddings. Pass None to disable caching.

    Usage
    -----
    retriever = EmbeddingRetriever("sentence-transformers/all-MiniLM-L6-v2")
    retriever.build_index(chunks)          # loads from cache if available
    results  = retriever.retrieve(query, top_k=5)
    """

    def __init__(
        self,
        embedding_model_name: str,
        cache_dir: Optional[Path] = _DEFAULT_CACHE_DIR,
    ) -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise ImportError(
                "sentence-transformers is required. "
                "Install with: pip install sentence-transformers"
            ) from exc

        self.model_name = embedding_model_name
        self.model      = SentenceTransformer(embedding_model_name)
        self.cache_dir  = cache_dir
        self.chunks: List[Chunk]              = []
        self.embeddings: Optional[np.ndarray] = None

    # ── Cache helpers ──────────────────────────────────────────────────────

    def _cache_paths(self, corpus_hash: str) -> Tuple[Path, Path]:
        """Return (embeddings .npy path, chunks .pkl path) for a corpus hash."""
        safe_model = self.model_name.replace("/", "_")
        base = self.cache_dir / f"{safe_model}_{corpus_hash}"
        return base.with_suffix(".npy"), base.with_suffix(".pkl")

    def _load_cache(self, corpus_hash: str) -> bool:
        """
        Try to load pre-computed embeddings and chunks from disk.
        Returns True on success, False if nothing is cached or load fails.
        """
        if self.cache_dir is None:
            return False
        emb_path, chunks_path = self._cache_paths(corpus_hash)
        if emb_path.exists() and chunks_path.exists():
            try:
                self.embeddings = np.load(str(emb_path))
                with chunks_path.open("rb") as f:
                    self.chunks = pickle.load(f)
                print(f"  [cache] Loaded embeddings from {emb_path.name}")
                return True
            except Exception as exc:
                print(f"  [cache] Load failed ({exc}) — recomputing.")
        return False

    def _save_cache(self, corpus_hash: str) -> None:
        """Persist current embeddings and chunks to disk."""
        if self.cache_dir is None or self.embeddings is None:
            return
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        emb_path, chunks_path = self._cache_paths(corpus_hash)
        np.save(str(emb_path), self.embeddings)
        with chunks_path.open("wb") as f:
            pickle.dump(self.chunks, f)
        print(f"  [cache] Saved embeddings to {emb_path.name}")

    # ── Public API ─────────────────────────────────────────────────────────

    def build_index(self, chunks: List[Chunk], force: bool = False) -> None:
        """
        Build the embedding index.

        Parameters
        ----------
        chunks : list of chunk dicts (must contain a 'text' key)
        force  : skip the cache and always recompute embeddings
        """
        if not chunks:
            raise ValueError("No chunks supplied to build_index().")

        corpus_hash = _corpus_hash(chunks)

        if not force and self._load_cache(corpus_hash):
            return  # cache hit — nothing more to do

        print(f"  [index] Encoding {len(chunks)} chunks with {self.model_name} ...")
        self.chunks = chunks
        texts = [chunk["text"] for chunk in chunks]
        self.embeddings = np.asarray(
            self.model.encode(texts, show_progress_bar=True, batch_size=64)
        )
        self._save_cache(corpus_hash)

    def retrieve(self, query: str, top_k: int = 3) -> List[Tuple[Chunk, float]]:
        """Return the top-k most relevant chunks for *query*."""
        if self.embeddings is None:
            raise RuntimeError("Index has not been built. Call build_index() first.")

        query_embedding = np.asarray(self.model.encode([query]))
        scores          = cosine_similarity(query_embedding, self.embeddings)[0]
        top_indices     = np.argsort(scores)[::-1][:top_k]
        return [(self.chunks[i], float(scores[i])) for i in top_indices]