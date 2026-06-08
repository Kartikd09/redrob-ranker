# src/faiss_index.py
import numpy as np
import faiss
from typing import Tuple


def build_index(embeddings: np.ndarray) -> faiss.IndexFlatIP:
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    return index


def save_index(index: faiss.IndexFlatIP, path: str) -> None:
    faiss.write_index(index, path)


def load_index(path: str) -> faiss.IndexFlatIP:
    return faiss.read_index(path)


def search_index(
    index: faiss.IndexFlatIP,
    query_embedding: np.ndarray,
    top_k: int = 1000,
) -> Tuple[np.ndarray, np.ndarray]:
    """Returns (scores, indices) arrays of shape (top_k,)."""
    query = query_embedding.reshape(1, -1).astype(np.float32)
    scores, indices = index.search(query, top_k)
    return scores[0], indices[0]
