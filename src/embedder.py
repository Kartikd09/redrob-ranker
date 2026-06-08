# src/embedder.py
from typing import List
import numpy as np

_MODEL_NAME = "BAAI/bge-m3"  # 1024-dim SOTA retrieval model


def get_model():
    from fastembed import TextEmbedding
    return TextEmbedding(model_name=_MODEL_NAME)


def embed_texts(texts: List[str], batch_size: int = 64) -> np.ndarray:
    """
    Embed list of texts. Returns float32 array, L2-normalized.
    BGE-M3: 1024-dim, SOTA multilingual retrieval (2024).
    """
    model = get_model()
    embeddings = list(model.embed(texts, batch_size=batch_size))
    arr = np.array(embeddings, dtype=np.float32)
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    return arr / norms
