"""Local embedding retrieval with sentence-transformers + numpy cosine."""
from dataclasses import dataclass

import numpy as np
from sentence_transformers import SentenceTransformer

_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
_model = None


def _get_model():
    global _model
    if _model is None:
        _model = SentenceTransformer(_MODEL_NAME)
    return _model


@dataclass
class Index:
    matrix: np.ndarray  # shape (N, d), L2-normalized
    chunks: list


def _embed(texts):
    model = _get_model()
    vecs = model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
    return np.asarray(vecs, dtype=np.float32)


def build_index(chunks):
    if not chunks:
        raise RuntimeError("Cannot build index over empty chunk list")
    matrix = _embed([c["text"] for c in chunks])
    return Index(matrix=matrix, chunks=chunks)


def retrieve(index, question, k=5):
    q = _embed([question])[0]
    sims = index.matrix @ q  # cosine since both sides L2-normalized
    k = min(k, len(index.chunks))
    top_idx = np.argsort(-sims)[:k]
    results = []
    for rank, idx in enumerate(top_idx, start=1):
        c = index.chunks[int(idx)]
        results.append({
            "rank": rank,
            "chunk_id": c["chunk_id"],
            "doc_title": c["doc_title"],
            "score": float(sims[int(idx)]),
            "chunk_text": c["text"],
        })
    return results
