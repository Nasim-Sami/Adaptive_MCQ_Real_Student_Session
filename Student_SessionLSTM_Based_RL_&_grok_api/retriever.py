"""
retriever.py — Embed PDF chunks with Ollama nomic-embed-text and expose
cosine-similarity retrieval.

Usage (one-time embedding build):
    python retriever.py --build

Then import and call from other modules:
    from retriever import retrieve
    chunks = retrieve("PID control", "what is integral windup", k=5)

Depends on:
    - pdf_chunks.json  (produced by pdf_ingest.py)
    - Ollama running locally with nomic-embed-text pulled
    - numpy
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE = Path(__file__).parent
CHUNKS_FILE = BASE / "pdf_chunks.json"
EMBEDDINGS_FILE = BASE / "pdf_embeddings.npy"
META_FILE = BASE / "pdf_embeddings_meta.json"

OLLAMA_URL = "http://localhost:11434/api/embeddings"
EMBED_MODEL = "nomic-embed-text"
BATCH_SIZE = 16          # chunks per Ollama call (avoids timeouts)
TOPIC_K = 8              # chunks pre-fetched per topic in the topic map


# ---------------------------------------------------------------------------
# Ollama embedding helper
# ---------------------------------------------------------------------------
def _embed_text(texts: list[str]) -> list[list[float]]:
    import requests
    vectors: list[list[float]] = []
    for i in range(0, len(texts), BATCH_SIZE):
        batch = texts[i: i + BATCH_SIZE]
        for text in batch:
            resp = requests.post(
                OLLAMA_URL,
                json={"model": EMBED_MODEL, "prompt": text},
                timeout=60,
            )
            resp.raise_for_status()
            vectors.append(resp.json()["embedding"])
        if i + BATCH_SIZE < len(texts):
            time.sleep(0.05)   # be gentle with local Ollama
    return vectors


def _cosine_sim(query_vec: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """Return cosine similarities between query_vec (D,) and matrix (N, D)."""
    q = query_vec / (np.linalg.norm(query_vec) + 1e-10)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True) + 1e-10
    return (matrix / norms) @ q


# ---------------------------------------------------------------------------
# Build (one-time)
# ---------------------------------------------------------------------------
def build_embeddings() -> None:
    """Embed all chunks and cache to disk."""
    if not CHUNKS_FILE.exists():
        raise FileNotFoundError(f"{CHUNKS_FILE} not found. Run pdf_ingest.py first.")

    chunks: list[dict] = json.loads(CHUNKS_FILE.read_text(encoding="utf-8"))
    texts = [c["text"] for c in chunks]
    print(f"Embedding {len(texts)} chunks with {EMBED_MODEL} ...")

    vectors = _embed_text(texts)
    matrix = np.array(vectors, dtype=np.float32)

    np.save(EMBEDDINGS_FILE, matrix)
    META_FILE.write_text(
        json.dumps({"n_chunks": len(chunks), "model": EMBED_MODEL}, indent=2),
        encoding="utf-8",
    )
    print(f"Saved {EMBEDDINGS_FILE} ({matrix.shape}) and {META_FILE}")


# ---------------------------------------------------------------------------
# Runtime retrieval
# ---------------------------------------------------------------------------
_MATRIX: np.ndarray | None = None
_CHUNKS: list[dict] | None = None
_TOPIC_MAP: dict[str, list[int]] | None = None   # topic -> list of top chunk indices


def _load() -> None:
    global _MATRIX, _CHUNKS
    if _MATRIX is not None:
        return
    if not EMBEDDINGS_FILE.exists():
        raise RuntimeError(
            "Embeddings not built. Run: python retriever.py --build"
        )
    _MATRIX = np.load(EMBEDDINGS_FILE)
    _CHUNKS = json.loads(CHUNKS_FILE.read_text(encoding="utf-8"))


def retrieve(topic: str, query: str, k: int = 5) -> list[dict[str, Any]]:
    """
    Return the top-k PDF chunks most relevant to (topic + query).

    Each result dict has keys: book, page, chunk_index, text, score.
    Falls back gracefully to empty list if embeddings are unavailable.
    """
    try:
        _load()
    except RuntimeError:
        return []

    combined_query = f"{topic}: {query}"
    try:
        import requests
        resp = requests.post(
            OLLAMA_URL,
            json={"model": EMBED_MODEL, "prompt": combined_query},
            timeout=15,
        )
        resp.raise_for_status()
        q_vec = np.array(resp.json()["embedding"], dtype=np.float32)
    except Exception:
        return []

    sims = _cosine_sim(q_vec, _MATRIX)
    top_idx = np.argsort(sims)[::-1][:k]

    results: list[dict] = []
    for idx in top_idx:
        chunk = dict(_CHUNKS[idx])
        chunk["score"] = float(sims[idx])
        results.append(chunk)
    return results


def retrieve_for_topic(topic: str, k: int = TOPIC_K) -> list[dict[str, Any]]:
    """Convenience: retrieve top-k chunks for a topic name used as the query."""
    return retrieve(topic, topic, k=k)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--build", action="store_true", help="Build embeddings cache.")
    p.add_argument("--query", type=str, default=None, help="Test query.")
    p.add_argument("--topic", type=str, default="PID control")
    p.add_argument("--k", type=int, default=5)
    args = p.parse_args()

    if args.build:
        build_embeddings()
    if args.query:
        results = retrieve(args.topic, args.query, k=args.k)
        for r in results:
            print(f"\n[{r['book']} p{r['page']} score={r['score']:.3f}]")
            print(r["text"][:300])
