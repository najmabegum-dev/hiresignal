"""
embeddings.py
=============
Semantic retrieval using BGE-Large embeddings + FAISS.

Used for the first stage of the pipeline: narrowing 100K candidates
down to the top 2000 semantically relevant ones.

Embeddings are cached to disk so re-runs are fast.
"""

import hashlib
import logging
from pathlib import Path
from typing import Any

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

logger = logging.getLogger(__name__)


def _file_hash(path: str) -> str:
    """MD5 hash of first 1MB of file — fast cache key."""
    h = hashlib.md5()
    with open(path, "rb") as f:
        h.update(f.read(1024 * 1024))
    return h.hexdigest()[:12]


def build_candidate_text(flat: dict[str, Any]) -> str:
    """
    Build a rich text representation of a candidate for embedding.
    Uses all high-signal fields so the embedding captures real depth.
    """
    parts = []

    # Headline and summary — gold signals, use fully
    if flat.get("headline"):
        parts.append(flat["headline"])
    if flat.get("summary"):
        parts.append(flat["summary"][:300])

    # Current title
    if flat.get("current_title"):
        parts.append(f"Current role: {flat['current_title']}")

    # Skills with proficiency
    skills_full = flat.get("skills_full", [])
    if skills_full:
        skill_strs = []
        for s in skills_full[:25]:  # cap at 25 for embedding quality
            name = s.get("name", "")
            prof = s.get("proficiency", "")
            if name:
                skill_strs.append(f"{name} ({prof})" if prof else name)
        parts.append("Skills: " + ", ".join(skill_strs))

    # Career history
    career = flat.get("career_history", [])
    for ch in career[:4]:  # top 4 roles
        title   = ch.get("title", "")
        company = ch.get("company", "")
        ind     = ch.get("industry", "")
        desc    = ch.get("description", "")[:100]
        if title:
            parts.append(f"{title} at {company} ({ind}): {desc}")

    # Education
    edu_text = flat.get("education_text", "")
    if edu_text:
        parts.append(edu_text[:150])

    # Certifications
    certs = flat.get("certifications", [])
    if certs:
        parts.append("Certifications: " + ", ".join(certs[:5]))

    return "\n".join(parts)


class EmbeddingEngine:
    """Manages text embeddings and FAISS index for semantic search."""

    def __init__(self, model_name: str, cache_dir: str = ".cache/") -> None:
        logger.info("Loading embedding model: %s", model_name)
        self.model = SentenceTransformer(model_name)
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.dim: int = self.model.get_sentence_embedding_dimension()

    def embed_text(self, text: str) -> np.ndarray:
        """Embed a single string. Returns L2-normalised float32 array."""
        vec = self.model.encode(text, normalize_embeddings=True, show_progress_bar=False)
        return vec.astype(np.float32)

    def embed_batch(
        self,
        texts: list[str],
        batch_size: int = 256,
        cache_key: str | None = None,
    ) -> np.ndarray:
        """Embed a list of texts with optional disk cache."""
        if cache_key:
            cache_path = self.cache_dir / f"{cache_key}.npy"
            if cache_path.exists():
                logger.info("Loading cached embeddings from %s", cache_path)
                return np.load(str(cache_path))

        logger.info("Embedding %d texts …", len(texts))
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            normalize_embeddings=True,
            show_progress_bar=True,
            convert_to_numpy=True,
        )
        embeddings = embeddings.astype(np.float32)

        if cache_key:
            np.save(str(cache_path), embeddings)
            logger.info("Cached embeddings to %s", cache_path)

        return embeddings

    def build_index(self, embeddings: np.ndarray) -> faiss.IndexFlatIP:
        """Build FAISS inner-product index (= cosine after L2 norm)."""
        index = faiss.IndexFlatIP(self.dim)
        index.add(embeddings)
        logger.info("FAISS index built: %d vectors.", index.ntotal)
        return index

    def search(
        self,
        query: np.ndarray,
        index: faiss.IndexFlatIP,
        k: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Search index, return (scores, indices)."""
        q = query.reshape(1, -1)
        scores, indices = index.search(q, k)
        return scores[0], indices[0]
