"""
embeddings.py
=============
Semantic retrieval using BGE-Large embeddings + FAISS.

First pipeline stage: narrow 100K candidates to the top 2000 by meaning.

Embeddings are cached to disk. The cache key is derived from the content
being embedded (model name + candidate texts), not from the input file's
bytes, so the same candidates hit the same cache whether they arrive as
.jsonl or .jsonl.gz. A loaded cache is verified (shape + spot-check) so a
stale or mismatched cache is never used silently.
"""

import hashlib
import logging
import os
from pathlib import Path
from typing import Any

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


def _file_hash(path: str) -> str:
    """Legacy helper: MD5 of the first 1MB of a file. No longer used for the cache key."""
    h = hashlib.md5()
    with open(path, "rb") as f:
        h.update(f.read(1024 * 1024))
    return h.hexdigest()[:12]


def texts_cache_key(texts: list[str], model_name: str) -> str:
    """Cache key from the model name and the exact texts being embedded."""
    h = hashlib.md5()
    h.update(model_name.encode("utf-8"))
    h.update(str(len(texts)).encode("utf-8"))
    for t in texts:
        h.update(t.encode("utf-8"))
        h.update(b"\x00")
    return f"emb_{h.hexdigest()[:16]}"


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

    def __init__(
        self,
        model_name: str,
        cache_dir: str = ".cache/",
        device: str | None = None,
    ) -> None:
        # device=None lets sentence-transformers pick (GPU if available, else CPU).
        # Override with the HIRESIGNAL_DEVICE env var, e.g. HIRESIGNAL_DEVICE=cpu
        device = device or os.environ.get("HIRESIGNAL_DEVICE") or None
        logger.info("Loading embedding model: %s (device=%s)", model_name, device or "auto")
        self.model = SentenceTransformer(model_name, device=device)
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        get_dim = getattr(self.model, "get_embedding_dimension", None) \
            or self.model.get_sentence_embedding_dimension
        self.dim: int = int(get_dim())

    def embed_text(self, text: str) -> np.ndarray:
        """Embed a single string. Returns L2-normalised float32 array."""
        vec = self.model.encode(text, normalize_embeddings=True, show_progress_bar=False)
        return vec.astype(np.float32)

    def _load_verified_cache(
        self,
        path: Path,
        texts: list[str],
        n_check: int = 16,
        min_cos: float = 0.99,
    ) -> np.ndarray | None:
        """Load a cache file only if it matches the data; otherwise return None."""
        try:
            cached = np.load(str(path))
        except Exception as e:
            logger.warning("Cache unreadable (%s) — recomputing.", e)
            return None

        expected = (len(texts), self.dim)
        if cached.shape != expected:
            logger.warning("Cache shape %s != expected %s — ignoring cache.", cached.shape, expected)
            return None

        # Spot-check: re-embed a few texts and compare with the cached rows.
        rng = np.random.default_rng(0)
        idx = rng.choice(len(texts), size=min(n_check, len(texts)), replace=False)
        fresh = self.model.encode(
            [texts[i] for i in idx],
            normalize_embeddings=True,
            show_progress_bar=False,
            convert_to_numpy=True,
        ).astype(np.float32)
        cos = np.sum(fresh * cached[idx], axis=1)
        if float(cos.min()) < min_cos:
            logger.warning(
                "Cache spot-check failed (min cosine %.4f < %.2f) — ignoring cache.",
                float(cos.min()), min_cos,
            )
            return None

        logger.info(
            "Loaded cached embeddings from %s (spot-check min cosine %.4f)",
            path, float(cos.min()),
        )
        return cached.astype(np.float32, copy=False)

    def embed_batch(
        self,
        texts: list[str],
        batch_size: int = 256,
        cache_key: str | None = None,
    ) -> np.ndarray:
        """Embed a list of texts with an optional, verified disk cache."""
        cache_path = self.cache_dir / f"{cache_key}.npy" if cache_key else None

        if cache_path is not None and cache_path.exists():
            cached = self._load_verified_cache(cache_path, texts)
            if cached is not None:
                return cached

        logger.info("Embedding %d texts …", len(texts))
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            normalize_embeddings=True,
            show_progress_bar=True,
            convert_to_numpy=True,
        )
        embeddings = embeddings.astype(np.float32)

        if cache_path is not None:
            # Atomic write: an interrupted run can't leave a corrupt cache behind.
            tmp_path = cache_path.with_name(cache_path.stem + ".tmp.npy")
            np.save(str(tmp_path), embeddings)
            os.replace(tmp_path, cache_path)
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
