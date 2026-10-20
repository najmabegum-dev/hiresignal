#!/usr/bin/env python3
"""
precompute.py
=============
One-time PRE-COMPUTATION step. Run this BEFORE rank.py. Network access and
extra time are expected here; the timed ranking step (rank.py) is separate.

It does three things:
  1. Downloads and caches the reranker weights.
  2. Downloads and caches the embedding model weights.
  3. Embeds every candidate and saves the embeddings to .cache/
     (roughly 2+ hours on a 2-vCPU CPU, ~1.5 hours on a free T4 GPU).
     If a valid cache already exists, this step is skipped automatically.

After this, rank.py can run fully offline:
    HF_HUB_OFFLINE=1 python rank.py --candidates data/candidates.jsonl --out output/submission.csv

Usage:
    python precompute.py --candidates data/candidates.jsonl
    python precompute.py --candidates data/candidates.jsonl.gz
"""

import argparse
import logging

import yaml

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)


def main() -> None:
    parser = argparse.ArgumentParser(description="HireSignal pre-computation (models + embeddings)")
    parser.add_argument("--candidates", required=True)
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    from sentence_transformers import CrossEncoder
    from src.embeddings import EmbeddingEngine, build_candidate_text, texts_cache_key
    from src.loader import flatten_candidate, load_candidates

    print("[1/3] Caching reranker model ...")
    CrossEncoder(config["model"]["reranker_model"])

    print("[2/3] Caching embedding model ...")
    cache_dir = config.get("paths", {}).get("cache_dir", ".cache/")
    engine = EmbeddingEngine(config["model"]["embedding_model"], cache_dir=cache_dir)

    print("[3/3] Embedding candidates (skipped if a valid cache exists) ...")
    raw = load_candidates(args.candidates)
    texts = [build_candidate_text(flatten_candidate(c)) for c in raw]
    key = texts_cache_key(texts, config["model"]["embedding_model"])
    engine.embed_batch(
        texts,
        batch_size=config["pipeline"]["embed_batch_size"],
        cache_key=key,
    )
    print(f"Done. Cache file: {cache_dir}{key}.npy")


if __name__ == "__main__":
    main()
