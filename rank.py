#!/usr/bin/env python3
"""
rank.py
=======
HireSignal — India Runs Track 1 submission entrypoint.

Produces the ranked top-100 CSV from the hackathon dataset.

Usage:
    python rank.py --candidates data/candidates.jsonl --out output/submission.csv
    python rank.py --candidates data/sample_candidates.json --out output/test.csv --sample

Pipeline: BGE-Large + FAISS semantic retrieval -> 4-signal scoring with honeypot
detection -> cross-encoder reranking -> template-based reasoning.

Measured runtime on 100,000 candidates:
  - Warm embedding cache (.cache/*.npy present): ~1m49s on Colab CPU, ~82s on T4 GPU
  - Cold CPU, no cache: embedding alone was projected at ~2h20m (BGE-Large, 2 vCPU)
The embedding cache is NOT stored in this repository, so a fresh clone starts
cold. Network access is needed on the first run to download model weights.
"""
import argparse
import csv
import logging
import sys
import time
from pathlib import Path

import numpy as np
import yaml
from colorama import Fore, Style, init as colorama_init
from tqdm import tqdm

colorama_init(autoreset=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ── Imports from src ─────────────────────────────────────────────────────────
from src.loader    import load_candidates, flatten_candidate
from src.scorer    import CandidateScorer
from src.honeypot  import get_honeypot_flags, is_excluded, apply_flag_cap
from src.reasoning import generate_reasoning


def banner() -> None:
    print(Fore.MAGENTA + Style.BRIGHT + """
  _   _ _           ____  _                   _
 | | | (_)_ __ ___ / ___|(_) __ _ _ __   __ _| |
 | |_| | | '__/ _ \\___ \\| |/ _` | '_ \\ / _` | |
 |  _  | | | |  __/___) | | (_| | | | | (_| | |
 |_| |_|_|_|  \\___|____/|_|\\__, |_| |_|\\__,_|_|
                            |___/
""" + Style.RESET_ALL)
    print(Fore.CYAN + "  India Runs Track 1 — Redrob AI Hackathon" + Style.RESET_ALL)
    print(Fore.YELLOW + "  Submission deadline: 28 June 2026\n" + Style.RESET_ALL)


def load_config(path: str = "config.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def load_jd(jd_path: str) -> str:
    p = Path(jd_path)
    if not p.exists():
        raise FileNotFoundError(f"JD file not found: {jd_path}")
    return p.read_text(encoding="utf-8").strip()


def save_csv(results: list[dict], output_path: str) -> None:
    """Save exactly 100 rows in the required format, validated."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

       # Sort by rounded score (descending), then candidate_id (ascending).
    # Rounding first matters: the CSV stores 4 decimals, so ties must be
    # judged on the same rounded value the validator sees.
    for r in results:
        r["score"] = round(r["score"], 4)
    results.sort(key=lambda x: (-x["score"], x["candidate_id"]))
    for i, r in enumerate(results):
        r["rank"] = i + 1

    # Verify non-increasing scores
    for i in range(len(results) - 1):
        if results[i]["score"] < results[i + 1]["score"]:
            logger.warning("Score ordering issue at rank %d — fixing…", i + 1)
            # Already sorted, this shouldn't happen

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["candidate_id", "rank", "score", "reasoning"])
        for r in results:
            writer.writerow([
                r["candidate_id"],
                r["rank"],
                f"{r['score']:.4f}",
                r["reasoning"],
            ])

    logger.info("Saved %d rows to %s", len(results), output_path)


def run_semantic_stage(
    flat_candidates: list[dict],
    jd_text: str,
    config: dict,
    candidates_path: str,
) -> list[int]:
    """
    Stage 1: Use BGE embeddings + FAISS to narrow 100K → top 2000.
    Returns list of indices into flat_candidates.
    """
    from src.embeddings import EmbeddingEngine, build_candidate_text, texts_cache_key

    cfg = config["model"]
    top_k = config["pipeline"]["top_k_semantic"]
    batch_size = config["pipeline"]["embed_batch_size"]
    cache_dir = config.get("paths", {}).get("cache_dir", ".cache/")

    print(Fore.CYAN + f"\n[Stage 1] Semantic retrieval — embedding {len(flat_candidates):,} candidates…" + Style.RESET_ALL)

    engine = EmbeddingEngine(cfg["embedding_model"], cache_dir=cache_dir)

    # Build cache key from file hash
    try:
        from src.embeddings import _file_hash
        cache_key = f"candidates_{_file_hash(candidates_path)}"
    except Exception:
        cache_key = None

    # Embed all candidates
    texts = [build_candidate_text(f) for f in flat_candidates]
    candidate_embeddings = engine.embed_batch(texts, batch_size=batch_size, cache_key=cache_key)

    # Build FAISS index
    index = engine.build_index(candidate_embeddings)

    # Embed JD query
    jd_embedding = engine.embed_text(jd_text)

    # Search
    k = min(top_k, len(flat_candidates))
    _, indices = engine.search(jd_embedding, index, k=k)

    valid_indices = [int(i) for i in indices if 0 <= i < len(flat_candidates)]
    print(Fore.GREEN + f"  Semantic stage returned {len(valid_indices):,} candidates." + Style.RESET_ALL)
    return valid_indices


def run_scoring_stage(
    flat_candidates: list[dict],
    raw_candidates: list[dict],
    indices: list[int],
    config: dict,
) -> list[dict]:
    """
    Stage 2: Multi-signal scoring on semantic top-K.
    Applies honeypot detection, behavioral + career scoring.
    Returns list of scored result dicts.
    """
    print(Fore.CYAN + f"\n[Stage 2] Scoring {len(indices):,} candidates…" + Style.RESET_ALL)

    scorer = CandidateScorer(config)
    hp_cfg = config.get("honeypot", {})
    exclude_threshold = hp_cfg.get("exclude_threshold", 2)
    flag_cap = hp_cfg.get("single_flag_cap", 0.40)

    results = []
    excluded = 0
    capped = 0

    for idx in tqdm(indices, desc="Scoring"):
        flat = flat_candidates[idx]
        raw  = raw_candidates[idx]

        # Honeypot detection
        flags = get_honeypot_flags(flat, raw)
        if is_excluded(flags, threshold=exclude_threshold):
            excluded += 1
            continue

        # Score
        score_dict = scorer.score(flat)
        composite  = score_dict["composite_score"]

        # Apply flag cap if 1 flag
        composite = apply_flag_cap(composite, flags, cap=flag_cap)
        if flags:
            capped += 1

        score_dict["composite_score"] = composite
        results.append({
            "candidate_id": flat["candidate_id"],
            "flat":         flat,
            "scores":       score_dict,
            "score":        composite,
            "flags":        flags,
        })

    print(Fore.YELLOW + f"  Excluded {excluded} honeypots, capped {capped} flagged candidates." + Style.RESET_ALL)
    print(Fore.GREEN  + f"  {len(results):,} clean candidates scored." + Style.RESET_ALL)
    return results


def run_reranking_stage(results: list[dict], jd_text: str, config: dict) -> list[dict]:
    """
    Stage 3: Cross-encoder reranking on top-300.
    Refines ordering of the best candidates.
    """
    top_k_rerank = config["pipeline"]["top_k_rerank"]

    # Sort by composite, take top for reranking
    results.sort(key=lambda x: x["score"], reverse=True)
    rerank_pool = results[:top_k_rerank]
    rest        = results[top_k_rerank:]

    print(Fore.CYAN + f"\n[Stage 3] Cross-encoder reranking top-{len(rerank_pool)}…" + Style.RESET_ALL)

    try:
        from sentence_transformers import CrossEncoder
        from src.embeddings import build_candidate_text

        reranker = CrossEncoder(config["model"]["reranker_model"])

        pairs = [
            [jd_text[:512], build_candidate_text(r["flat"])[:512]]
            for r in rerank_pool
        ]
        rerank_scores = reranker.predict(pairs, show_progress_bar=True)

        # Blend: 60% reranker + 40% composite
        for r, rs in zip(rerank_pool, rerank_scores):
            blended = 0.60 * float(rs) + 0.40 * r["score"]
            # Normalise reranker score to ~0-1 range (cross-encoder outputs vary)
            # MiniLM outputs roughly -10 to +10; sigmoid-normalise
            import math
            sigmoid_rs = 1.0 / (1.0 + math.exp(-float(rs) * 0.5))
            blended = 0.60 * sigmoid_rs + 0.40 * r["score"]
            r["score"] = round(blended, 4)

        print(Fore.GREEN + "  Reranking complete." + Style.RESET_ALL)

    except Exception as e:
        logger.warning("Reranker failed (%s) — using composite scores.", e)

    # Merge back
    all_results = rerank_pool + rest
    all_results.sort(key=lambda x: x["score"], reverse=True)
    return all_results


def add_reasoning(results: list[dict]) -> list[dict]:
    """Stage 4: Generate reasoning for top 100."""
    print(Fore.CYAN + "\n[Stage 4] Generating reasoning…" + Style.RESET_ALL)
    for r in tqdm(results[:100], desc="Reasoning"):
        r["reasoning"] = generate_reasoning(r["flat"], r["scores"], r["rank"] if "rank" in r else 0)
    return results


# ── MAIN ─────────────────────────────────────────────────────────────────────

def main() -> None:
    banner()

    parser = argparse.ArgumentParser(
        description="HireSignal — rank candidates for India Runs Track 1",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--candidates", required=True,
                        help="Path to candidates.jsonl.gz or sample_candidates.json")
    parser.add_argument("--out", default="output/submission.csv",
                        help="Output CSV path (default: output/submission.csv)")
    parser.add_argument("--jd",  default="data/job_description.txt",
                        help="Path to job description text file")
    parser.add_argument("--config", default="config.yaml",
                        help="Path to config.yaml")
    parser.add_argument("--sample", action="store_true",
                        help="Sample mode — skip semantic stage, score all directly")
    parser.add_argument("--no-rerank", action="store_true",
                        help="Skip cross-encoder reranking (faster, less accurate)")
    args = parser.parse_args()

    t0 = time.time()

    # ── Load ─────────────────────────────────────────────────────────
    config = load_config(args.config)

    print(Fore.CYAN + f"→ Loading JD from: {args.jd}" + Style.RESET_ALL)
    jd_text = load_jd(args.jd)
    print(Fore.GREEN + f"  JD loaded ({len(jd_text)} chars)" + Style.RESET_ALL)

    print(Fore.CYAN + f"\n→ Loading candidates from: {args.candidates}" + Style.RESET_ALL)
    raw_candidates = load_candidates(args.candidates)
    print(Fore.GREEN + f"  {len(raw_candidates):,} candidates loaded." + Style.RESET_ALL)

    # ── Flatten ───────────────────────────────────────────────────────
    print(Fore.CYAN + "\n→ Flattening candidate schemas…" + Style.RESET_ALL)
    flat_candidates = [flatten_candidate(c) for c in tqdm(raw_candidates, desc="Flattening")]

    # ── Stage 1: Semantic retrieval ────────────────────────────────────
    if args.sample or len(raw_candidates) <= 200:
        # For small samples, score all directly
        print(Fore.YELLOW + "  Small dataset — scoring all candidates directly." + Style.RESET_ALL)
        semantic_indices = list(range(len(flat_candidates)))
    else:
        semantic_indices = run_semantic_stage(
            flat_candidates, jd_text, config, args.candidates
        )

    # ── Stage 2: Multi-signal scoring ─────────────────────────────────
    scored_results = run_scoring_stage(
        flat_candidates, raw_candidates, semantic_indices, config
    )

    if len(scored_results) < 100:
        logger.warning(
            "Only %d clean candidates after honeypot removal — "
            "consider reducing exclusion threshold.", len(scored_results)
        )
        if len(scored_results) < 100:
            print(Fore.RED + f"ERROR: Only {len(scored_results)} candidates. Need 100." + Style.RESET_ALL)
            sys.exit(1)

    # ── Stage 3: Reranking ─────────────────────────────────────────────
    if not args.no_rerank and not args.sample:
        scored_results = run_reranking_stage(scored_results, jd_text, config)
    else:
        scored_results.sort(key=lambda x: x["score"], reverse=True)

    # ── Stage 4: Take top 100 + add reasoning ─────────────────────────
    for r in scored_results:
        r["score"] = round(r["score"], 4)
    scored_results.sort(key=lambda x: (-x["score"], x["candidate_id"]))
    top100 = scored_results[:100]
    for i, r in enumerate(top100):
        r["rank"] = i + 1

    top100 = add_reasoning(top100)

    # ── Save output ────────────────────────────────────────────────────
    final = [
        {
            "candidate_id": r["candidate_id"],
            "rank":         r["rank"],
            "score":        r["score"],
            "reasoning":    r.get("reasoning", ""),
        }
        for r in top100
    ]

    print(Fore.CYAN + f"\n→ Saving output to: {args.out}" + Style.RESET_ALL)
    save_csv(final, args.out)

    # ── Summary ────────────────────────────────────────────────────────
    elapsed = time.time() - t0
    print()
    print(Fore.GREEN + Style.BRIGHT + "✓ Done!" + Style.RESET_ALL)
    print(f"  Candidates processed : {len(raw_candidates):,}")
    print(f"  Top 100 ranked       : {len(final)}")
    print(f"  Total time           : {elapsed:.1f}s ({elapsed/60:.1f} min)")
    print(f"  Output               : {args.out}")
    print()
    print(Fore.YELLOW + "  Top 5 candidates:" + Style.RESET_ALL)
    for r in top100[:5]:
        print(f"    #{r['rank']}  {r['candidate_id']}  score={r['score']:.4f}")
        print(f"       {r.get('reasoning', '')[:90]}")
    print()
    print(Fore.CYAN + "  Next step: python validate_submission.py " + args.out + Style.RESET_ALL)


if __name__ == "__main__":
    main()
