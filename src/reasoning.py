"""
reasoning.py
============
Generates the 'reasoning' column for the submission CSV.

Rules from the spec's Stage 4 review, and how this module meets them:
  - Specific facts       -> title, years of experience, named skills, signal values
  - JD connection        -> skills are mapped to named JD requirement areas
                            (vector search, embeddings, ranking, LLM/RAG, MLOps, NLP);
                            the text says which areas a candidate covers and which
                            have no listed skills
  - Honest concerns      -> notice period, inactivity, response rate, experience band
  - No hallucination     -> only values present in the candidate record
  - Variation            -> phrasing is chosen deterministically from the candidate_id,
                            so identical input always gives identical output
  - Rank consistency     -> a verdict word and structure that follow the rank band

All of this is done offline. There are no API or LLM calls.
"""

import hashlib
import re
from datetime import datetime
from typing import Any

from src.refdate import today as ref_today

# Core JD skills (kept from the original module; used for the "non-JD skills" list).
CORE_JD_SKILLS = {
    "embeddings", "faiss", "pinecone", "weaviate", "qdrant", "milvus",
    "sentence-transformers", "sentence transformers", "bge", "e5",
    "information retrieval", "retrieval", "ranking", "vector",
    "recommendation", "nlp", "natural language processing",
    "learning to rank", "ndcg", "hybrid search", "semantic search",
    "rag", "fine-tuning llms", "fine-tuning", "lora", "qlora",
    "pytorch", "hugging face", "hugging face transformers",
    "mlops", "mlflow", "a/b testing", "reranking", "re-ranking",
}

# The same vocabulary, grouped into named JD requirement areas.
JD_AREAS: dict[str, set[str]] = {
    "vector search / retrieval": {
        "faiss", "pinecone", "weaviate", "qdrant", "milvus", "vector",
        "retrieval", "information retrieval", "semantic search", "hybrid search",
    },
    "embeddings": {"embeddings", "sentence-transformers", "sentence transformers", "bge", "e5"},
    "ranking / recommendation": {
        "ranking", "learning to rank", "ndcg", "reranking", "re-ranking", "recommendation",
    },
    "LLM / RAG": {
        "rag", "fine-tuning llms", "fine-tuning", "lora", "qlora",
        "hugging face", "hugging face transformers",
    },
    "MLOps / experimentation": {"mlops", "mlflow", "a/b testing"},
    "NLP / deep learning": {"nlp", "natural language processing", "pytorch"},
}

_PROF_RANK = {"expert": 3, "advanced": 2, "intermediate": 1, "beginner": 0}


def _term_in(term: str, text: str) -> bool:
    """Whole-word match, so 'rag' does not match 'storage' and 'e5' does not match 'ge5x'."""
    return re.search(r"(?<![a-z0-9])" + re.escape(term) + r"(?![a-z0-9])", text) is not None


def _skill_matches(name_lower: str, terms: set[str]) -> bool:
    return bool(name_lower) and any(_term_in(t, name_lower) for t in terms)


def _pick(flat: dict[str, Any], options: list[str]) -> str:
    """Deterministic, input-based choice among phrasings (no randomness)."""
    key = str(flat.get("candidate_id", "")) or str(flat.get("current_title", ""))
    idx = int(hashlib.md5(key.encode("utf-8")).hexdigest(), 16) % len(options)
    return options[idx]


def _matched_skills(flat: dict[str, Any]) -> list[tuple[str, str, float]]:
    out = []
    for s in flat.get("skills_full", []):
        name = s.get("name", "")
        if _skill_matches(name.lower(), CORE_JD_SKILLS):
            out.append((name, s.get("proficiency", ""), s.get("duration_months", 0) or 0))
    out.sort(key=lambda x: (_PROF_RANK.get(x[1], 0), x[2]), reverse=True)
    return out


def _top_jd_skills(flat: dict[str, Any], n: int = 3) -> list[str]:
    """Top N JD-relevant skills with proficiency, e.g. 'FAISS (advanced)'."""
    return [f"{m[0]} ({m[1]})" if m[1] else m[0] for m in _matched_skills(flat)[:n]]


def _area_coverage(flat: dict[str, Any]) -> tuple[list[str], list[str]]:
    """Return (areas covered by listed skills, areas with no listed skills)."""
    names = [s.get("name", "").lower() for s in flat.get("skills_full", [])]
    covered, missing = [], []
    for area, terms in JD_AREAS.items():
        if any(_skill_matches(n, terms) for n in names):
            covered.append(area)
        else:
            missing.append(area)
    return covered, missing


def _join(items: list[str], limit: int) -> str:
    items = items[:limit]
    if len(items) <= 1:
        return "".join(items)
    return ", ".join(items[:-1]) + " and " + items[-1]


def _concern_text(flat: dict[str, Any], rank: int) -> str:
    """Honest concern string drawn from weak signals in the record."""
    concerns = []

    notice = flat.get("notice_period_days", 90)
    if notice > 90:
        concerns.append(f"{notice}-day notice period")

    try:
        last = datetime.strptime(flat.get("last_active_date", "2020-01-01")[:10], "%Y-%m-%d").date()
        days = (ref_today() - last).days
        if days > 60:
            concerns.append(f"last active {days} days before the data snapshot")
    except Exception:
        pass

    rr = flat.get("recruiter_response_rate", 0)
    if 0 < rr < 0.3:
        concerns.append(f"low response rate ({rr:.0%})")

    yoe = flat.get("years_experience", 0)
    if yoe < 5:
        concerns.append(f"slightly under JD experience band ({yoe:.1f} yrs)")
    elif yoe > 10:
        concerns.append(f"over-experienced at {yoe:.1f} yrs")

    if not concerns:
        return ""
    if rank <= 20:
        return f" Note: {'; '.join(concerns[:1])}."
    return f" Concern: {'; '.join(concerns[:2])}."


def generate_reasoning(
    flat: dict[str, Any],
    scores: dict[str, float],
    rank: int,
) -> str:
    """
    Generate a specific, honest reasoning string (1-2 sentences).

    Args:
        flat:   Flattened candidate dict.
        scores: Score dict from CandidateScorer.score().
        rank:   Final rank (1 = best).
    """
    title  = flat.get("current_title", "Candidate")
    yoe    = flat.get("years_experience", 0)
    ind    = flat.get("industry", "")
    loc    = flat.get("location", "")
    comp   = flat.get("company", "")
    github = flat.get("github_score", -1)
    otw    = flat.get("open_to_work", False)
    rr     = flat.get("recruiter_response_rate", 0)
    edu_t  = flat.get("education_tier", "tier_4")
    score  = scores.get("composite_score", 0)

    top_skills = _top_jd_skills(flat, n=3)
    skills_str = ", ".join(top_skills) if top_skills else "adjacent technical skills"
    covered, missing = _area_coverage(flat)
    covered_str = _join(covered, 2)
    missing_str = _join(missing, 2)

    comp_part = f" at {comp}" if comp else ""
    loc_part = f", {loc}" if loc else ""
    ind_part = f" ({ind} background)" if ind else ""

    if rank <= 10:
        # Strong fit: lead with the JD areas it covers.
        what = covered_str or "adjacent areas"
        sent1 = _pick(flat, [
            f"Strong fit: {title}{comp_part}{loc_part}, {yoe:.1f} yrs, covers {what} ({skills_str}).",
            f"{title}{comp_part}{loc_part} with {yoe:.1f} yrs; direct match on {what}: {skills_str}.",
            f"{title}, {yoe:.1f} yrs{comp_part}; hands-on {what} skills listed: {skills_str}.",
        ])
    elif rank <= 30:
        what = covered_str or "adjacent areas"
        gap = f" Nothing listed for {missing_str}." if missing and covered else ""
        sent1 = _pick(flat, [
            f"Solid fit: {title}{ind_part}, {yoe:.1f} yrs; covers {what} ({skills_str}).{gap}",
            f"{title}{ind_part} with {yoe:.1f} yrs; relevant to {what}: {skills_str}.{gap}",
            f"{title}, {yoe:.1f} yrs; JD overlap in {what} ({skills_str}).{gap}",
        ])
    elif rank <= 60:
        what = covered_str or "only adjacent skills"
        gap = f"; no skills listed for {missing_str}" if missing else ""
        sent1 = _pick(flat, [
            f"Partial fit: {title}, {yoe:.1f} yrs; overlap in {what} ({skills_str}){gap}.",
            f"{title} with {yoe:.1f} yrs; some relevant skills ({skills_str}){gap}.",
            f"{title}, {yoe:.1f} yrs; covers {what} only ({skills_str}){gap}.",
        ])
    else:
        # Weak fit: be explicit about what is missing.
        non_jd = [
            s for s in flat.get("skills", [])[:6]
            if isinstance(s, str) and not _skill_matches(s.lower(), CORE_JD_SKILLS)
        ]
        non_str = ", ".join(non_jd[:3]) if non_jd else "non-core skills"
        if covered:
            fit = f"touches {_join(covered, 2)} only"
        else:
            fit = f"no listed skills in any core JD area (e.g. {missing_str})" if missing_str else "no listed skills in the core JD areas"
        sent1 = _pick(flat, [
            f"Weak fit: {title}, {yoe:.1f} yrs; {fit}; skills are mostly {non_str}.",
            f"{title} with {yoe:.1f} yrs; {fit}, listed skills lean {non_str}.",
            f"{title}, {yoe:.1f} yrs; included to fill the top 100: {fit} (mainly {non_str}).",
        ])

    signal_parts = []
    if otw:
        signal_parts.append("open to work")
    if github >= 50:
        signal_parts.append(f"active GitHub ({github:.0f}/100)")
    elif github >= 20:
        signal_parts.append(f"moderate GitHub ({github:.0f}/100)")
    if rr >= 0.7:
        signal_parts.append(f"high recruiter response rate ({rr:.0%})")
    if edu_t == "tier_1":
        signal_parts.append("tier-1 education")
    elif edu_t == "tier_2":
        signal_parts.append("tier-2 institution")

    concern = _concern_text(flat, rank)

    if signal_parts:
        sent2 = "Signals: " + "; ".join(signal_parts[:3]) + "." + concern
    elif concern:
        sent2 = concern.strip()
    else:
        sent2 = f"Overall fit score: {score:.2f}."

    return f"{sent1} {sent2}".strip()
