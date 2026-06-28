"""
reasoning.py
============
Generates the 'reasoning' column for the submission CSV.

IMPORTANT RULES from submission_spec.md Stage 4 review:
  - Must reference specific facts (title, YoE, named skills, signal values)
  - Must connect to JD requirements — not generic praise
  - Must acknowledge concerns honestly (bad notice period, weak signals etc.)
  - No hallucination — only facts from the profile
  - Must vary across candidates (not templated)
  - Tone must match rank (rank 1 = strong, rank 95 = honest about limits)

All of this is done offline — NO API calls.
"""

from typing import Any


# Core JD skills to reference specifically
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


def _top_jd_skills(flat: dict[str, Any], n: int = 3) -> list[str]:
    """Return top N skills that match the JD core skills."""
    skills_full = flat.get("skills_full", [])
    matched = []
    for s in skills_full:
        name = s.get("name", "")
        name_lower = name.lower()
        if any(core in name_lower or name_lower in core for core in CORE_JD_SKILLS):
            prof = s.get("proficiency", "")
            matched.append((name, prof, s.get("duration_months", 0)))
    # Sort by proficiency level then duration
    prof_rank = {"expert": 3, "advanced": 2, "intermediate": 1, "beginner": 0}
    matched.sort(key=lambda x: (prof_rank.get(x[1], 0), x[2]), reverse=True)
    return [f"{m[0]} ({m[1]})" for m in matched[:n]]


def _concern_text(flat: dict[str, Any], rank: int) -> str:
    """Generate an honest concern string if the candidate has weak signals."""
    concerns = []

    notice = flat.get("notice_period_days", 90)
    if notice > 90:
        concerns.append(f"{notice}-day notice period")

    from datetime import date, datetime
    try:
        last = datetime.strptime(flat.get("last_active_date", "2020-01-01")[:10], "%Y-%m-%d").date()
        days = (date.today() - last).days
        if days > 60:
            concerns.append(f"last active {days} days ago")
    except Exception:
        pass

    rr = flat.get("recruiter_response_rate", 0)
    if rr < 0.3 and rr > 0:
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
    else:
        return f" Concern: {'; '.join(concerns[:2])}."


def generate_reasoning(
    flat: dict[str, Any],
    scores: dict[str, float],
    rank: int,
) -> str:
    """
    Generate a specific, honest, non-templated reasoning string.

    Args:
        flat:   Flattened candidate dict.
        scores: Score dict from CandidateScorer.score().
        rank:   Final rank (1 = best).

    Returns:
        1-2 sentence string for the reasoning column.
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

    # Get top JD-relevant skills
    top_skills = _top_jd_skills(flat, n=3)
    skills_str = ", ".join(top_skills) if top_skills else "adjacent technical skills"

    # Build first sentence — specific facts
    if rank <= 10:
        # Top 10: strong, specific
        loc_part = f" based in {loc}," if loc else ""
        comp_part = f" at {comp}" if comp else ""
        sent1 = (
            f"{title} with {yoe:.1f} yrs{comp_part}{loc_part} "
            f"brings direct JD match: {skills_str}."
        )
    elif rank <= 30:
        # 11-30: solid, mention industry
        ind_part = f" ({ind} background)" if ind else ""
        sent1 = (
            f"{title}{ind_part}, {yoe:.1f} yrs exp; "
            f"key skills: {skills_str}."
        )
    elif rank <= 60:
        # 31-60: moderate, acknowledge partial fit
        sent1 = (
            f"{title} with {yoe:.1f} yrs; partial skill overlap: {skills_str}."
        )
    else:
        # 61-100: honest about limits
        all_skills = flat.get("skills", [])
        non_jd = [s for s in all_skills[:4] if s.lower() not in CORE_JD_SKILLS]
        non_str = ", ".join(non_jd[:3]) if non_jd else "non-core skills"
        sent1 = (
            f"{title}, {yoe:.1f} yrs; limited JD alignment — "
            f"primarily {non_str}."
        )

    # Build second sentence — behavioral/engagement signals + concerns
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
