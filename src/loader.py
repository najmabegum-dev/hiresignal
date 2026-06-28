"""
loader.py
=========
Loads candidates from the hackathon dataset.
Handles:
  - .jsonl.gz  (the full 100K pool — gzipped JSONL)
  - .jsonl     (uncompressed JSONL)
  - .json      (sample_candidates.json — plain JSON list)

Also flattens the nested schema into what the rest of the
pipeline expects.
"""

import gzip
import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def load_candidates(path: str) -> list[dict[str, Any]]:
    """
    Load candidates from any supported format.

    Args:
        path: Path to .jsonl.gz, .jsonl, or .json file.

    Returns:
        List of raw candidate dicts (nested schema as-is from dataset).
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(
            f"Dataset file not found: {path}\n"
            "Make sure you placed candidates.jsonl.gz in your data/ folder."
        )

    suffix = "".join(p.suffixes).lower()  # handles .jsonl.gz

    if suffix == ".jsonl.gz" or path.endswith(".jsonl.gz"):
        logger.info("Loading gzipped JSONL from %s …", path)
        candidates = []
        with gzip.open(path, "rt", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    candidates.append(json.loads(line))
        logger.info("Loaded %d candidates from gzipped JSONL.", len(candidates))
        return candidates

    elif p.suffix.lower() == ".jsonl":
        logger.info("Loading JSONL from %s …", path)
        candidates = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    candidates.append(json.loads(line))
        logger.info("Loaded %d candidates from JSONL.", len(candidates))
        return candidates

    elif p.suffix.lower() == ".json":
        logger.info("Loading JSON from %s …", path)
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            logger.info("Loaded %d candidates from JSON list.", len(data))
            return data
        elif isinstance(data, dict) and "candidates" in data:
            logger.info("Loaded %d candidates from JSON object.", len(data["candidates"]))
            return data["candidates"]
        else:
            raise ValueError("JSON file must be a list of candidates or {'candidates': [...]}")

    else:
        raise ValueError(f"Unsupported file format: {suffix}. Use .jsonl.gz, .jsonl, or .json")


def flatten_candidate(c: dict[str, Any]) -> dict[str, Any]:
    """
    Flatten the nested hackathon candidate schema into the flat dict
    the rest of the pipeline works with.

    All field accesses use .get() with safe defaults so the pipeline
    never crashes on a missing field.
    """
    p = c.get("profile", {})
    sig = c.get("redrob_signals", {})
    skills_list = c.get("skills", [])
    career = c.get("career_history", [])
    education = c.get("education", [])

    # ── Skills ────────────────────────────────────────────────────────
    skill_names = [s["name"] for s in skills_list if "name" in s]

    # Weighted skill string: expert skills count more in embedding text
    skill_text_parts = []
    for s in skills_list:
        name = s.get("name", "")
        prof = s.get("proficiency", "intermediate")
        dur  = s.get("duration_months", 0)
        end  = s.get("endorsements", 0)
        # repeat expert/advanced skills so they weight more in the embedding
        if prof == "expert":
            skill_text_parts.extend([name, name, name])
        elif prof == "advanced":
            skill_text_parts.extend([name, name])
        else:
            skill_text_parts.append(name)

    # ── Career ────────────────────────────────────────────────────────
    past_titles = [ch.get("title", "") for ch in career]
    past_industries = list({ch.get("industry", "") for ch in career if ch.get("industry")})
    career_descriptions = " ".join(
        ch.get("description", "")[:120] for ch in career
    )

    # ── Education ─────────────────────────────────────────────────────
    best_edu_tier = "tier_4"
    edu_text_parts = []
    tier_rank = {"tier_1": 4, "tier_2": 3, "tier_3": 2, "tier_4": 1, "unknown": 0}
    for e in education:
        tier = e.get("tier", "unknown")
        if tier_rank.get(tier, 0) > tier_rank.get(best_edu_tier, 0):
            best_edu_tier = tier
        edu_text_parts.append(
            f"{e.get('degree', '')} {e.get('field_of_study', '')} "
            f"{e.get('institution', '')} {tier}"
        )

    # ── Certifications ────────────────────────────────────────────────
    certs = c.get("certifications", [])
    cert_names = [cert.get("name", "") for cert in certs]

    # ── Skill assessment scores ────────────────────────────────────────
    assessment_scores: dict[str, float] = sig.get("skill_assessment_scores", {})

    return {
        # Identity
        "candidate_id": c.get("candidate_id", ""),

        # Profile fields
        "current_title":    p.get("current_title", ""),
        "headline":         p.get("headline", ""),
        "summary":          p.get("summary", ""),
        "location":         p.get("location", ""),
        "country":          p.get("country", "India"),
        "years_experience": float(p.get("years_of_experience", 0)),
        "industry":         p.get("current_industry", ""),
        "company":          p.get("current_company", ""),
        "company_size":     p.get("current_company_size", ""),

        # Skills
        "skills":           skill_names,
        "skills_full":      skills_list,
        "skill_text":       ", ".join(skill_text_parts),

        # Career
        "past_titles":      past_titles,
        "past_industries":  past_industries,
        "career_history":   career,
        "career_text":      career_descriptions,

        # Education
        "education":        education,
        "education_tier":   best_edu_tier,
        "education_text":   " | ".join(edu_text_parts),

        # Certifications
        "certifications":   cert_names,

        # Behavioral signals (all from redrob_signals)
        "last_active_date":          sig.get("last_active_date", "2020-01-01"),
        "profile_completeness":      float(sig.get("profile_completeness_score", 0)) / 100.0,
        "open_to_work":              bool(sig.get("open_to_work_flag", False)),
        "applications_30d":          int(sig.get("applications_submitted_30d", 0)),
        "recruiter_response_rate":   float(sig.get("recruiter_response_rate", 0.0)),
        "avg_response_time_hours":   float(sig.get("avg_response_time_hours", 72)),
        "interview_completion_rate": float(sig.get("interview_completion_rate", 0.0)),
        "offer_acceptance_rate":     float(sig.get("offer_acceptance_rate", -1)),
        "github_score":              float(sig.get("github_activity_score", -1)),
        "profile_views_30d":         int(sig.get("profile_views_received_30d", 0)),
        "saved_by_recruiters_30d":   int(sig.get("saved_by_recruiters_30d", 0)),
        "search_appearance_30d":     int(sig.get("search_appearance_30d", 0)),
        "notice_period_days":        int(sig.get("notice_period_days", 90)),
        "willing_to_relocate":       bool(sig.get("willing_to_relocate", False)),
        "preferred_work_mode":       sig.get("preferred_work_mode", "hybrid"),
        "salary_min":                float((sig.get("expected_salary_range_inr_lpa") or {}).get("min", 0)),
        "salary_max":                float((sig.get("expected_salary_range_inr_lpa") or {}).get("max", 0)),
        "connection_count":          int(sig.get("connection_count", 0)),
        "endorsements_received":     int(sig.get("endorsements_received", 0)),
        "verified":                  bool(sig.get("verified_email", False) and sig.get("verified_phone", False)),
        "linkedin_connected":        bool(sig.get("linkedin_connected", False)),
        "assessment_scores":         assessment_scores,
        "signup_date":               sig.get("signup_date", "2020-01-01"),
    }
