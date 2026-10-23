"""
honeypot.py
===========
Detects honeypot and low-quality trap candidates.

The dataset contains ~80 honeypots with subtly impossible profiles.
If >10% of your top 100 are honeypots → instant disqualification.

This module flags candidates based on the signal patterns described
in the redrob_signals_doc and submission_spec.
"""

import logging
from datetime import date, datetime
from typing import Any
from src.refdate import today as ref_today
logger = logging.getLogger(__name__)


def _parse_date(d: str) -> date:
    try:
        return datetime.strptime(str(d)[:10], "%Y-%m-%d").date()
    except Exception:
        return date(2000, 1, 1)


def get_honeypot_flags(flat: dict[str, Any], raw: dict[str, Any]) -> list[str]:
    """
    Return a list of honeypot flag strings for a candidate.
    Empty list = clean candidate.
    1 flag  = suspicious, cap score at 0.40
    2+ flags = exclude entirely from ranking

    Args:
        flat: Flattened candidate dict (from loader.flatten_candidate)
        raw:  Original raw candidate dict (for career history date checks)

    Returns:
        List of flag name strings.
    """
    flags: list[str] = []
    career = raw.get("career_history", [])
    skills_full = raw.get("skills", [])
    sig = raw.get("redrob_signals", {})

    # ── Flag 1: All skills marked expert ──────────────────────────────
    if skills_full:
        all_expert = all(s.get("proficiency") == "expert" for s in skills_full)
        if all_expert and len(skills_full) >= 5:
            flags.append("all_expert_skills")

    # ── Flag 2: Too many skills (keyword stuffer) ──────────────────────
    if len(skills_full) > 40:
        flags.append("skill_stuffer_count")

    # ── Flag 3: Skills listed with 0 months duration but expert ────────
    zero_duration_experts = [
        s for s in skills_full
        if s.get("proficiency") == "expert" and s.get("duration_months", 1) == 0
    ]
    if len(zero_duration_experts) >= 3:
        flags.append("expert_zero_duration")

    # ── Flag 4: Triple perfect signals (statistically impossible) ──────
    rr  = sig.get("recruiter_response_rate", 0)
    icr = sig.get("interview_completion_rate", 0)
    oar = sig.get("offer_acceptance_rate", -1)
    if rr == 1.0 and icr == 1.0 and oar == 1.0:
        flags.append("triple_perfect_signals")

    # ── Flag 5: Perfect completeness + completely inactive ─────────────
    completeness = sig.get("profile_completeness_score", 0)
    last_active = _parse_date(sig.get("last_active_date", "2020-01-01"))
    days_inactive = (ref_today() - last_active).days
    apps_30d = sig.get("applications_submitted_30d", 0)
    if completeness >= 99 and days_inactive > 180 and apps_30d == 0:
        flags.append("perfect_but_completely_inactive")

    # ── Flag 6: Experience overflow ────────────────────────────────────
    yoe = float(raw.get("profile", {}).get("years_of_experience", 0))
    if yoe > 35:
        flags.append("experience_overflow")

    # ── Flag 7: Impossible career timeline ────────────────────────────
    # Company founded after candidate claims they started working there
    # We approximate: if a role duration_months > (yoe * 12 + 24)
    if career:
        total_months_claimed = sum(ch.get("duration_months", 0) for ch in career)
        # Total months > years_experience + 5yr buffer is suspicious
        if total_months_claimed > (yoe * 12) + 60:
            flags.append("impossible_career_timeline")

    # ── Flag 8: No notice period + no offer history + zero activity ────
    notice = sig.get("notice_period_days", 90)
    if notice == 0 and oar == -1 and apps_30d == 0 and days_inactive > 90:
        flags.append("no_history_ghost_profile")

    # ── Flag 9: GitHub score 100 but no github in skills/career ────────
    github = sig.get("github_activity_score", -1)
    if github == 100.0:
        flags.append("perfect_github_score")

    # ── Flag 10: Completely unverified with high engagement ────────────
    verified_email = sig.get("verified_email", False)
    verified_phone = sig.get("verified_phone", False)
    profile_views  = sig.get("profile_views_received_30d", 0)
    saved          = sig.get("saved_by_recruiters_30d", 0)
    if not verified_email and not verified_phone and profile_views > 100 and saved > 10:
        flags.append("unverified_high_engagement")

    if flags:
        logger.debug("Candidate %s flagged: %s", flat.get("candidate_id", "?"), flags)

    return flags


def is_excluded(flags: list[str], threshold: int = 2) -> bool:
    """Return True if candidate should be excluded from ranking entirely."""
    return len(flags) >= threshold


def apply_flag_cap(score: float, flags: list[str], cap: float = 0.40) -> float:
    """If candidate has exactly 1 flag, cap their score."""
    if len(flags) == 1:
        return min(score, cap)
    return score
