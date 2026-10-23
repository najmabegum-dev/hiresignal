"""
scorer.py
=========
Multi-signal candidate scorer for the HireSignal pipeline.

Scores each candidate on 4 dimensions:
  1. skill_match  — explicit skill overlap with JD (weighted by proficiency)
  2. career       — trajectory quality, company type, industry fit
  3. behavioral   — engagement, availability, activity signals
  4. education    — institution tier bonus

Final composite = weighted sum of all four.
No LLM calls here — runs fully offline on CPU.
"""

import logging
import math
from datetime import date, datetime
from typing import Any
from src.refdate import today as ref_today
logger = logging.getLogger(__name__)

# Proficiency multipliers — expert skills count more
PROFICIENCY_WEIGHT = {
    "expert":       1.0,
    "advanced":     0.75,
    "intermediate": 0.50,
    "beginner":     0.20,
}

# Education tier scores
EDUCATION_TIER_SCORE = {
    "tier_1": 1.0,
    "tier_2": 0.75,
    "tier_3": 0.50,
    "tier_4": 0.25,
    "unknown": 0.10,
}

# Company size score (larger = more production experience)
COMPANY_SIZE_SCORE = {
    "10001+":    1.0,
    "5001-10000": 0.9,
    "1001-5000": 0.8,
    "501-1000":  0.7,
    "201-500":   0.6,
    "51-200":    0.5,
    "11-50":     0.4,
    "1-10":      0.3,
}


def _parse_date(d: str) -> date:
    try:
        return datetime.strptime(str(d)[:10], "%Y-%m-%d").date()
    except Exception:
        return date(2020, 1, 1)


class CandidateScorer:
    """
    Scores candidates against the JD using multi-signal fusion.
    Designed to run on 100K candidates in < 5 minutes on CPU.
    No network calls, no LLM.
    """

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        self.weights = config["weights"]

        # Pre-build lowercase skill sets for fast matching
        core = config.get("core_skills", {})
        self.must_have_skills: set[str] = {
            s.lower() for s in core.get("must_have", [])
        }
        self.good_to_have_skills: set[str] = {
            s.lower() for s in core.get("good_to_have", [])
        }
        self.disqualifying_titles: list[str] = [
            t.lower() for t in core.get("disqualifying_titles", [])
        ]
        self.preferred_industries: set[str] = {
            i.lower() for i in config.get("jd", {}).get("preferred_industries", [])
        }
        self.location_bonus_cities: list[str] = [
            c.lower() for c in config.get("jd", {}).get("location_bonus_cities", [])
        ]
        self.jd_yoe_min: float = float(config.get("jd", {}).get("required_yoe_min", 5))
        self.jd_yoe_max: float = float(config.get("jd", {}).get("required_yoe_max", 9))

    # ── PUBLIC API ────────────────────────────────────────────────────

    def score(self, flat: dict[str, Any]) -> dict[str, float]:
        """
        Score one candidate. Returns dict of all component scores
        and the final composite.
        """
        skill  = self.skill_match_score(flat)
        career = self.career_score(flat)
        beh    = self.behavioral_score(flat)
        edu    = self.education_score(flat)

        composite = (
            self.weights["skill_match"] * skill
            + self.weights["career"]     * career
            + self.weights["behavioral"] * beh
            + self.weights["education"]  * edu
        )

        return {
            "skill_score":      round(skill, 4),
            "career_score":     round(career, 4),
            "behavioral_score": round(beh, 4),
            "education_score":  round(edu, 4),
            "composite_score":  round(float(composite), 4),
        }

    # ── SKILL MATCH ───────────────────────────────────────────────────

    def skill_match_score(self, flat: dict[str, Any]) -> float:
        """
        Score based on explicit skill overlap with JD requirements.
        
        - must_have skills that match: high weight per skill
        - good_to_have skills: medium weight
        - Proficiency multiplier applied per skill
        - Skill assessment scores from platform give a bonus
        - Disqualifying titles give a hard penalty
        """
        # Hard check: disqualifying title
        title_lower = flat.get("current_title", "").lower()
        for bad_title in self.disqualifying_titles:
            if bad_title in title_lower:
                return 0.05  # near-zero but not zero (avoids edge cases)

        skills_full = flat.get("skills_full", [])
        if not skills_full:
            return 0.0

        must_score = 0.0
        good_score = 0.0
        must_matched = 0
        good_matched = 0
        assessment_bonus = 0.0

        assessment_scores = flat.get("assessment_scores", {})

        for skill_obj in skills_full:
            name   = skill_obj.get("name", "").lower()
            prof   = skill_obj.get("proficiency", "intermediate")
            dur    = skill_obj.get("duration_months", 0)
            end    = skill_obj.get("endorsements", 0)
            p_mult = PROFICIENCY_WEIGHT.get(prof, 0.5)

            # Duration bonus: > 24 months = 1.2x, > 12 months = 1.1x
            dur_mult = 1.2 if dur >= 24 else (1.1 if dur >= 12 else 1.0)

            # Check if skill matches must-have list
            is_must = any(req in name or name in req for req in self.must_have_skills)
            is_good = any(req in name or name in req for req in self.good_to_have_skills)

            if is_must:
                must_score += p_mult * dur_mult
                must_matched += 1
                # Bonus if they have a platform assessment for this skill
                for assess_skill, assess_score in assessment_scores.items():
                    if assess_skill.lower() in name or name in assess_skill.lower():
                        assessment_bonus += (assess_score / 100.0) * 0.05
                        break

            elif is_good:
                good_score += p_mult * dur_mult * 0.6
                good_matched += 1

        # Normalise: perfect score = 8+ must-have matches
        must_norm = min(1.0, must_score / 8.0)
        good_norm = min(1.0, good_score / 6.0)

        # Endorsements received signal — people endorse real skills
        endorsements = flat.get("endorsements_received", 0)
        endorse_bonus = min(0.05, endorsements / 1000.0)

        raw = 0.65 * must_norm + 0.25 * good_norm + assessment_bonus + endorse_bonus
        return min(1.0, raw)

    # ── CAREER SCORE ──────────────────────────────────────────────────

    def career_score(self, flat: dict[str, Any]) -> float:
        """
        Score based on career trajectory and depth.
        
        - Years of experience vs JD requirement
        - Progression rate (are they growing in seniority?)
        - Industry relevance
        - Company quality (product companies > pure services)
        - No pure research background
        """
        yoe = flat.get("years_experience", 0)

        # YoE fit — JD wants 5-9 years
        if yoe < self.jd_yoe_min:
            yoe_score = yoe / self.jd_yoe_min  # partial credit
        elif yoe <= self.jd_yoe_max:
            yoe_score = 1.0  # perfect band
        else:
            # Over-experienced — slight penalty (JD explicitly mentions this)
            over = yoe - self.jd_yoe_max
            yoe_score = max(0.5, 1.0 - (over * 0.04))

        # Industry relevance
        industry_lower = flat.get("industry", "").lower()
        past_industries = [i.lower() for i in flat.get("past_industries", [])]
        all_industries = [industry_lower] + past_industries

        industry_score = 0.0
        for ind in all_industries:
            if any(pref in ind or ind in pref for pref in self.preferred_industries):
                industry_score = 1.0
                break
            elif "it services" in ind or "consulting" in ind:
                industry_score = max(industry_score, 0.4)  # services discount
            elif ind:
                industry_score = max(industry_score, 0.5)

        # Career progression — check if they're growing
        career = flat.get("career_history", [])
        progression_score = self._progression_score(career, yoe)

        # Company size quality signal
        size = flat.get("company_size", "")
        size_score = COMPANY_SIZE_SCORE.get(size, 0.5)

        # Location bonus
        location_lower = flat.get("location", "").lower()
        loc_bonus = 0.0
        if any(city in location_lower for city in self.location_bonus_cities):
            loc_bonus = 0.05
        if flat.get("willing_to_relocate", False):
            loc_bonus = max(loc_bonus, 0.03)

        raw = (
            0.35 * yoe_score
            + 0.25 * industry_score
            + 0.25 * progression_score
            + 0.15 * size_score
            + loc_bonus
        )
        return min(1.0, raw)

    def _progression_score(self, career: list[dict], yoe: float) -> float:
        """Score career progression from junior → senior."""
        if not career or yoe == 0:
            return 0.3

        SENIORITY = {
            "intern": 0, "trainee": 0, "fresher": 0, "graduate": 0,
            "junior": 1, "associate": 1,
            "engineer": 2, "analyst": 2, "developer": 2, "specialist": 2,
            "senior": 3, "staff": 3, "lead": 3,
            "principal": 4, "manager": 4, "architect": 4,
            "director": 5, "head": 5, "vp": 5,
        }

        levels = []
        for ch in career:
            title = ch.get("title", "").lower()
            for kw, level in sorted(SENIORITY.items(), key=lambda x: -x[1]):
                if kw in title:
                    levels.append(level)
                    break

        if not levels:
            return 0.3

        delta = levels[-1] - levels[0]  # growth from first to last role
        rate  = delta / max(yoe, 1.0)   # growth per year
        return min(1.0, max(0.0, 0.3 + rate * 0.5))

    # ── BEHAVIORAL SCORE ──────────────────────────────────────────────

    def behavioral_score(self, flat: dict[str, Any]) -> float:
        """
        Score based on 8 Redrob behavioral signals.
        
        These signals predict actual hirability — a great-on-paper
        candidate who hasn't logged in for 6 months is not actually available.
        """
        # 1. Recency — when did they last log in?
        last_active = _parse_date(flat.get("last_active_date", "2020-01-01"))
        days_ago = (ref_today() - last_active).days
        if days_ago <= 7:
            recency = 1.0
        elif days_ago <= 30:
            recency = 0.85
        elif days_ago <= 90:
            recency = 0.65
        elif days_ago <= 180:
            recency = 0.40
        else:
            recency = max(0.0, 1.0 - days_ago / 365.0)

        # 2. Open to work flag — biggest single signal
        otw_bonus = 0.20 if flat.get("open_to_work", False) else 0.0

        # 3. Profile completeness (0-1)
        completeness = flat.get("profile_completeness", 0.0)

        # 4. Recruiter response rate (0-1)
        response_rate = flat.get("recruiter_response_rate", 0.0)

        # 5. Interview completion rate (0-1)
        interview_rate = flat.get("interview_completion_rate", 0.0)

        # 6. GitHub activity (-1 = no github, 0-100)
        github_raw = flat.get("github_score", -1)
        github = (github_raw / 100.0) if github_raw >= 0 else 0.3  # neutral if no github

        # 7. Saved by recruiters in last 30 days (0-50 → 0-1)
        saved = min(1.0, flat.get("saved_by_recruiters_30d", 0) / 20.0)

        # 8. Applications submitted in last 30 days (active job seeker signal)
        apps = min(1.0, flat.get("applications_30d", 0) / 5.0)

        # Notice period — shorter is better for urgency
        notice = flat.get("notice_period_days", 90)
        if notice <= 30:
            notice_score = 1.0
        elif notice <= 60:
            notice_score = 0.8
        elif notice <= 90:
            notice_score = 0.6
        elif notice <= 120:
            notice_score = 0.4
        else:
            notice_score = 0.2

        # Verification bonus
        verified_bonus = 0.05 if flat.get("verified", False) else 0.0

        raw = (
            0.20 * recency
            + 0.15 * completeness
            + 0.15 * response_rate
            + 0.15 * interview_rate
            + 0.10 * github
            + 0.10 * saved
            + 0.10 * apps
            + 0.05 * notice_score
            + otw_bonus
            + verified_bonus
        )
        return min(1.0, raw)

    # ── EDUCATION SCORE ───────────────────────────────────────────────

    def education_score(self, flat: dict[str, Any]) -> float:
        """
        Score based on highest education tier.
        Tier 1 (IIT/IISc/BITS) = 1.0, Tier 4 = 0.25.
        """
        tier = flat.get("education_tier", "tier_4")
        return EDUCATION_TIER_SCORE.get(tier, 0.10)
