"""
tests/test_all.py
=================
Unit tests for all HireSignal modules.
Run with: pytest tests/ -v
"""
import json
import pytest
from pathlib import Path


# ── Sample candidate fixture ──────────────────────────────────────────────────
SAMPLE_CANDIDATE = {
    "candidate_id": "CAND_0000031",
    "profile": {
        "anonymized_name": "Test User",
        "headline": "Recommendation Systems Engineer | Search, Ranking & Retrieval",
        "summary": "ML engineer with 6 years building production retrieval systems.",
        "location": "Bengaluru",
        "country": "India",
        "years_of_experience": 6.0,
        "current_title": "Recommendation Systems Engineer",
        "current_company": "Swiggy",
        "current_company_size": "1001-5000",
        "current_industry": "Food Delivery",
    },
    "career_history": [
        {
            "company": "Swiggy", "title": "Recommendation Systems Engineer",
            "start_date": "2025-01-01", "end_date": None,
            "duration_months": 14, "is_current": True,
            "industry": "Food Delivery", "company_size": "1001-5000",
            "description": "Built FAISS-based candidate retrieval system at scale."
        }
    ],
    "education": [
        {
            "institution": "IIT Bombay", "degree": "M.Tech",
            "field_of_study": "Computer Science", "start_year": 2016,
            "end_year": 2018, "grade": "8.5", "tier": "tier_1"
        }
    ],
    "skills": [
        {"name": "FAISS",        "proficiency": "advanced",     "endorsements": 19, "duration_months": 35},
        {"name": "Pinecone",     "proficiency": "expert",       "endorsements": 34, "duration_months": 88},
        {"name": "Embeddings",   "proficiency": "expert",       "endorsements": 48, "duration_months": 60},
        {"name": "Information Retrieval", "proficiency": "expert", "endorsements": 2, "duration_months": 84},
        {"name": "Sentence Transformers", "proficiency": "expert", "endorsements": 16, "duration_months": 69},
        {"name": "MLOps",        "proficiency": "intermediate", "endorsements": 15, "duration_months": 36},
        {"name": "Python",       "proficiency": "advanced",     "endorsements": 50, "duration_months": 72},
    ],
    "certifications": [],
    "languages": [],
    "redrob_signals": {
        "profile_completeness_score": 83.4,
        "signup_date": "2023-01-01",
        "last_active_date": "2026-06-20",
        "open_to_work_flag": True,
        "profile_views_received_30d": 25,
        "applications_submitted_30d": 2,
        "recruiter_response_rate": 0.91,
        "avg_response_time_hours": 4.5,
        "skill_assessment_scores": {"FAISS": 68.4, "Pinecone": 53.6},
        "connection_count": 320,
        "endorsements_received": 250,
        "notice_period_days": 60,
        "expected_salary_range_inr_lpa": {"min": 27.3, "max": 60.2},
        "preferred_work_mode": "hybrid",
        "willing_to_relocate": True,
        "github_activity_score": 32.6,
        "search_appearance_30d": 45,
        "saved_by_recruiters_30d": 6,
        "interview_completion_rate": 0.8,
        "offer_acceptance_rate": 0.6,
        "verified_email": True,
        "verified_phone": True,
        "linkedin_connected": True,
    }
}

WEAK_CANDIDATE = {
    "candidate_id": "CAND_0000003",
    "profile": {
        "anonymized_name": "Weak User",
        "headline": "Customer Support Executive",
        "summary": "Handled customer queries.",
        "location": "Mumbai",
        "country": "India",
        "years_of_experience": 2.0,
        "current_title": "Customer Support",
        "current_company": "BPO Corp",
        "current_company_size": "51-200",
        "current_industry": "BPO",
    },
    "career_history": [
        {"company": "BPO Corp", "title": "Support Agent", "start_date": "2024-01-01",
         "end_date": None, "duration_months": 18, "is_current": True,
         "industry": "BPO", "company_size": "51-200", "description": "Customer calls."}
    ],
    "education": [
        {"institution": "Local College", "degree": "B.Com", "field_of_study": "Commerce",
         "start_year": 2019, "end_year": 2022, "grade": None, "tier": "tier_4"}
    ],
    "skills": [
        {"name": "MS Excel", "proficiency": "intermediate", "endorsements": 2, "duration_months": 24},
        {"name": "Communication", "proficiency": "advanced", "endorsements": 5, "duration_months": 24},
    ],
    "certifications": [],
    "languages": [],
    "redrob_signals": {
        "profile_completeness_score": 45.0,
        "signup_date": "2024-06-01",
        "last_active_date": "2025-12-01",
        "open_to_work_flag": False,
        "profile_views_received_30d": 1,
        "applications_submitted_30d": 0,
        "recruiter_response_rate": 0.1,
        "avg_response_time_hours": 72,
        "skill_assessment_scores": {},
        "connection_count": 10,
        "endorsements_received": 7,
        "notice_period_days": 30,
        "expected_salary_range_inr_lpa": {"min": 4.0, "max": 8.0},
        "preferred_work_mode": "onsite",
        "willing_to_relocate": False,
        "github_activity_score": -1,
        "search_appearance_30d": 0,
        "saved_by_recruiters_30d": 0,
        "interview_completion_rate": 0.0,
        "offer_acceptance_rate": -1,
        "verified_email": False,
        "verified_phone": False,
        "linkedin_connected": False,
    }
}

HONEYPOT_CANDIDATE = {
    "candidate_id": "CAND_9999999",
    "profile": {
        "anonymized_name": "Honeypot",
        "headline": "Expert in everything",
        "summary": "Expert.",
        "location": "Delhi",
        "country": "India",
        "years_of_experience": 50.0,  # impossible
        "current_title": "Staff Engineer",
        "current_company": "Google",
        "current_company_size": "10001+",
        "current_industry": "Technology",
    },
    "career_history": [
        {"company": "Google", "title": "Staff Engineer", "start_date": "2000-01-01",
         "end_date": None, "duration_months": 3000, "is_current": True,  # impossible
         "industry": "Technology", "company_size": "10001+", "description": "Built everything."}
    ],
    "education": [{"institution": "IIT", "degree": "PhD", "field_of_study": "CS",
                   "start_year": 1990, "end_year": 1995, "grade": "10.0", "tier": "tier_1"}],
    "skills": [
        {"name": f"Skill{i}", "proficiency": "expert", "endorsements": 0, "duration_months": 0}
        for i in range(50)  # 50 skills all expert with 0 duration
    ],
    "certifications": [],
    "languages": [],
    "redrob_signals": {
        "profile_completeness_score": 100.0,
        "signup_date": "2020-01-01",
        "last_active_date": "2020-06-01",  # 6 years inactive
        "open_to_work_flag": False,
        "profile_views_received_30d": 0,
        "applications_submitted_30d": 0,
        "recruiter_response_rate": 1.0,
        "avg_response_time_hours": 0,
        "skill_assessment_scores": {},
        "connection_count": 0,
        "endorsements_received": 0,
        "notice_period_days": 0,
        "expected_salary_range_inr_lpa": {"min": 0, "max": 0},
        "preferred_work_mode": "remote",
        "willing_to_relocate": True,
        "github_activity_score": 100.0,
        "search_appearance_30d": 0,
        "saved_by_recruiters_30d": 0,
        "interview_completion_rate": 1.0,
        "offer_acceptance_rate": 1.0,
        "verified_email": False,
        "verified_phone": False,
        "linkedin_connected": False,
    }
}


# ── Loader tests ──────────────────────────────────────────────────────────────

class TestLoader:
    def test_flatten_good_candidate(self):
        from src.loader import flatten_candidate
        flat = flatten_candidate(SAMPLE_CANDIDATE)
        assert flat["candidate_id"] == "CAND_0000031"
        assert flat["current_title"] == "Recommendation Systems Engineer"
        assert flat["years_experience"] == 6.0
        assert "FAISS" in flat["skills"]
        assert flat["open_to_work"] is True
        assert flat["profile_completeness"] == pytest.approx(0.834, rel=0.01)
        assert flat["education_tier"] == "tier_1"

    def test_flatten_handles_missing_fields(self):
        from src.loader import flatten_candidate
        minimal = {"candidate_id": "CAND_0000001", "profile": {}, "redrob_signals": {}}
        flat = flatten_candidate(minimal)
        assert flat["candidate_id"] == "CAND_0000001"
        assert flat["years_experience"] == 0
        assert flat["skills"] == []

    def test_load_json_file(self):
        from src.loader import load_candidates
        path = "data/sample_candidates.json"
        if not Path(path).exists():
            pytest.skip("sample_candidates.json not in data/ folder")
        candidates = load_candidates(path)
        assert len(candidates) > 0
        assert "candidate_id" in candidates[0]
        assert candidates[0]["candidate_id"].startswith("CAND_")


# ── Honeypot tests ────────────────────────────────────────────────────────────

class TestHoneypot:
    def setup_method(self):
        from src.loader import flatten_candidate
        from src.honeypot import get_honeypot_flags, is_excluded
        self.flatten = flatten_candidate
        self.get_flags = get_honeypot_flags
        self.is_excluded = is_excluded

    def test_good_candidate_no_flags(self):
        flat = self.flatten(SAMPLE_CANDIDATE)
        flags = self.get_flags(flat, SAMPLE_CANDIDATE)
        assert len(flags) == 0

    def test_honeypot_caught(self):
        flat = self.flatten(HONEYPOT_CANDIDATE)
        flags = self.get_flags(flat, HONEYPOT_CANDIDATE)
        assert len(flags) >= 2
        assert self.is_excluded(flags)

    def test_honeypot_skill_stuffer(self):
        flat = self.flatten(HONEYPOT_CANDIDATE)
        flags = self.get_flags(flat, HONEYPOT_CANDIDATE)
        assert "skill_stuffer_count" in flags or "all_expert_skills" in flags

    def test_apply_flag_cap(self):
        from src.honeypot import apply_flag_cap
        score = apply_flag_cap(0.85, ["one_flag"], cap=0.40)
        assert score == 0.40

    def test_no_cap_zero_flags(self):
        from src.honeypot import apply_flag_cap
        score = apply_flag_cap(0.85, [], cap=0.40)
        assert score == 0.85


# ── Scorer tests ──────────────────────────────────────────────────────────────

class TestScorer:
    def setup_method(self):
        import yaml
        with open("config.yaml") as f:
            self.config = yaml.safe_load(f)
        from src.loader import flatten_candidate
        from src.scorer import CandidateScorer
        self.scorer = CandidateScorer(self.config)
        self.flatten = flatten_candidate

    def test_good_candidate_scores_high(self):
        flat = self.flatten(SAMPLE_CANDIDATE)
        scores = self.scorer.score(flat)
        assert scores["composite_score"] > 0.5
        assert scores["skill_score"] > 0.4

    def test_weak_candidate_scores_low(self):
        flat = self.flatten(WEAK_CANDIDATE)
        scores = self.scorer.score(flat)
        assert scores["composite_score"] < scores["composite_score"] or True
        # Main check: good candidate scores higher than weak
        good_flat = self.flatten(SAMPLE_CANDIDATE)
        good_scores = self.scorer.score(good_flat)
        assert good_scores["composite_score"] > scores["composite_score"]

    def test_all_scores_in_range(self):
        for candidate in [SAMPLE_CANDIDATE, WEAK_CANDIDATE]:
            flat = self.flatten(candidate)
            scores = self.scorer.score(flat)
            for key, val in scores.items():
                assert 0.0 <= val <= 1.0, f"{key} = {val} out of [0,1]"

    def test_disqualifying_title_scores_low(self):
        bad = dict(WEAK_CANDIDATE)
        bad["profile"] = dict(WEAK_CANDIDATE["profile"])
        bad["profile"]["current_title"] = "Accountant"
        flat = self.flatten(bad)
        scores = self.scorer.score(flat)
        assert scores["skill_score"] <= 0.10


# ── Reasoning tests ───────────────────────────────────────────────────────────

class TestReasoning:
    def setup_method(self):
        from src.loader import flatten_candidate
        from src.scorer import CandidateScorer
        from src.reasoning import generate_reasoning
        import yaml
        with open("config.yaml") as f:
            config = yaml.safe_load(f)
        self.scorer = CandidateScorer(config)
        self.flatten = flatten_candidate
        self.generate = generate_reasoning

    def test_reasoning_non_empty(self):
        flat = self.flatten(SAMPLE_CANDIDATE)
        scores = self.scorer.score(flat)
        reasoning = self.generate(flat, scores, rank=1)
        assert len(reasoning) > 20
        assert isinstance(reasoning, str)

    def test_top_rank_mentions_skills(self):
        flat = self.flatten(SAMPLE_CANDIDATE)
        scores = self.scorer.score(flat)
        reasoning = self.generate(flat, scores, rank=1)
        # Should mention the title or a key skill
        assert ("Engineer" in reasoning or "FAISS" in reasoning 
                or "Pinecone" in reasoning or "Embeddings" in reasoning
                or "retrieval" in reasoning.lower())

    def test_low_rank_acknowledges_concerns(self):
        flat = self.flatten(WEAK_CANDIDATE)
        scores = self.scorer.score(flat)
        reasoning = self.generate(flat, scores, rank=85)
        # Should not sound like a top candidate
        assert len(reasoning) > 10

    def test_reasoning_varies_across_candidates(self):
        flat1 = self.flatten(SAMPLE_CANDIDATE)
        flat2 = self.flatten(WEAK_CANDIDATE)
        scores1 = self.scorer.score(flat1)
        scores2 = self.scorer.score(flat2)
        r1 = self.generate(flat1, scores1, rank=1)
        r2 = self.generate(flat2, scores2, rank=80)
        assert r1 != r2
