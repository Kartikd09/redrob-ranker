import pytest
import numpy as np
from src.scorer import (
    score_skill_match, score_career_trajectory, score_behavioral,
    score_constraint_match, compute_disqualifier_penalty, compute_final_score,
)
from src.jd_parser import JDProfile

JD = JDProfile(
    required_skills=["embeddings", "FAISS", "NLP", "Python", "ranking system"],
    nice_to_have_skills=["LoRA", "open source"],
    consulting_disqualifiers=["TCS", "Infosys", "Wipro", "Accenture", "Cognizant", "Capgemini"],
    cv_speech_disqualifiers=["computer vision", "object detection", "ASR"],
    yoe_min=5, yoe_max=9,
    preferred_locations=["Pune", "Noida", "Bengaluru"],
    notice_period_soft_max_days=30,
)


def _make_signals(**overrides):
    base = {
        "open_to_work_flag": True, "last_active_date": "2026-05-01",
        "recruiter_response_rate": 0.7, "applications_submitted_30d": 3,
        "profile_completeness_score": 85, "interview_completion_rate": 0.9,
        "offer_acceptance_rate": 0.8, "verified_email": True, "verified_phone": True,
        "github_activity_score": 60, "notice_period_days": 30,
        "expected_salary_range_inr_lpa": {"min": 25, "max": 40},
        "skill_assessment_scores": {},
    }
    base.update(overrides)
    return base


def test_skill_match_with_relevant_skills():
    skills = [
        {"name": "FAISS", "proficiency": "advanced", "duration_months": 24},
        {"name": "Python", "proficiency": "expert", "duration_months": 48},
        {"name": "NLP", "proficiency": "advanced", "duration_months": 36},
    ]
    score = score_skill_match(skills, JD, {})
    assert 0.3 < score <= 1.0


def test_skill_match_with_irrelevant_skills():
    skills = [
        {"name": "Photoshop", "proficiency": "expert", "duration_months": 60},
        {"name": "SEO", "proficiency": "advanced", "duration_months": 48},
    ]
    score = score_skill_match(skills, JD, {})
    assert score < 0.2


def test_behavioral_high_engagement():
    signals = _make_signals(open_to_work_flag=True, recruiter_response_rate=0.9)
    score = score_behavioral(signals)
    assert score > 0.6


def test_behavioral_unavailable_candidate():
    signals = _make_signals(open_to_work_flag=False, recruiter_response_rate=0.05,
                            last_active_date="2025-01-01")
    score = score_behavioral(signals)
    assert score < 0.5


def test_disqualifier_consulting_only():
    career = [
        {"company": "TCS", "title": "Dev", "duration_months": 36, "is_current": False, "description": ""},
        {"company": "Infosys", "title": "Dev", "duration_months": 36, "is_current": True, "description": ""},
    ]
    penalty = compute_disqualifier_penalty(career, [], 7.0, JD)
    assert penalty <= 0.25


def test_disqualifier_clean_candidate():
    career = [
        {"company": "Flipkart", "title": "ML Engineer", "duration_months": 36,
         "is_current": True, "description": "Built retrieval system for search ranking."},
    ]
    penalty = compute_disqualifier_penalty(career, [], 6.0, JD)
    assert penalty >= 0.85


def test_final_score_bounded():
    skills = [{"name": "FAISS", "proficiency": "advanced", "duration_months": 24}]
    career = [{"company": "Razorpay", "title": "ML Engineer", "duration_months": 36,
               "is_current": True, "description": "embedding retrieval ranking"}]
    signals = _make_signals()
    score = compute_final_score(
        semantic_score=0.8, skills=skills, career=career, signals=signals,
        yoe=6.5, location="Pune, Maharashtra", country="India", jd=JD, assessment_scores={},
    )
    assert 0.0 <= score <= 1.0


def test_constraint_india_preferred_location():
    signals = _make_signals()
    score = score_constraint_match(7.0, "Pune, Maharashtra", "India", signals, JD)
    assert score > 0.8


def test_constraint_outside_india():
    signals = _make_signals()
    score = score_constraint_match(7.0, "London, UK", "UK", signals, JD)
    assert score < 0.5
