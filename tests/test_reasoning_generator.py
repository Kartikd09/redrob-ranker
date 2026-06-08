import pytest
from src.reasoning_generator import generate_reasoning
from src.candidate_loader import CandidateProfile


def _make_candidate(title="ML Engineer", yoe=6.5, skills=None, company="Razorpay",
                    response_rate=0.7, open_to_work=True, github=60,
                    location="Pune, Maharashtra", country="India",
                    notice=30, summary="Built retrieval systems at scale."):
    return CandidateProfile(
        candidate_id="CAND_0000001",
        current_title=title, headline=f"{title} | embeddings and ranking",
        summary=summary, location=location, country=country,
        years_of_experience=yoe, current_company=company,
        current_company_size="201-500", current_industry="Technology",
        career_history=[
            {"company": company, "title": title, "duration_months": 36,
             "is_current": True, "description": "Built vector search and ranking."}
        ],
        skills=skills or [
            {"name": "FAISS", "proficiency": "advanced", "duration_months": 24},
            {"name": "Python", "proficiency": "expert", "duration_months": 48},
        ],
        redrob_signals={
            "open_to_work_flag": open_to_work, "recruiter_response_rate": response_rate,
            "github_activity_score": github, "notice_period_days": notice,
        },
    )


def test_reasoning_is_string():
    c = _make_candidate()
    r = generate_reasoning(c, rank=1, score=0.92)
    assert isinstance(r, str) and len(r) > 20


def test_reasoning_mentions_yoe():
    c = _make_candidate(yoe=6.5)
    r = generate_reasoning(c, rank=1, score=0.92)
    assert "6.5" in r or "6" in r


def test_reasoning_mentions_title():
    c = _make_candidate(title="Senior ML Engineer")
    r = generate_reasoning(c, rank=1, score=0.88)
    assert "ML Engineer" in r or "Senior" in r


def test_reasoning_mentions_concern_for_low_rank():
    c = _make_candidate(title="Marketing Manager", yoe=8.0, open_to_work=False)
    r = generate_reasoning(c, rank=90, score=0.25)
    assert any(word in r.lower() for word in ["concern", "gap", "limited", "not", "mismatch", "open"])


def test_reasoning_varies_by_candidate():
    c1 = _make_candidate(title="ML Engineer", yoe=6.0)
    c2 = _make_candidate(title="Data Scientist", yoe=3.0, company="TCS")
    r1 = generate_reasoning(c1, rank=5, score=0.85)
    r2 = generate_reasoning(c2, rank=80, score=0.30)
    assert r1 != r2


def test_no_hallucination_skills():
    c = _make_candidate(skills=[{"name": "Python", "proficiency": "advanced", "duration_months": 24}])
    r = generate_reasoning(c, rank=10, score=0.75)
    assert "Pinecone" not in r
    assert "FAISS" not in r


def test_reasoning_max_length():
    c = _make_candidate()
    r = generate_reasoning(c, rank=1, score=0.92)
    assert len(r) <= 300
