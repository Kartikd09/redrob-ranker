import pytest
from src.honeypot_detector import is_honeypot, FICTIONAL_COMPANIES


def _make_candidate(overrides=None):
    base = {
        "candidate_id": "CAND_0000001",
        "profile": {"years_of_experience": 5.0},
        "skills": [
            {"name": "Python", "proficiency": "advanced", "duration_months": 24}
        ],
        "career_history": [
            {
                "company": "Flipkart",
                "title": "Engineer",
                "start_date": "2019-01-01",
                "end_date": None,
                "duration_months": 60,
                "is_current": True,
            }
        ],
        "education": [
            {"degree": "B.Tech", "field_of_study": "Computer Science",
             "start_year": 2013, "end_year": 2017}
        ],
    }
    if overrides:
        for k, v in overrides.items():
            base[k] = v
    return base


def test_clean_candidate_not_honeypot():
    assert not is_honeypot(_make_candidate())


def test_expert_zero_duration_is_honeypot():
    c = _make_candidate({
        "skills": [{"name": "MLflow", "proficiency": "expert", "duration_months": 0}]
    })
    assert is_honeypot(c)


def test_career_longer_than_claimed_yoe_is_honeypot():
    c = _make_candidate({
        "profile": {"years_of_experience": 8.0},
        "career_history": [
            {"company": "TCS", "title": "Dev", "start_date": "2000-01-01",
             "end_date": None, "duration_months": 300, "is_current": True}
        ],
    })
    assert is_honeypot(c)


def test_fictional_company_is_honeypot():
    c = _make_candidate({
        "career_history": [
            {"company": "Wayne Enterprises", "title": "Dev", "start_date": "2019-01-01",
             "end_date": None, "duration_months": 60, "is_current": True}
        ]
    })
    assert is_honeypot(c)


def test_phd_in_mba_is_honeypot():
    c = _make_candidate({
        "education": [
            {"degree": "Ph.D", "field_of_study": "MBA",
             "start_year": 2010, "end_year": 2014}
        ]
    })
    assert is_honeypot(c)


def test_multiple_expert_zero_duration_is_honeypot():
    c = _make_candidate({
        "skills": [
            {"name": "Python", "proficiency": "expert", "duration_months": 0},
            {"name": "Java", "proficiency": "expert", "duration_months": 0},
        ]
    })
    assert is_honeypot(c)


def test_fictional_companies_set_not_empty():
    assert len(FICTIONAL_COMPANIES) > 5
    assert "Wayne Enterprises" in FICTIONAL_COMPANIES
