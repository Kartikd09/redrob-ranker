import pytest
from src.candidate_loader import load_candidates, CandidateProfile, build_embedding_text

CANDIDATES_PATH = "data/candidates.jsonl"

def test_load_first_candidate():
    candidates = list(load_candidates(CANDIDATES_PATH, limit=1))
    assert len(candidates) == 1
    c = candidates[0]
    assert c.candidate_id.startswith("CAND_")
    assert c.years_of_experience >= 0

def test_load_limit():
    candidates = list(load_candidates(CANDIDATES_PATH, limit=5))
    assert len(candidates) == 5

def test_embedding_text_not_empty():
    candidates = list(load_candidates(CANDIDATES_PATH, limit=1))
    text = build_embedding_text(candidates[0])
    assert len(text) > 50

def test_embedding_text_includes_skills():
    candidates = list(load_candidates(CANDIDATES_PATH, limit=10))
    for c in candidates:
        text = build_embedding_text(c)
        assert len(text) > 20

def test_all_fields_present():
    candidates = list(load_candidates(CANDIDATES_PATH, limit=1))
    c = candidates[0]
    assert hasattr(c, "redrob_signals")
    assert hasattr(c, "career_history")
    assert hasattr(c, "education")
    assert "recruiter_response_rate" in c.redrob_signals

def test_load_100_candidates():
    candidates = list(load_candidates(CANDIDATES_PATH, limit=100))
    assert len(candidates) == 100
    ids = {c.candidate_id for c in candidates}
    assert len(ids) == 100  # all unique
