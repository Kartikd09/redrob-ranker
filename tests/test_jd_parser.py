import pytest
from src.jd_parser import JDProfile, parse_jd

JD_PATH = "data/job_description.docx"

def test_parse_jd_returns_profile():
    profile = parse_jd(JD_PATH)
    assert isinstance(profile, JDProfile)

def test_jd_has_required_skills():
    profile = parse_jd(JD_PATH)
    assert len(profile.required_skills) > 0
    skill_names_lower = [s.lower() for s in profile.required_skills]
    assert any("embed" in s for s in skill_names_lower)

def test_jd_has_disqualifier_firms():
    profile = parse_jd(JD_PATH)
    assert "TCS" in profile.consulting_disqualifiers
    assert "Infosys" in profile.consulting_disqualifiers

def test_jd_has_yoe_range():
    profile = parse_jd(JD_PATH)
    assert profile.yoe_min == 5
    assert profile.yoe_max == 9

def test_jd_has_locations():
    profile = parse_jd(JD_PATH)
    assert len(profile.preferred_locations) > 0
    assert any("Pune" in loc or "Noida" in loc for loc in profile.preferred_locations)

def test_save_and_load_roundtrip(tmp_path):
    from src.jd_parser import save_jd_profile, load_jd_profile
    profile = parse_jd(JD_PATH)
    path = str(tmp_path / "jd_profile.json")
    save_jd_profile(profile, path)
    loaded = load_jd_profile(path)
    assert loaded.yoe_min == profile.yoe_min
    assert loaded.required_skills == profile.required_skills
