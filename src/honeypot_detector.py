# src/honeypot_detector.py
from typing import Any, Dict, Set

FICTIONAL_COMPANIES: Set[str] = {
    "Wayne Enterprises", "Initech", "Umbrella Corporation", "Aperture Science",
    "Stark Industries", "Acme Corp", "Globex Corporation", "Dunder Mifflin",
    "Vandelay Industries", "Bluth Company", "Los Pollos Hermanos",
    "Cyberdyne Systems", "Weyland-Yutani", "Tyrell Corporation",
}

_DEGREE_FIELD_CONTRADICTIONS = {
    "Ph.D": {"MBA", "Management", "Business Administration"},
    "MBA": {"Computer Science", "Physics", "Chemistry", "Mathematics"},
}


def is_honeypot(raw: Dict[str, Any]) -> bool:
    """Return True if candidate profile has impossible/contradictory signals."""
    skills = raw.get("skills", [])
    career = raw.get("career_history", [])
    education = raw.get("education", [])
    claimed_yoe = float(raw.get("profile", {}).get("years_of_experience", 0))

    # Check 1: expert proficiency with zero usage duration
    expert_zero = [
        s for s in skills
        if s.get("proficiency") == "expert" and s.get("duration_months", 1) == 0
    ]
    if len(expert_zero) >= 1:
        return True

    # Check 2: total career months > claimed YOE + 3-year buffer
    total_career_months = sum(r.get("duration_months", 0) for r in career)
    if total_career_months > (claimed_yoe * 12) + 36:
        return True

    # Check 3: fictional company names
    for role in career:
        if role.get("company", "") in FICTIONAL_COMPANIES:
            return True

    # Check 4: contradictory degree + field combinations
    for edu in education:
        degree = edu.get("degree", "")
        field = edu.get("field_of_study", "")
        forbidden_fields = _DEGREE_FIELD_CONTRADICTIONS.get(degree, set())
        if field in forbidden_fields:
            return True

    # Check 5: single role duration > 480 months (40 years)
    for role in career:
        if role.get("duration_months", 0) > 480:
            return True

    return False


def build_honeypot_flags(jsonl_path: str) -> Set[str]:
    """Stream full candidate file and return set of honeypot candidate_ids."""
    import json
    flags: Set[str] = set()
    with open(jsonl_path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            raw = json.loads(line)
            if is_honeypot(raw):
                flags.add(raw["candidate_id"])
    return flags
