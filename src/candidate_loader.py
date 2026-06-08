import json
from dataclasses import dataclass, field
from typing import Any, Dict, Iterator, List, Optional


@dataclass
class CandidateProfile:
    candidate_id: str
    current_title: str
    headline: str
    summary: str
    location: str
    country: str
    years_of_experience: float
    current_company: str
    current_company_size: str
    current_industry: str
    career_history: List[Dict[str, Any]] = field(default_factory=list)
    education: List[Dict[str, Any]] = field(default_factory=list)
    skills: List[Dict[str, Any]] = field(default_factory=list)
    certifications: List[Dict[str, Any]] = field(default_factory=list)
    languages: List[Dict[str, Any]] = field(default_factory=list)
    redrob_signals: Dict[str, Any] = field(default_factory=dict)
    raw: Dict[str, Any] = field(default_factory=dict)


def load_candidates(
    jsonl_path: str, limit: Optional[int] = None
) -> Iterator[CandidateProfile]:
    with open(jsonl_path, encoding="utf-8") as f:
        for i, line in enumerate(f):
            if limit is not None and i >= limit:
                break
            if not line.strip():
                continue
            raw = json.loads(line)
            prof = raw["profile"]
            yield CandidateProfile(
                candidate_id=raw["candidate_id"],
                current_title=prof.get("current_title", ""),
                headline=prof.get("headline", ""),
                summary=prof.get("summary", ""),
                location=prof.get("location", ""),
                country=prof.get("country", ""),
                years_of_experience=float(prof.get("years_of_experience", 0)),
                current_company=prof.get("current_company", ""),
                current_company_size=prof.get("current_company_size", ""),
                current_industry=prof.get("current_industry", ""),
                career_history=raw.get("career_history", []),
                education=raw.get("education", []),
                skills=raw.get("skills", []),
                certifications=raw.get("certifications", []),
                languages=raw.get("languages", []),
                redrob_signals=raw.get("redrob_signals", {}),
                raw=raw,
            )


def build_embedding_text(c: CandidateProfile) -> str:
    """Build rich text for embedding. More signal = better retrieval."""
    parts = []
    parts.append(f"Title: {c.current_title}. {c.headline}.")
    if c.summary:
        parts.append(c.summary)
    for role in c.career_history[:4]:
        desc = role.get("description", "")
        title = role.get("title", "")
        company = role.get("company", "")
        if desc:
            parts.append(f"Role at {company} as {title}: {desc[:300]}")
    skill_parts = []
    for s in c.skills[:20]:
        prof = s.get("proficiency", "")
        name = s.get("name", "")
        skill_parts.append(f"{name} ({prof})")
    if skill_parts:
        parts.append("Skills: " + ", ".join(skill_parts))
    for edu in c.education[:2]:
        deg = edu.get("degree", "")
        field_s = edu.get("field_of_study", "")
        inst = edu.get("institution", "")
        parts.append(f"Education: {deg} in {field_s} from {inst}")
    return " ".join(parts)
