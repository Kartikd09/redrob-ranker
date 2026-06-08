# src/reasoning_generator.py
from typing import List, Optional
from src.candidate_loader import CandidateProfile

_CONSULTING_FIRMS = {
    "TCS", "Infosys", "Wipro", "Accenture", "Cognizant", "Capgemini",
    "HCL", "Tech Mahindra", "Mphasis",
}

_JD_REQUIRED_SKILLS = {
    "embeddings", "FAISS", "vector database", "Pinecone", "Weaviate", "Qdrant",
    "Elasticsearch", "OpenSearch", "retrieval", "ranking", "NLP", "RAG",
    "sentence-transformers", "BGE", "E5", "fine-tuning", "LoRA", "NDCG",
    "evaluation", "Python", "MLOps", "LLM", "hybrid search", "BM25",
}


def _get_top_skills(candidate: CandidateProfile, max_skills: int = 3) -> List[str]:
    """Return up to max_skills skill names relevant to the JD."""
    relevant = []
    for s in candidate.skills:
        name = s.get("name", "")
        if any(jd_s.lower() in name.lower() or name.lower() in jd_s.lower()
               for jd_s in _JD_REQUIRED_SKILLS):
            relevant.append(name)
    if not relevant:
        relevant = [
            s["name"] for s in candidate.skills
            if s.get("proficiency") in ("advanced", "expert")
        ]
    return relevant[:max_skills]


def _get_concerns(candidate: CandidateProfile, rank: int) -> Optional[str]:
    """Identify honest concerns to include in reasoning."""
    sigs = candidate.redrob_signals
    concerns = []
    if not sigs.get("open_to_work_flag", True):
        concerns.append("not flagged as open to work")
    notice = sigs.get("notice_period_days", 30)
    if notice > 60:
        concerns.append(f"{notice}-day notice period")
    response_rate = sigs.get("recruiter_response_rate", 1.0)
    if response_rate < 0.2:
        concerns.append(f"low recruiter response rate ({response_rate:.0%})")
    is_consulting = any(
        c.get("company", "") in _CONSULTING_FIRMS
        for c in candidate.career_history
    )
    if is_consulting and len(candidate.career_history) <= 2:
        concerns.append("consulting-heavy background")
    yoe = candidate.years_of_experience
    if yoe < 4:
        concerns.append(f"only {yoe:.1f}yrs exp (JD asks 5-9)")
    if not concerns:
        return None
    return "; ".join(concerns[:2])


def generate_reasoning(
    candidate: CandidateProfile, rank: int, score: float
) -> str:
    sigs = candidate.redrob_signals
    yoe = candidate.years_of_experience
    title = candidate.current_title
    company = candidate.current_company
    top_skills = _get_top_skills(candidate)
    concern = _get_concerns(candidate, rank)
    response_rate = sigs.get("recruiter_response_rate", 0)
    github = sigs.get("github_activity_score", -1)
    notice = sigs.get("notice_period_days", 90)
    open_to_work = sigs.get("open_to_work_flag", False)

    if top_skills:
        skills_str = ", ".join(top_skills)
        strength = f"{title} with {yoe:.1f}yrs at {company}; JD-relevant skills: {skills_str}"
    else:
        strength = f"{title} with {yoe:.1f}yrs at {company}"

    signal_note = ""
    if rank <= 20:
        if open_to_work and response_rate >= 0.5:
            signal_note = f"; active, {response_rate:.0%} response rate"
        elif github > 50:
            signal_note = f"; GitHub score {github:.0f}"
        if notice <= 15:
            signal_note += "; immediate joiner"
        elif notice <= 30:
            signal_note += f"; {notice}d notice"

    concern_str = ""
    if concern and rank > 10:
        concern_str = f"; concern: {concern}"
    elif concern and rank <= 10:
        concern_str = f"; minor flag: {concern}"

    reasoning = strength + signal_note + concern_str
    return reasoning[:300]
