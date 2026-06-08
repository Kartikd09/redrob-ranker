# src/scorer.py
from datetime import date, datetime
from typing import Any, Dict, List
import numpy as np
from src.jd_parser import JDProfile

_SKILL_ALIASES: Dict[str, List[str]] = {
    "embeddings": ["text embedding", "sentence embedding", "dense vector", "word embedding",
                   "semantic embedding", "vector representation"],
    "retrieval": ["information retrieval", "semantic search", "vector search", "dense retrieval",
                  "document retrieval", "passage retrieval"],
    "ranking": ["reranking", "re-ranking", "ranking system", "learning to rank", "LTR",
                "candidate ranking", "result ranking"],
    "FAISS": ["faiss"],
    "NLP": ["natural language processing", "text processing", "language model"],
    "Python": ["python3", "python 3"],
    "LLM": ["large language model", "GPT", "language model", "transformer model",
             "claude", "gemini", "llama"],
    "RAG": ["retrieval augmented generation", "retrieval-augmented"],
    "fine-tuning": ["finetuning", "fine tune", "LoRA", "QLoRA", "PEFT", "model tuning"],
    "vector database": ["vector store", "vector index", "Pinecone", "Weaviate", "Qdrant",
                        "Milvus", "Chroma", "pgvector"],
    "Elasticsearch": ["elastic search", "opensearch", "solr"],
    "hybrid search": ["BM25 + embedding", "sparse + dense", "lexical + semantic"],
    "evaluation": ["NDCG", "MRR", "MAP", "precision recall", "A/B test", "offline eval"],
    "MLOps": ["ml ops", "model deployment", "model serving", "model monitoring"],
}

_CONSULTING_FIRMS = {
    "TCS", "Tata Consultancy Services", "Infosys", "Wipro", "Accenture",
    "Cognizant", "Capgemini", "HCL", "Tech Mahindra", "Mphasis",
    "LTI", "LTTS", "Hexaware", "Mindtree", "Persistent Systems",
}

_PRODUCT_COMPANIES_SIGNALS = {
    "Flipkart", "Amazon", "Google", "Microsoft", "Meta", "Apple", "Netflix",
    "Swiggy", "Zomato", "Razorpay", "PhonePe", "Paytm", "CRED", "Meesho",
    "Ola", "Uber", "LinkedIn", "Salesforce", "Atlassian", "Freshworks",
    "Zoho", "InMobi", "Sharechat", "Dream11", "Byju", "Unacademy",
}


def _normalize(score: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, score))


def _match_skill(skill_name: str, jd_skills: List[str]) -> float:
    skill_lower = skill_name.lower()
    for jd_skill in jd_skills:
        jd_lower = jd_skill.lower()
        if jd_lower == skill_lower:
            return 1.0
        if jd_lower in skill_lower or skill_lower in jd_lower:
            return 0.85
        for canonical, aliases in _SKILL_ALIASES.items():
            if jd_lower in canonical.lower() or jd_lower in [a.lower() for a in aliases]:
                if skill_lower in canonical.lower() or skill_lower in [a.lower() for a in aliases]:
                    return 0.80
    return 0.0


def score_skill_match(
    skills: List[Dict], jd: JDProfile, assessment_scores: Dict[str, float]
) -> float:
    if not skills:
        return 0.0
    required = jd.required_skills
    nice = jd.nice_to_have_skills
    total_required = len(required)
    if total_required == 0:
        return 0.5
    covered_score = 0.0
    for skill in skills:
        name = skill.get("name", "")
        proficiency = skill.get("proficiency", "intermediate")
        duration_months = skill.get("duration_months", 0)
        match = _match_skill(name, required)
        if match > 0:
            prof_mult = {"beginner": 0.5, "intermediate": 0.75, "advanced": 1.0, "expert": 1.1}.get(proficiency, 0.75)
            dur_mult = min(duration_months / 24, 1.0) if duration_months > 0 else 0.3
            assess_bonus = 0.05 if assessment_scores.get(name, 0) > 70 else 0.0
            covered_score += match * prof_mult * dur_mult + assess_bonus
    nice_bonus = 0.0
    for skill in skills:
        name = skill.get("name", "")
        if _match_skill(name, nice) > 0.8:
            nice_bonus = min(nice_bonus + 0.02, 0.10)
    raw = (covered_score / total_required) + nice_bonus
    return _normalize(raw)


def score_career_trajectory(
    career: List[Dict], yoe: float, jd: JDProfile
) -> float:
    if not career:
        return 0.1
    scores = []
    total_months = sum(r.get("duration_months", 0) for r in career)
    product_months = sum(
        r.get("duration_months", 0) for r in career
        if r.get("company", "") in _PRODUCT_COMPANIES_SIGNALS
    )
    consulting_months = sum(
        r.get("duration_months", 0) for r in career
        if r.get("company", "") in _CONSULTING_FIRMS
    )
    product_ratio = product_months / max(total_months, 1)
    consulting_ratio = consulting_months / max(total_months, 1)
    scores.append(product_ratio * 0.4 + (1 - consulting_ratio) * 0.3)
    tenures = [r.get("duration_months", 0) for r in career if r.get("duration_months", 0) > 0]
    if tenures:
        avg_tenure = float(np.mean(tenures))
        tenure_score = _normalize((avg_tenure - 12) / 36)
        scores.append(tenure_score)
    prod_keywords = ["shipped", "deployed", "production", "scale", "users", "launched",
                     "built", "designed", "architected", "improved", "reduced", "increased"]
    prod_keyword_hits = 0
    for role in career:
        desc = role.get("description", "").lower()
        prod_keyword_hits += sum(1 for kw in prod_keywords if kw in desc)
    prod_score = _normalize(prod_keyword_hits / max(len(career), 1) / 5)
    scores.append(prod_score)
    startup_sizes = {"1-10", "11-50", "51-200"}
    startup_months = sum(
        r.get("duration_months", 0) for r in career
        if r.get("company_size", "") in startup_sizes
    )
    startup_ratio = startup_months / max(total_months, 1)
    scores.append(startup_ratio * 0.5 + 0.5)
    return _normalize(float(np.mean(scores)))


def score_behavioral(signals: Dict[str, Any]) -> float:
    today = date.today()
    open_to_work = 1.0 if signals.get("open_to_work_flag", False) else 0.0
    last_active_str = signals.get("last_active_date", "")
    try:
        last_active = datetime.strptime(last_active_str, "%Y-%m-%d").date()
        days_inactive = (today - last_active).days
        recency_score = _normalize(1.0 - days_inactive / 180)
    except (ValueError, TypeError):
        recency_score = 0.3
    notice = signals.get("notice_period_days", 90)
    notice_score = _normalize(1.0 - (notice - 30) / 150) if notice > 30 else 1.0
    availability = 0.5 * open_to_work + 0.3 * recency_score + 0.2 * notice_score
    response_rate = float(signals.get("recruiter_response_rate", 0))
    apps_30d = min(signals.get("applications_submitted_30d", 0) / 5, 1.0)
    completeness = signals.get("profile_completeness_score", 50) / 100
    engagement = 0.4 * response_rate + 0.3 * apps_30d + 0.3 * completeness
    interview_rate = float(signals.get("interview_completion_rate", 0.5))
    offer_rate_raw = signals.get("offer_acceptance_rate", -1)
    offer_rate = float(offer_rate_raw) if offer_rate_raw >= 0 else 0.5
    verified = (
        (1 if signals.get("verified_email", False) else 0)
        + (1 if signals.get("verified_phone", False) else 0)
    ) / 2
    reliability = 0.4 * interview_rate + 0.3 * offer_rate + 0.3 * verified
    return _normalize(0.4 * availability + 0.35 * engagement + 0.25 * reliability)


def score_constraint_match(
    yoe: float, location: str, country: str, signals: Dict[str, Any], jd: JDProfile
) -> float:
    if jd.yoe_min <= yoe <= jd.yoe_max:
        yoe_score = 1.0
    elif yoe < jd.yoe_min:
        yoe_score = _normalize(yoe / jd.yoe_min)
    else:
        overshoot = yoe - jd.yoe_max
        yoe_score = _normalize(1.0 - overshoot / 10)
    loc_score = 0.1
    if country == "India":
        loc_score = 0.55
    for pref in jd.preferred_locations:
        if pref.lower() in location.lower():
            loc_score = 1.0
            break
    if signals.get("willing_to_relocate", False) and country == "India":
        loc_score = max(loc_score, 0.70)
    work_mode = signals.get("preferred_work_mode", "flexible")
    mode_score = {"onsite": 0.7, "hybrid": 1.0, "remote": 0.6, "flexible": 0.9}.get(work_mode, 0.7)
    # Weighted combination: yoe 0.30, location 0.55, work_mode 0.15
    return _normalize(0.30 * yoe_score + 0.55 * loc_score + 0.15 * mode_score)


def compute_disqualifier_penalty(
    career: List[Dict], skills: List[Dict], yoe: float, jd: JDProfile
) -> float:
    """Returns multiplier 0.0-1.0. Lower = more disqualified."""
    penalty = 1.0
    total_months = sum(r.get("duration_months", 0) for r in career)
    consulting_months = sum(
        r.get("duration_months", 0) for r in career
        if r.get("company", "") in _CONSULTING_FIRMS
    )
    if total_months > 0 and consulting_months / total_months > 0.8:
        penalty *= 0.20
    cv_skills = {
        "computer vision", "object detection", "image classification", "ocr",
        "speech recognition", "asr", "tts", "text to speech", "robotics",
    }
    if skills:
        cv_count = sum(1 for s in skills if s.get("name", "").lower() in cv_skills)
        if cv_count / len(skills) > 0.6:
            penalty *= 0.25
    tenures = [r.get("duration_months", 0) for r in career if r.get("duration_months", 0) > 0]
    if len(tenures) >= 3 and float(np.mean(tenures)) < 18:
        penalty *= 0.55
    retrieval_skills = {"FAISS", "Elasticsearch", "embeddings", "BM25", "Pinecone",
                        "Weaviate", "Qdrant", "retrieval", "ranking", "NLP", "sklearn"}
    skill_names = {s.get("name", "") for s in skills}
    has_retrieval_depth = bool(skill_names & retrieval_skills)
    langchain_only = any(
        s.get("name", "") in {"LangChain", "LlamaIndex"} for s in skills
    ) and not has_retrieval_depth
    if langchain_only:
        penalty *= 0.40
    return _normalize(penalty)


def compute_final_score(
    semantic_score: float,
    skills: List[Dict],
    career: List[Dict],
    signals: Dict[str, Any],
    yoe: float,
    location: str,
    country: str,
    jd: JDProfile,
    assessment_scores: Dict[str, float],
) -> float:
    skill_score = score_skill_match(skills, jd, assessment_scores)
    trajectory_score = score_career_trajectory(career, yoe, jd)
    behavioral_score = score_behavioral(signals)
    constraint_score = score_constraint_match(yoe, location, country, signals, jd)
    disqualifier = compute_disqualifier_penalty(career, skills, yoe, jd)
    weighted = (
        0.35 * semantic_score
        + 0.25 * skill_score
        + 0.15 * trajectory_score
        + 0.15 * behavioral_score
        + 0.10 * constraint_score
    )
    return _normalize(weighted * disqualifier)
