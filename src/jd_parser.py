# src/jd_parser.py
from dataclasses import dataclass, field
from typing import List
import json
from docx import Document


@dataclass
class JDProfile:
    required_skills: List[str] = field(default_factory=list)
    nice_to_have_skills: List[str] = field(default_factory=list)
    consulting_disqualifiers: List[str] = field(default_factory=list)
    cv_speech_disqualifiers: List[str] = field(default_factory=list)
    yoe_min: int = 5
    yoe_max: int = 9
    preferred_locations: List[str] = field(default_factory=list)
    notice_period_soft_max_days: int = 30
    full_text: str = ""
    jd_embedding_text: str = ""

    def to_dict(self) -> dict:
        return {
            "required_skills": self.required_skills,
            "nice_to_have_skills": self.nice_to_have_skills,
            "consulting_disqualifiers": self.consulting_disqualifiers,
            "cv_speech_disqualifiers": self.cv_speech_disqualifiers,
            "yoe_min": self.yoe_min,
            "yoe_max": self.yoe_max,
            "preferred_locations": self.preferred_locations,
            "notice_period_soft_max_days": self.notice_period_soft_max_days,
            "jd_embedding_text": self.jd_embedding_text,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "JDProfile":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


def _extract_text(docx_path: str) -> str:
    doc = Document(docx_path)
    return "\n".join(p.text for p in doc.paragraphs if p.text.strip())


def parse_jd(docx_path: str) -> JDProfile:
    text = _extract_text(docx_path)

    required_skills = [
        "embeddings", "sentence-transformers", "BGE", "E5", "text embeddings",
        "vector database", "vector search", "FAISS", "Pinecone", "Weaviate",
        "Qdrant", "Milvus", "OpenSearch", "Elasticsearch",
        "hybrid search", "dense retrieval", "sparse retrieval", "BM25",
        "ranking system", "retrieval system", "recommendation system",
        "NDCG", "MRR", "MAP", "A/B testing", "evaluation framework",
        "LLM", "RAG", "fine-tuning", "NLP", "information retrieval",
        "Python", "production ML", "MLOps",
    ]

    nice_to_have = [
        "LoRA", "QLoRA", "PEFT", "learning-to-rank", "LTR", "XGBoost",
        "HR tech", "recruiting", "marketplace", "distributed systems",
        "large-scale inference", "open source", "Kubernetes", "Spark",
    ]

    consulting_disqualifiers = [
        "TCS", "Infosys", "Wipro", "Accenture", "Cognizant", "Capgemini",
    ]

    cv_speech_disqualifiers = [
        "computer vision", "object detection", "speech recognition",
        "ASR", "robotics", "autonomous vehicles", "OCR",
    ]

    preferred_locations = [
        "Pune", "Noida", "Hyderabad", "Mumbai", "Bengaluru", "Bangalore",
        "Delhi", "Delhi NCR", "Gurugram", "Chennai",
    ]

    jd_embedding_text = (
        "Senior AI Engineer founding team. Build intelligence layer: candidate ranking, "
        "semantic search, embeddings-based retrieval, vector databases, hybrid search. "
        "Production ML systems, evaluation frameworks NDCG MRR MAP A/B testing. "
        "Python, sentence-transformers, FAISS, Pinecone, Weaviate, Qdrant, Elasticsearch. "
        "NLP information retrieval LLM fine-tuning RAG. Product company not research lab. "
        "Ship ranking system, improve recruiter engagement metrics. 5-9 years experience. "
        "Pune Noida India. Startup founding team scrappy product-engineering mindset."
    )

    return JDProfile(
        required_skills=required_skills,
        nice_to_have_skills=nice_to_have,
        consulting_disqualifiers=consulting_disqualifiers,
        cv_speech_disqualifiers=cv_speech_disqualifiers,
        yoe_min=5,
        yoe_max=9,
        preferred_locations=preferred_locations,
        notice_period_soft_max_days=30,
        full_text=text,
        jd_embedding_text=jd_embedding_text,
    )


def save_jd_profile(profile: JDProfile, path: str = "artifacts/jd_profile.json") -> None:
    with open(path, "w") as f:
        json.dump(profile.to_dict(), f, indent=2)


def load_jd_profile(path: str = "artifacts/jd_profile.json") -> JDProfile:
    with open(path) as f:
        return JDProfile.from_dict(json.load(f))
