# RecruitBrain — Redrob Hackathon Ranker Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a two-phase CPU-only candidate ranking system that scores 100K candidates against a Senior AI Engineer JD and outputs a top-100 ranked CSV with fact-grounded reasoning — compliant with all Redrob hackathon constraints.

**Architecture:** Phase 1 (offline pre-computation, no time limit) embeds all 100K candidate profiles using BGE-M3 via fastembed/ONNX, builds a FAISS index, extracts structured JD requirements, and flags honeypot candidates. Phase 2 (`rank.py`, ≤5 min CPU, no network) loads the precomputed artifacts, runs ANN retrieval + multi-signal re-ranking, applies honeypot disqualification, and generates fact-grounded reasoning per candidate.

**Tech Stack:** Python 3.12, fastembed (ONNX BGE-M3, no PyTorch dep), faiss-cpu, numpy, pandas, scikit-learn, rank-bm25, rich, pydantic

---

## Hackathon Constraints (Non-Negotiable)

| Constraint | Limit | Enforced At |
|---|---|---|
| `rank.py` total runtime | ≤5 min wall-clock | Stage 3 Docker sandbox |
| RAM during ranking | ≤16 GB | Stage 3 Docker sandbox |
| Compute | CPU only, no GPU | Stage 3 Docker sandbox |
| Network during ranking | OFF — no API calls | Stage 3 Docker sandbox |
| Disk intermediate state | ≤5 GB | Stage 3 Docker sandbox |
| Pre-computation | Allowed (no time limit) | Pre-computed artifacts committed to repo |
| Honeypot rate in top-100 | ≤10% or disqualified | Stage 3 honeypot check |
| Submission format | Exactly 100 rows, unique ranks 1–100, score non-increasing | Stage 1 auto-validator |

## Scoring Weights (Hidden Ground Truth)

```
Final = 0.50 × NDCG@10 + 0.30 × NDCG@50 + 0.15 × MAP + 0.05 × P@10
```

**Top-10 quality is worth 50% of the score. Get the top-10 right.**

---

## What Makes a Great Match (JD Analysis)

### Hard Requirements
- 5–9 years experience (soft — strong signal elsewhere can override)
- Production experience: embeddings-based retrieval (sentence-transformers, BGE, E5, OpenAI embeddings)
- Production experience: vector databases (Pinecone, Weaviate, Qdrant, Milvus, OpenSearch, Elasticsearch, FAISS)
- Experience designing evaluation frameworks (NDCG, MRR, MAP, A/B testing, offline-to-online correlation)
- Strong Python, care about code quality
- At least one end-to-end ranking/search/recommendation system shipped to real users

### Nice To Have
- LLM fine-tuning (LoRA, QLoRA, PEFT)
- Learning-to-rank (XGBoost/neural LTR)
- HR-tech / recruiting tech / marketplace products
- Distributed systems / large-scale inference
- Open-source contributions in AI/ML
- Located in Noida/Pune/Hyderabad/Mumbai/Delhi NCR or willing to relocate

### Explicit Disqualifiers (JD-stated)
- Pure research background (academic labs, research-only, no prod deploy)
- AI experience ≤12 months, LangChain-only, no pre-LLM ML production experience
- Senior engineers who haven't written production code in last 18 months
- Consulting-firm-only career: TCS, Infosys, Wipro, Accenture, Cognizant, Capgemini
- Primary expertise: computer vision / speech / robotics without NLP/IR exposure
- Closed-source proprietary systems only for 5+ years, no external validation
- Title-chaser pattern: switching companies every ~1.5 years optimizing for title
- Notice period >30 days (not disqualifying but bar rises)

### Behavioral Availability Signals (JD-stated)
- `open_to_work_flag = True` → strong positive
- `last_active_date` recency → staleness penalty
- `recruiter_response_rate` → availability proxy
- `github_activity_score` → production coding evidence

### Honeypot Detection Patterns
- `proficiency = "expert"` AND `duration_months = 0` for same skill
- Total career months > (years_of_experience × 12) + 24 (impossible overlap)
- Career role `duration_months` implies start before age ~18 given education dates
- `degree = "Ph.D"` in field with `field_of_study = "MBA"` (contradictory)
- Fictional company names (Wayne Enterprises, Initech, etc.) in career history

---

## File Structure

```
H2S-Hackthon/
├── rank.py                          # Phase 2 entry point — produces submission CSV
├── precompute.py                    # Phase 1 entry point — runs offline, no time limit
├── src/
│   ├── jd_parser.py                 # Parse JD docx → structured JDProfile dataclass
│   ├── candidate_loader.py          # Stream/load candidates.jsonl → CandidateProfile
│   ├── embedder.py                  # fastembed BGE-M3 wrapper, batched encoding
│   ├── faiss_index.py               # Build/save/load FAISS flat-IP index
│   ├── honeypot_detector.py         # Rule-based impossible-profile detector
│   ├── scorer.py                    # Multi-signal scoring: 5 components + multipliers
│   ├── reasoning_generator.py       # Fact-grounded per-candidate reasoning strings
│   └── submission_writer.py         # Write + validate final CSV
├── artifacts/                       # Pre-computed outputs (committed to repo)
│   ├── jd_profile.json              # Structured JD requirements (from precompute)
│   ├── candidate_embeddings.npy     # 100K × 384 float32 embeddings
│   ├── faiss.index                  # FAISS flat-IP index
│   ├── feature_matrix.npy           # 100K × N engineered features
│   ├── candidate_ids.json           # Ordered list of IDs matching embedding rows
│   └── honeypot_flags.json          # Set of candidate_ids flagged as honeypots
├── data/                            # Symlink or copy of hackathon data
│   ├── candidates.jsonl
│   └── job_description.docx
├── tests/
│   ├── test_honeypot_detector.py
│   ├── test_scorer.py
│   ├── test_reasoning_generator.py
│   └── test_submission_writer.py
├── requirements.txt
├── submission_metadata.yaml
└── README.md
```

---

## Scoring Model Detail

### Component Weights

```
final_score = (
    0.35 × semantic_score          # BGE-M3 cos-sim: JD embedding ↔ profile embedding
  + 0.25 × skill_match_score       # Required skills coverage, proficiency+duration weighted
  + 0.15 × career_trajectory_score # Product arc, tenure health, founding-team fit
  + 0.15 × behavioral_score        # Availability × engagement × reliability
  + 0.10 × constraint_match_score  # Location, YOE range, notice period, salary
) × disqualifier_penalty           # 0.0–1.0 multiplier; 0.0 = hard disqualify
  × honeypot_penalty               # 0.0 if flagged as honeypot
```

### Skill Match Score
```python
# For each JD required skill concept (not exact keyword):
#   - Exact name match in skills list: full credit
#   - Semantic neighbor (cos-sim > 0.82): 0.8× credit
#   - Duration-weighted: score × min(duration_months/24, 1.0)
#   - Proficiency multiplier: beginner=0.5, intermediate=0.75, advanced=1.0, expert=1.1
#   - Assessment score bonus: if skill_assessment_scores[skill] > 70: +0.05
# Also: career description NLP — mentions of "shipped", "production", "scale" near skill names
```

### Career Trajectory Score
```python
# product_company_ratio: fraction of career at non-consulting product companies
# tenure_health: penalize <18mo stints (title-chasing signal)
# recent_code: last role description mentions production coding (not "architecture/strategy")
# founding_fit: current company_size in ["1-10","11-50","51-200"] = startup experience
# industry_relevance: tech/AI/product industries weighted higher
```

### Behavioral Score (Availability × Engagement × Reliability)
```python
# availability  = 0.5×open_to_work + 0.3×recency_score + 0.2×notice_period_score
# engagement    = 0.4×recruiter_response_rate + 0.3×applications_30d_norm + 0.3×profile_completeness_norm
# reliability   = 0.4×interview_completion_rate + 0.3×offer_acceptance_rate_norm + 0.3×verified_signals
# behavioral    = availability × 0.4 + engagement × 0.35 + reliability × 0.25
```

### Disqualifier Penalty Multipliers
```python
# consulting_only_career:       0.20  (JD says explicitly bad fit)
# pure_research_no_prod:        0.15  (JD says tried twice, didn't work)
# cv_speech_only_no_nlp:        0.25  (JD says would be relearning fundamentals)
# langchain_only_under_12mo:    0.40  (JD says "probably not")
# no_prod_code_18mo:            0.50  (role writes code)
# title_chaser (avg tenure <18mo across 3+ roles): 0.55
# notice_gt_90d:                0.85  (not disqualifying, bar rises)
```

---

## Tasks

### Task 1: Project Scaffold + Dependencies

**Files:**
- Create: `requirements.txt`
- Create: `src/__init__.py`
- Create: `tests/__init__.py`

- [ ] **Step 1: Create requirements.txt**

```
fastembed==0.3.6
faiss-cpu==1.8.0
numpy==1.26.4
pandas==2.2.2
scikit-learn==1.4.2
rank-bm25==0.2.2
rich==13.7.1
pydantic==2.7.1
python-docx==1.1.2
pytest==8.2.0
```

- [ ] **Step 2: Install dependencies**

```bash
pip install -r requirements.txt
```

Expected: All packages install without error. fastembed will download the BGE-M3 ONNX model (~580MB) on first use.

- [ ] **Step 3: Create package init files**

```bash
mkdir -p src tests artifacts data
touch src/__init__.py tests/__init__.py
```

- [ ] **Step 4: Verify fastembed + FAISS work**

```python
# Run this once to confirm environment:
from fastembed import TextEmbedding
model = TextEmbedding("BAAI/bge-small-en-v1.5")  # small model for test
docs = ["hello world", "test embedding"]
embeds = list(model.embed(docs))
print(len(embeds), len(embeds[0]))  # Should print: 2 384
import faiss
index = faiss.IndexFlatIP(384)
import numpy as np
index.add(np.array(embeds, dtype=np.float32))
print(index.ntotal)  # Should print: 2
```

- [ ] **Step 5: Commit**

```bash
git init
git add requirements.txt src/ tests/ artifacts/ data/
git commit -m "feat: project scaffold and dependencies"
```

---

### Task 2: JD Parser

**Files:**
- Create: `src/jd_parser.py`
- Create: `tests/test_jd_parser.py`

- [ ] **Step 1: Write failing test**

```python
# tests/test_jd_parser.py
import pytest
from src.jd_parser import JDProfile, parse_jd

def test_parse_jd_returns_profile():
    jd_path = "data/job_description.docx"
    profile = parse_jd(jd_path)
    assert isinstance(profile, JDProfile)

def test_jd_has_required_skills():
    profile = parse_jd("data/job_description.docx")
    assert len(profile.required_skills) > 0
    # BGE/embeddings/FAISS are in the JD
    skill_names_lower = [s.lower() for s in profile.required_skills]
    assert any("embed" in s for s in skill_names_lower)

def test_jd_has_disqualifier_firms():
    profile = parse_jd("data/job_description.docx")
    assert "TCS" in profile.consulting_disqualifiers
    assert "Infosys" in profile.consulting_disqualifiers

def test_jd_has_yoe_range():
    profile = parse_jd("data/job_description.docx")
    assert profile.yoe_min == 5
    assert profile.yoe_max == 9

def test_jd_has_locations():
    profile = parse_jd("data/job_description.docx")
    assert len(profile.preferred_locations) > 0
    assert any("Pune" in loc or "Noida" in loc for loc in profile.preferred_locations)
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/test_jd_parser.py -v
```

Expected: `ImportError: cannot import name 'JDProfile' from 'src.jd_parser'`

- [ ] **Step 3: Implement jd_parser.py**

```python
# src/jd_parser.py
from dataclasses import dataclass, field
from typing import List
import json
import re
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
    jd_embedding_text: str = ""  # Condensed text for embedding

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

    # Required skills — explicit from JD + semantic equivalents
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

    # Condensed embedding text — captures the essence of what makes a great fit
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
```

- [ ] **Step 4: Run tests to verify pass**

```bash
pytest tests/test_jd_parser.py -v
```

Expected: All 5 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/jd_parser.py tests/test_jd_parser.py
git commit -m "feat: jd parser with structured requirements extraction"
```

---

### Task 3: Candidate Loader + Data Models

**Files:**
- Create: `src/candidate_loader.py`
- Create: `tests/test_candidate_loader.py`

- [ ] **Step 1: Write failing test**

```python
# tests/test_candidate_loader.py
import pytest
from src.candidate_loader import load_candidates, CandidateProfile, build_embedding_text

def test_load_first_candidate():
    candidates = list(load_candidates("data/candidates.jsonl", limit=1))
    assert len(candidates) == 1
    c = candidates[0]
    assert c.candidate_id.startswith("CAND_")
    assert c.years_of_experience >= 0

def test_load_limit():
    candidates = list(load_candidates("data/candidates.jsonl", limit=5))
    assert len(candidates) == 5

def test_embedding_text_not_empty():
    candidates = list(load_candidates("data/candidates.jsonl", limit=1))
    text = build_embedding_text(candidates[0])
    assert len(text) > 50

def test_embedding_text_includes_skills():
    candidates = list(load_candidates("data/candidates.jsonl", limit=10))
    for c in candidates:
        text = build_embedding_text(c)
        if c.skills:
            assert c.skills[0]["name"].lower() in text.lower() or len(text) > 100

def test_all_fields_present():
    candidates = list(load_candidates("data/candidates.jsonl", limit=1))
    c = candidates[0]
    assert hasattr(c, "redrob_signals")
    assert hasattr(c, "career_history")
    assert hasattr(c, "education")
    assert "recruiter_response_rate" in c.redrob_signals
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/test_candidate_loader.py -v
```

Expected: `ImportError`

- [ ] **Step 3: Implement candidate_loader.py**

```python
# src/candidate_loader.py
import json
from dataclasses import dataclass, field
from typing import Any, Dict, Generator, Iterator, List, Optional


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
    """Build a rich text representation for embedding. More signal = better retrieval."""
    parts = []

    # Title + headline carry strong role signal
    parts.append(f"Title: {c.current_title}. {c.headline}.")

    # Summary is the richest free-text signal
    if c.summary:
        parts.append(c.summary)

    # Career descriptions — production evidence lives here
    for role in c.career_history[:4]:  # cap at 4 most recent roles
        desc = role.get("description", "")
        title = role.get("title", "")
        company = role.get("company", "")
        if desc:
            parts.append(f"Role at {company} as {title}: {desc[:300]}")

    # Skills with proficiency
    skill_parts = []
    for s in c.skills[:20]:
        prof = s.get("proficiency", "")
        name = s.get("name", "")
        skill_parts.append(f"{name} ({prof})")
    if skill_parts:
        parts.append("Skills: " + ", ".join(skill_parts))

    # Education
    for edu in c.education[:2]:
        deg = edu.get("degree", "")
        field = edu.get("field_of_study", "")
        inst = edu.get("institution", "")
        parts.append(f"Education: {deg} in {field} from {inst}")

    return " ".join(parts)
```

- [ ] **Step 4: Run tests to verify pass**

```bash
pytest tests/test_candidate_loader.py -v
```

Expected: All 5 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/candidate_loader.py tests/test_candidate_loader.py
git commit -m "feat: candidate loader with embedding text builder"
```

---

### Task 4: Honeypot Detector

**Files:**
- Create: `src/honeypot_detector.py`
- Create: `tests/test_honeypot_detector.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_honeypot_detector.py
import pytest
from src.honeypot_detector import is_honeypot, FICTIONAL_COMPANIES

def _make_candidate(overrides=None):
    """Minimal valid candidate dict for testing."""
    base = {
        "candidate_id": "CAND_0000001",
        "profile": {
            "years_of_experience": 5.0,
        },
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
        # Deep merge top-level keys
        for k, v in overrides.items():
            base[k] = v
    return base


def test_clean_candidate_not_honeypot():
    c = _make_candidate()
    assert not is_honeypot(c)


def test_expert_zero_duration_is_honeypot():
    c = _make_candidate({
        "skills": [{"name": "MLflow", "proficiency": "expert", "duration_months": 0}]
    })
    assert is_honeypot(c)


def test_career_longer_than_claimed_yoe_is_honeypot():
    # Claims 8 years but career_history totals 25 years
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
            {"name": "Rust", "proficiency": "expert", "duration_months": 0},
        ]
    })
    assert is_honeypot(c)
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/test_honeypot_detector.py -v
```

Expected: `ImportError`

- [ ] **Step 3: Implement honeypot_detector.py**

```python
# src/honeypot_detector.py
from typing import Any, Dict, Set

# Companies that only exist in fiction — dataset uses these as honeypot signals
FICTIONAL_COMPANIES: Set[str] = {
    "Wayne Enterprises", "Initech", "Umbrella Corporation", "Aperture Science",
    "Stark Industries", "Acme Corp", "Globex Corporation", "Dunder Mifflin",
    "Vandelay Industries", "Bluth Company", "Los Pollos Hermanos",
    "Cyberdyne Systems", "Weyland-Yutani", "Tyrell Corporation",
}

# Degree/field contradictions: degree type cannot match this field
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

    # Check 2: total career months > claimed YOE + 3-year buffer (impossible)
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

    # Check 5: single role duration > 480 months (40 years — impossible modern career)
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
```

- [ ] **Step 4: Run tests to verify pass**

```bash
pytest tests/test_honeypot_detector.py -v
```

Expected: All 6 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/honeypot_detector.py tests/test_honeypot_detector.py
git commit -m "feat: honeypot detector with 5 impossible-profile checks"
```

---

### Task 5: Embedder + FAISS Index Builder

**Files:**
- Create: `src/embedder.py`
- Create: `src/faiss_index.py`

> Note: No unit tests for embedder (it wraps fastembed, tested by integration). Tests for FAISS index builder are functional checks.

- [ ] **Step 1: Implement embedder.py**

```python
# src/embedder.py
"""
Wraps fastembed BGE-M3 (ONNX, no PyTorch needed).
BGE-M3 is SOTA multilingual embedding model (2024), 384-dim, ~580MB download.
Falls back to bge-small-en-v1.5 (33MB) if M3 unavailable.
"""
from typing import Iterator, List
import numpy as np


_MODEL_NAME = "BAAI/bge-small-en-v1.5"  # 384-dim, fast CPU, ~33MB
# Use bge-small for speed. BGE-M3 is 1024-dim and slower on CPU.
# For hackathon: bge-small gives good quality at 3x the speed of larger models.


def get_model():
    from fastembed import TextEmbedding
    return TextEmbedding(model_name=_MODEL_NAME)


def embed_texts(texts: List[str], batch_size: int = 256) -> np.ndarray:
    """
    Embed list of texts, return float32 array shape (N, dim).
    Normalized for cosine similarity via inner product.
    """
    model = get_model()
    embeddings = list(model.embed(texts, batch_size=batch_size))
    arr = np.array(embeddings, dtype=np.float32)
    # L2-normalize for cosine sim via IP
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    return arr / norms
```

- [ ] **Step 2: Implement faiss_index.py**

```python
# src/faiss_index.py
"""
FAISS flat inner-product index for ANN retrieval.
IndexFlatIP on L2-normalized vectors = exact cosine similarity search.
For 100K candidates × 384-dim: ~150MB RAM, ~50ms query time.
"""
import numpy as np
import faiss
import json
from typing import List, Tuple


def build_index(embeddings: np.ndarray) -> faiss.IndexFlatIP:
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    return index


def save_index(index: faiss.IndexFlatIP, path: str) -> None:
    faiss.write_index(index, path)


def load_index(path: str) -> faiss.IndexFlatIP:
    return faiss.read_index(path)


def search_index(
    index: faiss.IndexFlatIP,
    query_embedding: np.ndarray,
    top_k: int = 1000,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns (scores, indices) arrays of shape (top_k,).
    scores are cosine similarities in [-1, 1].
    """
    query = query_embedding.reshape(1, -1).astype(np.float32)
    scores, indices = index.search(query, top_k)
    return scores[0], indices[0]
```

- [ ] **Step 3: Quick smoke test (manual run)**

```python
# Run interactively to verify:
from src.embedder import embed_texts
from src.faiss_index import build_index, search_index
import numpy as np

texts = ["Senior AI engineer with embeddings experience", "Marketing manager at startup"]
embeddings = embed_texts(texts)
print("Embedding shape:", embeddings.shape)  # (2, 384)

index = build_index(embeddings)
query = embed_texts(["retrieval ranking embeddings vector search"])
scores, indices = search_index(index, query[0], top_k=2)
print("Scores:", scores)   # First result should score higher (~0.8+)
print("Indices:", indices)  # Should be [0, 1]
```

- [ ] **Step 4: Commit**

```bash
git add src/embedder.py src/faiss_index.py
git commit -m "feat: fastembed BGE embedder and FAISS index builder"
```

---

### Task 6: Multi-Signal Scorer

**Files:**
- Create: `src/scorer.py`
- Create: `tests/test_scorer.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_scorer.py
import pytest
import numpy as np
from src.scorer import (
    score_skill_match,
    score_career_trajectory,
    score_behavioral,
    score_constraint_match,
    compute_disqualifier_penalty,
    compute_final_score,
)
from src.jd_parser import JDProfile

JD = JDProfile(
    required_skills=["embeddings", "FAISS", "NLP", "Python", "ranking system"],
    nice_to_have_skills=["LoRA", "open source"],
    consulting_disqualifiers=["TCS", "Infosys", "Wipro", "Accenture", "Cognizant", "Capgemini"],
    cv_speech_disqualifiers=["computer vision", "object detection", "ASR"],
    yoe_min=5,
    yoe_max=9,
    preferred_locations=["Pune", "Noida", "Bengaluru"],
    notice_period_soft_max_days=30,
)


def _make_signals(**overrides):
    base = {
        "open_to_work_flag": True,
        "last_active_date": "2026-05-01",
        "recruiter_response_rate": 0.7,
        "applications_submitted_30d": 3,
        "profile_completeness_score": 85,
        "interview_completion_rate": 0.9,
        "offer_acceptance_rate": 0.8,
        "verified_email": True,
        "verified_phone": True,
        "github_activity_score": 60,
        "notice_period_days": 30,
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
    assert 0.4 < score <= 1.0


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
    assert score < 0.35


def test_disqualifier_consulting_only():
    career = [
        {"company": "TCS", "title": "Dev", "duration_months": 36, "is_current": False,
         "description": ""},
        {"company": "Infosys", "title": "Dev", "duration_months": 36, "is_current": True,
         "description": ""},
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
        semantic_score=0.8,
        skills=skills,
        career=career,
        signals=signals,
        yoe=6.5,
        location="Pune, Maharashtra",
        country="India",
        jd=JD,
        assessment_scores={},
    )
    assert 0.0 <= score <= 1.0
```

- [ ] **Step 2: Run tests to verify fail**

```bash
pytest tests/test_scorer.py -v
```

Expected: `ImportError`

- [ ] **Step 3: Implement scorer.py**

```python
# src/scorer.py
from datetime import date, datetime
from typing import Any, Dict, List, Optional
import numpy as np
from src.jd_parser import JDProfile

# Exact + fuzzy skill matching — map JD concepts to surface forms
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

# Consulting firms — consulting-only career is an explicit JD disqualifier
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
    """Return 0-1 match score for a single skill against JD required skills."""
    skill_lower = skill_name.lower()
    for jd_skill in jd_skills:
        jd_lower = jd_skill.lower()
        if jd_lower == skill_lower:
            return 1.0
        if jd_lower in skill_lower or skill_lower in jd_lower:
            return 0.85
        # Check aliases
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
        endorsements = skill.get("endorsements", 0)

        match = _match_skill(name, required)
        if match > 0:
            # Proficiency multiplier
            prof_mult = {"beginner": 0.5, "intermediate": 0.75, "advanced": 1.0, "expert": 1.1}.get(
                proficiency, 0.75
            )
            # Duration multiplier (cap at 36 months = full credit)
            dur_mult = min(duration_months / 24, 1.0) if duration_months > 0 else 0.3
            # Assessment bonus
            assess_bonus = 0.05 if assessment_scores.get(name, 0) > 70 else 0.0
            covered_score += match * prof_mult * dur_mult + assess_bonus

    # Nice-to-have bonus (up to 0.1 extra)
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

    # 1. Product company ratio
    total_months = sum(r.get("duration_months", 0) for r in career)
    product_months = sum(
        r.get("duration_months", 0)
        for r in career
        if r.get("company", "") in _PRODUCT_COMPANIES_SIGNALS
    )
    consulting_months = sum(
        r.get("duration_months", 0)
        for r in career
        if r.get("company", "") in _CONSULTING_FIRMS
    )
    product_ratio = product_months / max(total_months, 1)
    consulting_ratio = consulting_months / max(total_months, 1)
    scores.append(product_ratio * 0.4 + (1 - consulting_ratio) * 0.3)

    # 2. Tenure health (penalize title-chasing: avg tenure < 18 months)
    tenures = [r.get("duration_months", 0) for r in career if r.get("duration_months", 0) > 0]
    if tenures:
        avg_tenure = np.mean(tenures)
        tenure_score = _normalize((avg_tenure - 12) / 36)  # 12mo=0, 48mo=1
        scores.append(tenure_score)

    # 3. Production coding signal from descriptions
    prod_keywords = [
        "shipped", "deployed", "production", "scale", "users", "launched",
        "built", "designed", "architected", "improved", "reduced", "increased",
    ]
    total_desc_len = 0
    prod_keyword_hits = 0
    for role in career:
        desc = role.get("description", "").lower()
        total_desc_len += len(desc)
        prod_keyword_hits += sum(1 for kw in prod_keywords if kw in desc)
    prod_score = _normalize(prod_keyword_hits / max(len(career), 1) / 5)
    scores.append(prod_score)

    # 4. Founding-team / startup fit (small company experience)
    startup_sizes = {"1-10", "11-50", "51-200"}
    startup_months = sum(
        r.get("duration_months", 0)
        for r in career
        if r.get("company_size", "") in startup_sizes
    )
    startup_ratio = startup_months / max(total_months, 1)
    scores.append(startup_ratio * 0.5 + 0.5)  # bias toward any experience

    return _normalize(float(np.mean(scores)))


def score_behavioral(signals: Dict[str, Any]) -> float:
    today = date.today()

    # Availability component
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

    # Engagement component
    response_rate = float(signals.get("recruiter_response_rate", 0))
    apps_30d = min(signals.get("applications_submitted_30d", 0) / 5, 1.0)
    completeness = signals.get("profile_completeness_score", 50) / 100
    engagement = 0.4 * response_rate + 0.3 * apps_30d + 0.3 * completeness

    # Reliability component
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
    scores = []

    # YOE range match
    if jd.yoe_min <= yoe <= jd.yoe_max:
        scores.append(1.0)
    elif yoe < jd.yoe_min:
        scores.append(_normalize(yoe / jd.yoe_min))
    else:  # over-experienced: slight penalty, not disqualifying
        overshoot = yoe - jd.yoe_max
        scores.append(_normalize(1.0 - overshoot / 10))

    # Location match
    loc_score = 0.3  # default: anywhere India
    if country == "India":
        loc_score = 0.6
    for pref in jd.preferred_locations:
        if pref.lower() in location.lower():
            loc_score = 1.0
            break
    if signals.get("willing_to_relocate", False) and country == "India":
        loc_score = max(loc_score, 0.75)
    scores.append(loc_score)

    # Work mode (JD is hybrid)
    work_mode = signals.get("preferred_work_mode", "flexible")
    mode_score = {"onsite": 0.7, "hybrid": 1.0, "remote": 0.6, "flexible": 0.9}.get(
        work_mode, 0.7
    )
    scores.append(mode_score)

    return _normalize(float(np.mean(scores)))


def compute_disqualifier_penalty(
    career: List[Dict], skills: List[Dict], yoe: float, jd: JDProfile
) -> float:
    """Returns multiplier 0.0–1.0. Lower = more disqualified. 0.0 = hard disqualify."""
    penalty = 1.0

    # Consulting-only career
    total_months = sum(r.get("duration_months", 0) for r in career)
    consulting_months = sum(
        r.get("duration_months", 0)
        for r in career
        if r.get("company", "") in _CONSULTING_FIRMS
    )
    if total_months > 0 and consulting_months / total_months > 0.8:
        penalty *= 0.20

    # CV/speech primary focus (>60% of skills are CV/speech domain)
    cv_skills = {
        "computer vision", "object detection", "image classification", "ocr",
        "speech recognition", "asr", "tts", "text to speech", "robotics",
    }
    if skills:
        cv_count = sum(1 for s in skills if s.get("name", "").lower() in cv_skills)
        if cv_count / len(skills) > 0.6:
            penalty *= 0.25

    # Title-chaser: avg tenure < 18 months across 3+ roles
    tenures = [r.get("duration_months", 0) for r in career if r.get("duration_months", 0) > 0]
    if len(tenures) >= 3 and np.mean(tenures) < 18:
        penalty *= 0.55

    # Recent-only LLM experience without pre-LLM ML history
    # Heuristic: only has LLM-era skills (2022+), no embedding/retrieval depth
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
```

- [ ] **Step 4: Run tests to verify pass**

```bash
pytest tests/test_scorer.py -v
```

Expected: All 7 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/scorer.py tests/test_scorer.py
git commit -m "feat: multi-signal scorer with disqualifier penalties"
```

---

### Task 7: Reasoning Generator

**Files:**
- Create: `src/reasoning_generator.py`
- Create: `tests/test_reasoning_generator.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_reasoning_generator.py
import pytest
from src.reasoning_generator import generate_reasoning
from src.candidate_loader import CandidateProfile


def _make_candidate(title="ML Engineer", yoe=6.5, skills=None, company="Razorpay",
                    response_rate=0.7, open_to_work=True, github=60,
                    location="Pune, Maharashtra", country="India",
                    notice=30, summary="Built retrieval systems at scale."):
    return CandidateProfile(
        candidate_id="CAND_0000001",
        current_title=title,
        headline=f"{title} | embeddings and ranking",
        summary=summary,
        location=location,
        country=country,
        years_of_experience=yoe,
        current_company=company,
        current_company_size="201-500",
        current_industry="Technology",
        career_history=[
            {"company": company, "title": title, "duration_months": 36,
             "is_current": True, "description": "Built vector search and ranking."}
        ],
        skills=skills or [
            {"name": "FAISS", "proficiency": "advanced", "duration_months": 24},
            {"name": "Python", "proficiency": "expert", "duration_months": 48},
        ],
        redrob_signals={
            "open_to_work_flag": open_to_work,
            "recruiter_response_rate": response_rate,
            "github_activity_score": github,
            "notice_period_days": notice,
        },
    )


def test_reasoning_is_string():
    c = _make_candidate()
    r = generate_reasoning(c, rank=1, score=0.92)
    assert isinstance(r, str)
    assert len(r) > 20


def test_reasoning_mentions_yoe():
    c = _make_candidate(yoe=6.5)
    r = generate_reasoning(c, rank=1, score=0.92)
    assert "6.5" in r or "6" in r


def test_reasoning_mentions_title():
    c = _make_candidate(title="Senior ML Engineer")
    r = generate_reasoning(c, rank=1, score=0.88)
    assert "ML Engineer" in r or "Senior" in r


def test_reasoning_mentions_concern_for_low_rank():
    # Low-ranked candidate should have honest concern noted
    c = _make_candidate(title="Marketing Manager", yoe=8.0, open_to_work=False)
    r = generate_reasoning(c, rank=90, score=0.25)
    assert any(word in r.lower() for word in ["concern", "gap", "limited", "not", "mismatch", "low"])


def test_reasoning_varies_by_candidate():
    c1 = _make_candidate(title="ML Engineer", yoe=6.0)
    c2 = _make_candidate(title="Data Scientist", yoe=3.0, company="TCS")
    r1 = generate_reasoning(c1, rank=5, score=0.85)
    r2 = generate_reasoning(c2, rank=80, score=0.30)
    assert r1 != r2


def test_no_hallucination_skills():
    # Reasoning should not mention skills not in candidate profile
    c = _make_candidate(skills=[{"name": "Python", "proficiency": "advanced", "duration_months": 24}])
    r = generate_reasoning(c, rank=10, score=0.75)
    assert "Pinecone" not in r
    assert "FAISS" not in r  # not in their skills
```

- [ ] **Step 2: Run tests to verify fail**

```bash
pytest tests/test_reasoning_generator.py -v
```

Expected: `ImportError`

- [ ] **Step 3: Implement reasoning_generator.py**

```python
# src/reasoning_generator.py
"""
Generate fact-grounded, JD-connected reasoning strings for each ranked candidate.
Rules from Stage 4 rubric:
  1. Specific facts from profile (yoe, title, named skills, signal values)
  2. JD connection (not generic praise)
  3. Honest concerns for gaps
  4. No hallucination (only mention what's in the profile)
  5. Variation (not templated)
  6. Rank consistency (tone matches rank)
"""
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
    "evaluation", "Python", "MLOps", "LLM", "hybrid search",
}


def _get_top_skills(candidate: CandidateProfile, max_skills: int = 3) -> List[str]:
    """Return up to max_skills skill names that are relevant to the JD."""
    relevant = []
    for s in candidate.skills:
        name = s.get("name", "")
        if any(jd_s.lower() in name.lower() or name.lower() in jd_s.lower()
               for jd_s in _JD_REQUIRED_SKILLS):
            relevant.append(name)
    if not relevant:
        # Fall back to any advanced/expert skills
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
        concerns.append("not currently flagged as open to work")

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
        concerns.append(f"only {yoe:.1f} years experience (JD asks 5–9)")

    if not concerns:
        return None
    return "; ".join(concerns[:2])  # cap at 2 concerns per candidate


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

    # Build strength statement
    if top_skills:
        skills_str = ", ".join(top_skills)
        strength = f"{title} with {yoe:.1f}yrs at {company}; JD-relevant skills: {skills_str}"
    else:
        strength = f"{title} with {yoe:.1f}yrs at {company}"

    # Add behavioral signal for high ranks (top 20) or notable signals
    signal_note = ""
    if rank <= 20:
        if open_to_work and response_rate >= 0.5:
            signal_note = f"; active on platform, {response_rate:.0%} response rate"
        elif github > 50:
            signal_note = f"; strong GitHub activity (score {github:.0f})"
        if notice <= 15:
            signal_note += f"; immediate joiner"
        elif notice <= 30:
            signal_note += f"; {notice}d notice"

    # Build concern statement for mid/low ranks
    concern_str = ""
    if concern and rank > 10:
        concern_str = f"; concern: {concern}"
    elif concern and rank <= 10:
        concern_str = f"; minor flag: {concern}"

    reasoning = strength + signal_note + concern_str
    return reasoning[:300]  # cap at 300 chars for CSV readability
```

- [ ] **Step 4: Run tests to verify pass**

```bash
pytest tests/test_reasoning_generator.py -v
```

Expected: All 6 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/reasoning_generator.py tests/test_reasoning_generator.py
git commit -m "feat: fact-grounded reasoning generator with Stage-4 rubric compliance"
```

---

### Task 8: Submission Writer + Validator

**Files:**
- Create: `src/submission_writer.py`
- Create: `tests/test_submission_writer.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_submission_writer.py
import pytest
import csv
import os
import tempfile
from src.submission_writer import write_submission, validate_submission


def _make_rows(n=100):
    return [
        {
            "candidate_id": f"CAND_{str(i).zfill(7)}",
            "rank": i,
            "score": round(1.0 - (i - 1) * 0.009, 4),
            "reasoning": f"Candidate {i} reasoning text here.",
        }
        for i in range(1, n + 1)
    ]


def test_write_creates_file():
    rows = _make_rows()
    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
        path = f.name
    write_submission(rows, path)
    assert os.path.exists(path)
    os.unlink(path)


def test_write_correct_row_count():
    rows = _make_rows()
    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode="w") as f:
        path = f.name
    write_submission(rows, path)
    with open(path) as f:
        lines = f.readlines()
    assert len(lines) == 101  # header + 100 rows
    os.unlink(path)


def test_validate_passes_valid_submission():
    rows = _make_rows()
    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode="w") as f:
        path = f.name
    write_submission(rows, path)
    errors = validate_submission(path)
    assert errors == []
    os.unlink(path)


def test_validate_catches_duplicate_rank():
    rows = _make_rows()
    rows[5]["rank"] = 1  # duplicate rank 1
    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode="w") as f:
        path = f.name
    write_submission(rows, path)
    errors = validate_submission(path)
    assert any("rank" in e.lower() for e in errors)
    os.unlink(path)


def test_validate_catches_non_monotonic_score():
    rows = _make_rows()
    rows[5]["score"] = 0.999  # score higher than rank-1 score
    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode="w") as f:
        path = f.name
    write_submission(rows, path)
    errors = validate_submission(path)
    assert any("score" in e.lower() or "monoton" in e.lower() for e in errors)
    os.unlink(path)
```

- [ ] **Step 2: Run tests to verify fail**

```bash
pytest tests/test_submission_writer.py -v
```

Expected: `ImportError`

- [ ] **Step 3: Implement submission_writer.py**

```python
# src/submission_writer.py
import csv
from typing import Any, Dict, List


def write_submission(rows: List[Dict[str, Any]], output_path: str) -> None:
    """Write ranked rows to CSV. rows must be sorted by rank ascending."""
    fieldnames = ["candidate_id", "rank", "score", "reasoning"]
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({
                "candidate_id": row["candidate_id"],
                "rank": int(row["rank"]),
                "score": f"{float(row['score']):.4f}",
                "reasoning": str(row.get("reasoning", "")),
            })


def validate_submission(csv_path: str) -> List[str]:
    """Return list of error strings. Empty list = valid."""
    errors = []
    rows = []

    with open(csv_path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        required_cols = {"candidate_id", "rank", "score", "reasoning"}
        if not required_cols.issubset(set(reader.fieldnames or [])):
            missing = required_cols - set(reader.fieldnames or [])
            errors.append(f"Missing columns: {missing}")
            return errors
        rows = list(reader)

    if len(rows) != 100:
        errors.append(f"Expected 100 rows, got {len(rows)}")

    ranks = [int(r["rank"]) for r in rows]
    if sorted(ranks) != list(range(1, 101)):
        errors.append(f"Ranks must be exactly 1–100 each appearing once. Got: {sorted(ranks)[:5]}...")

    if len(set(r["candidate_id"] for r in rows)) != len(rows):
        errors.append("Duplicate candidate_ids found")

    scores = [float(r["score"]) for r in rows]
    rows_sorted = sorted(rows, key=lambda r: int(r["rank"]))
    scores_sorted = [float(r["score"]) for r in rows_sorted]
    for i in range(len(scores_sorted) - 1):
        if scores_sorted[i] < scores_sorted[i + 1] - 1e-6:
            errors.append(
                f"Score not monotonically non-increasing at rank {i+1} "
                f"({scores_sorted[i]:.4f}) vs rank {i+2} ({scores_sorted[i+1]:.4f})"
            )
            break

    return errors
```

- [ ] **Step 4: Run tests to verify pass**

```bash
pytest tests/test_submission_writer.py -v
```

Expected: All 5 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/submission_writer.py tests/test_submission_writer.py
git commit -m "feat: submission writer with spec-compliant validator"
```

---

### Task 9: Precompute Script (Phase 1)

**Files:**
- Create: `precompute.py`

> This runs offline once, no time limit. Outputs artifacts/ that rank.py loads.

- [ ] **Step 1: Implement precompute.py**

```python
#!/usr/bin/env python3
# precompute.py
"""
Phase 1: Offline pre-computation. Run once before submission.
Outputs to artifacts/:
  - jd_profile.json       : Structured JD requirements
  - candidate_ids.json    : Ordered list of all candidate IDs
  - candidate_embeddings.npy : float32 array (N, 384)
  - faiss.index           : FAISS IndexFlatIP
  - feature_matrix.npy    : float32 array (N, num_features)
  - honeypot_flags.json   : Set of honeypot candidate_ids

Runtime: ~5-10 minutes for 100K candidates on CPU.
Memory: ~4GB peak (embeddings in RAM during FAISS build).
"""
import argparse
import json
import os
import time
import numpy as np
from rich.console import Console
from rich.progress import track

from src.jd_parser import parse_jd, save_jd_profile
from src.candidate_loader import load_candidates, build_embedding_text
from src.embedder import embed_texts
from src.faiss_index import build_index, save_index
from src.honeypot_detector import build_honeypot_flags
from src.scorer import (
    score_skill_match, score_career_trajectory,
    score_behavioral, score_constraint_match, compute_disqualifier_penalty
)

console = Console()


def build_feature_vector(candidate, jd_profile) -> list:
    """Extract numeric features for each candidate. Returns list of floats."""
    sigs = candidate.redrob_signals
    skills = candidate.skills
    career = candidate.career_history
    assessment_scores = sigs.get("skill_assessment_scores", {})

    skill_score = score_skill_match(skills, jd_profile, assessment_scores)
    trajectory_score = score_career_trajectory(career, candidate.years_of_experience, jd_profile)
    behavioral_score = score_behavioral(sigs)
    constraint_score = score_constraint_match(
        candidate.years_of_experience, candidate.location,
        candidate.country, sigs, jd_profile
    )
    disqualifier = compute_disqualifier_penalty(career, skills, candidate.years_of_experience, jd_profile)

    return [skill_score, trajectory_score, behavioral_score, constraint_score, disqualifier]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", default="data/candidates.jsonl")
    parser.add_argument("--jd", default="data/job_description.docx")
    parser.add_argument("--artifacts", default="artifacts")
    parser.add_argument("--batch-size", type=int, default=512)
    args = parser.parse_args()

    os.makedirs(args.artifacts, exist_ok=True)
    t0 = time.time()

    # Step 1: Parse JD
    console.print("[bold cyan]Step 1/5: Parsing job description...")
    jd_profile = parse_jd(args.jd)
    save_jd_profile(jd_profile, os.path.join(args.artifacts, "jd_profile.json"))
    console.print(f"  JD parsed: {len(jd_profile.required_skills)} required skills")

    # Step 2: Detect honeypots
    console.print("[bold cyan]Step 2/5: Detecting honeypots...")
    honeypot_ids = build_honeypot_flags(args.candidates)
    with open(os.path.join(args.artifacts, "honeypot_flags.json"), "w") as f:
        json.dump(list(honeypot_ids), f)
    console.print(f"  Honeypots flagged: {len(honeypot_ids)}")

    # Step 3: Load candidates + build texts + features
    console.print("[bold cyan]Step 3/5: Loading candidates and extracting features...")
    candidate_ids = []
    embedding_texts = []
    feature_rows = []

    candidates_iter = load_candidates(args.candidates)
    for c in track(candidates_iter, description="Loading...", total=100000):
        candidate_ids.append(c.candidate_id)
        embedding_texts.append(build_embedding_text(c))
        feature_rows.append(build_feature_vector(c, jd_profile))

    with open(os.path.join(args.artifacts, "candidate_ids.json"), "w") as f:
        json.dump(candidate_ids, f)
    feature_matrix = np.array(feature_rows, dtype=np.float32)
    np.save(os.path.join(args.artifacts, "feature_matrix.npy"), feature_matrix)
    console.print(f"  Features extracted: {feature_matrix.shape}")

    # Step 4: Embed all candidates
    console.print("[bold cyan]Step 4/5: Embedding candidates (this takes ~5-8 min)...")
    embeddings = embed_texts(embedding_texts, batch_size=args.batch_size)
    np.save(os.path.join(args.artifacts, "candidate_embeddings.npy"), embeddings)
    console.print(f"  Embeddings shape: {embeddings.shape}")

    # Step 5: Build FAISS index
    console.print("[bold cyan]Step 5/5: Building FAISS index...")
    index = build_index(embeddings)
    save_index(index, os.path.join(args.artifacts, "faiss.index"))
    console.print(f"  FAISS index built: {index.ntotal} vectors")

    elapsed = time.time() - t0
    console.print(f"\n[bold green]Pre-computation complete in {elapsed:.1f}s")
    console.print(f"Artifacts saved to: {args.artifacts}/")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run precompute (will take 5-10 min, this is expected)**

```bash
# First: symlink or copy data
ln -sf "/home/kartik/Kartik/Claude-projects/H2S-Hackthon/[PUB] India_runs_data_and_ai_challenge/[PUB] India_runs_data_and_ai_challenge/India_runs_data_and_ai_challenge/candidates.jsonl" data/candidates.jsonl
ln -sf "/home/kartik/Kartik/Claude-projects/H2S-Hackthon/[PUB] India_runs_data_and_ai_challenge/[PUB] India_runs_data_and_ai_challenge/India_runs_data_and_ai_challenge/job_description.docx" data/job_description.docx

python precompute.py
```

Expected output:
```
Step 1/5: Parsing job description...
  JD parsed: 30 required skills
Step 2/5: Detecting honeypots...
  Honeypots flagged: ~80
Step 3/5: Loading candidates and extracting features...
  Features extracted: (100000, 5)
Step 4/5: Embedding candidates...
  Embeddings shape: (100000, 384)
Step 5/5: Building FAISS index...
  FAISS index built: 100000 vectors
Pre-computation complete in ~420s
```

- [ ] **Step 3: Verify artifacts exist**

```bash
ls -lh artifacts/
# Should show: jd_profile.json, candidate_ids.json, candidate_embeddings.npy,
# faiss.index, feature_matrix.npy, honeypot_flags.json
```

- [ ] **Step 4: Commit**

```bash
git add precompute.py artifacts/jd_profile.json artifacts/candidate_ids.json artifacts/honeypot_flags.json
# Note: DO NOT commit large binary artifacts (embeddings.npy, faiss.index)
# Add to .gitignore instead — too large for GitHub
echo "artifacts/candidate_embeddings.npy" >> .gitignore
echo "artifacts/faiss.index" >> .gitignore
echo "artifacts/feature_matrix.npy" >> .gitignore
git add .gitignore
git commit -m "feat: precompute script for phase 1 offline artifacts"
```

---

### Task 10: rank.py — Main Ranking Entry Point (Phase 2)

**Files:**
- Create: `rank.py`

> This is the file judges will run. Must complete in ≤5 min, CPU only, no network.

- [ ] **Step 1: Implement rank.py**

```python
#!/usr/bin/env python3
# rank.py
"""
Phase 2: Ranking step. Runs within 5-minute CPU budget, no network access.
Usage: python rank.py --candidates ./data/candidates.jsonl --out ./submission.csv

Requires precomputed artifacts in ./artifacts/:
  - jd_profile.json
  - candidate_ids.json
  - candidate_embeddings.npy
  - faiss.index
  - feature_matrix.npy
  - honeypot_flags.json
"""
import argparse
import json
import time
import numpy as np
from rich.console import Console

from src.jd_parser import load_jd_profile
from src.embedder import embed_texts
from src.faiss_index import load_index, search_index
from src.scorer import compute_final_score
from src.candidate_loader import load_candidates, build_embedding_text
from src.reasoning_generator import generate_reasoning
from src.submission_writer import write_submission, validate_submission

console = Console()

TOP_K_RETRIEVE = 1000  # ANN retrieval pool size before re-ranking
TOP_N_SUBMIT = 100     # Final submission size


def main():
    parser = argparse.ArgumentParser(description="RecruitBrain candidate ranker")
    parser.add_argument("--candidates", default="data/candidates.jsonl")
    parser.add_argument("--artifacts", default="artifacts")
    parser.add_argument("--out", default="submission.csv")
    args = parser.parse_args()

    t0 = time.time()
    console.print("[bold cyan]RecruitBrain Ranker — Phase 2[/bold cyan]")

    # Load precomputed artifacts
    console.print("Loading artifacts...")
    jd_profile = load_jd_profile(f"{args.artifacts}/jd_profile.json")

    with open(f"{args.artifacts}/candidate_ids.json") as f:
        candidate_ids = json.load(f)
    id_to_idx = {cid: i for i, cid in enumerate(candidate_ids)}

    embeddings = np.load(f"{args.artifacts}/candidate_embeddings.npy")
    feature_matrix = np.load(f"{args.artifacts}/feature_matrix.npy")
    faiss_index = load_index(f"{args.artifacts}/faiss.index")

    with open(f"{args.artifacts}/honeypot_flags.json") as f:
        honeypot_ids = set(json.load(f))

    console.print(f"  Loaded {len(candidate_ids):,} candidates, {len(honeypot_ids)} honeypots flagged")

    # Embed JD query
    console.print("Embedding JD query...")
    jd_embedding = embed_texts([jd_profile.jd_embedding_text])[0]

    # ANN retrieval: top-K candidates
    console.print(f"ANN retrieval: top {TOP_K_RETRIEVE} candidates...")
    semantic_scores, ann_indices = search_index(faiss_index, jd_embedding, top_k=TOP_K_RETRIEVE)

    # Build candidate pool with all signals
    console.print("Loading candidate profiles for re-ranking pool...")
    # Load only the candidates we need (top-K pool)
    pool_indices = set(ann_indices.tolist())
    pool_candidates = {}
    for c in load_candidates(args.candidates):
        idx = id_to_idx.get(c.candidate_id)
        if idx in pool_indices:
            pool_candidates[c.candidate_id] = c
        if len(pool_candidates) >= TOP_K_RETRIEVE:
            break

    # Re-rank pool with multi-signal scorer
    console.print("Re-ranking with multi-signal scorer...")
    ranked = []
    for i, idx in enumerate(ann_indices):
        cid = candidate_ids[idx]
        sem_score = float(semantic_scores[i])

        # Honeypot hard disqualify
        if cid in honeypot_ids:
            continue

        c = pool_candidates.get(cid)
        if c is None:
            continue

        features = feature_matrix[idx]  # [skill, trajectory, behavioral, constraint, disqualifier]
        skill_s, traj_s, behav_s, constr_s, disq_mult = features

        # Final weighted score
        final_score = (
            0.35 * sem_score
            + 0.25 * skill_s
            + 0.15 * traj_s
            + 0.15 * behav_s
            + 0.10 * constr_s
        ) * disq_mult

        ranked.append((final_score, cid, c))

    # Sort by score descending
    ranked.sort(key=lambda x: -x[0])
    top_100 = ranked[:TOP_N_SUBMIT]

    # Generate submission rows with reasoning
    console.print("Generating reasoning strings...")
    rows = []
    for rank_pos, (score, cid, candidate) in enumerate(top_100, start=1):
        reasoning = generate_reasoning(candidate, rank=rank_pos, score=score)
        rows.append({
            "candidate_id": cid,
            "rank": rank_pos,
            "score": round(score, 4),
            "reasoning": reasoning,
        })

    # Write + validate
    console.print(f"Writing submission to {args.out}...")
    write_submission(rows, args.out)
    errors = validate_submission(args.out)
    if errors:
        console.print(f"[bold red]VALIDATION ERRORS:[/bold red]")
        for e in errors:
            console.print(f"  [red]{e}[/red]")
        raise SystemExit(1)

    elapsed = time.time() - t0
    console.print(f"\n[bold green]Done in {elapsed:.1f}s[/bold green]")
    console.print(f"Submission: {args.out} — {len(rows)} candidates ranked")
    console.print(f"Top-5 preview:")
    for r in rows[:5]:
        console.print(f"  #{r['rank']} {r['candidate_id']} score={r['score']} — {r['reasoning'][:80]}...")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run rank.py and time it**

```bash
time python rank.py --candidates data/candidates.jsonl --out submission.csv
```

Expected:
```
RecruitBrain Ranker — Phase 2
Loading artifacts... (2-5s)
Embedding JD query... (1s)
ANN retrieval: top 1000 candidates... (<1s)
Re-ranking with multi-signal scorer... (5-10s)
Writing submission to submission.csv...
Done in ~30s
Submission: submission.csv — 100 candidates ranked
```

Total must be well under 5 minutes.

- [ ] **Step 3: Verify submission passes validator**

```bash
python -c "
from src.submission_writer import validate_submission
errors = validate_submission('submission.csv')
if errors:
    for e in errors: print('ERROR:', e)
else:
    print('VALID: submission.csv passes all checks')
"
```

Expected: `VALID: submission.csv passes all checks`

- [ ] **Step 4: Run hackathon's own validator**

```bash
python "/home/kartik/Kartik/Claude-projects/H2S-Hackthon/[PUB] India_runs_data_and_ai_challenge/[PUB] India_runs_data_and_ai_challenge/India_runs_data_and_ai_challenge/validate_submission.py" submission.csv
```

Expected: No errors from their validator.

- [ ] **Step 5: Commit**

```bash
git add rank.py submission.csv
git commit -m "feat: rank.py phase-2 entry point, end-to-end ranking pipeline"
```

---

### Task 11: README + submission_metadata.yaml

**Files:**
- Create: `README.md`
- Create: `submission_metadata.yaml`

- [ ] **Step 1: Create README.md**

```markdown
# RecruitBrain — Redrob AI Hackathon

AI candidate ranking system for the Intelligent Candidate Discovery & Ranking Challenge.

## Architecture

Two-phase system:

**Phase 1 — Pre-computation** (offline, no time limit)
- Parses job description into structured requirements
- Detects honeypot candidates via impossible-profile checks
- Embeds all 100K candidates using BGE-small (ONNX via fastembed)
- Builds FAISS flat-IP index for ANN retrieval
- Extracts 5-dimensional feature vector per candidate

**Phase 2 — Ranking** (≤5 min CPU, no network)
- Loads precomputed artifacts
- ANN retrieval: top-1000 candidates via FAISS
- Multi-signal re-ranking: semantic (35%) + skill match (25%) + career trajectory (15%) + behavioral signals (15%) + constraints (10%), multiplied by disqualifier penalty
- Honeypot hard-disqualification
- Generates fact-grounded reasoning per candidate

## Setup

```bash
pip install -r requirements.txt
```

## Pre-computation (run once)

```bash
python precompute.py --candidates ./data/candidates.jsonl --jd ./data/job_description.docx
```

## Ranking

```bash
python rank.py --candidates ./data/candidates.jsonl --out ./submission.csv
```

## Tests

```bash
pytest tests/ -v
```

## Compute Requirements

- Python 3.12
- CPU only (no GPU)
- 16 GB RAM
- ~600 MB disk for model + artifacts
- rank.py runtime: ~30s
```

- [ ] **Step 2: Fill submission_metadata.yaml**

```yaml
team_name: "recruitbrain"

primary_contact:
  name: "Kartik"
  email: "kartik2@quark.com"
  phone: "+91-XXXXXXXXXX"

team_members:
  - name: "Kartik"
    email: "kartik2@quark.com"
    role: "Data & AI Engineer"

github_repo: "https://github.com/Kartikd09/redrob-ranker"
sandbox_link: "https://huggingface.co/spaces/Kartikd09/redrob-ranker"
reproduce_command: "python rank.py --candidates ./data/candidates.jsonl --out ./submission.csv"

compute:
  platform: "Linux x86_64"
  cpu_cores: 8
  ram_gb: 16
  python_version: "3.12.3"
  os: "Ubuntu 22.04 LTS"
  uses_gpu_for_inference: false
  has_network_during_ranking: false
  pre_computation_required: true
  pre_computation_time_minutes: 10

ai_tools_used:
  - "Claude"

ai_usage_summary: |
  Used Claude for architecture design, code review, and implementation.
  No candidate data was fed to any LLM API. All ranking runs fully offline.

methodology_summary: |
  Two-phase CPU-only ranker. Phase 1 (offline): BGE-small embeddings via
  fastembed/ONNX for all 100K candidates, FAISS flat-IP index, 5-feature
  matrix (skill match, career trajectory, behavioral signals, constraints,
  disqualifier penalty). Honeypot detector flags impossible profiles.
  Phase 2 (rank.py, <60s): FAISS ANN retrieval top-1000, multi-signal
  re-rank with weights tuned to JD requirements (semantic 35%, skill 25%,
  trajectory 15%, behavioral 15%, constraints 10%), honeypot hard-disqualify,
  fact-grounded reasoning generation. Explicit JD disqualifier detection
  penalizes consulting-only careers, CV/speech specialists, title-chasers.

declarations:
  read_submission_spec: true
  code_is_original_work: true
  no_collusion: true
  honeypot_check_done: true
  reproduction_tested: true
```

- [ ] **Step 3: Commit**

```bash
git add README.md submission_metadata.yaml
git commit -m "docs: README and submission metadata"
```

---

### Task 12: GitHub Repo + HuggingFace Sandbox

- [ ] **Step 1: Create GitHub repo**

```bash
gh repo create redrob-ranker --public --description "Redrob AI Hackathon — Intelligent Candidate Ranking"
git remote add origin git@github.com:Kartikd09/redrob-ranker.git
git push -u origin main
```

- [ ] **Step 2: Create HuggingFace Space (Streamlit demo)**

Create `app.py` for the sandbox demo:

```python
# app.py — Streamlit sandbox for Stage 3 verification
import streamlit as st
import json
import tempfile
import os

st.title("RecruitBrain — Candidate Ranker Demo")
st.write("Upload a small candidates JSONL (≤100 candidates) to see rankings.")

uploaded = st.file_uploader("Upload candidates.jsonl", type=["jsonl"])

if uploaded:
    with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False, mode="wb") as f:
        f.write(uploaded.read())
        tmp_path = f.name

    import subprocess
    result = subprocess.run(
        ["python", "rank.py", "--candidates", tmp_path, "--out", "/tmp/demo_out.csv"],
        capture_output=True, text=True, timeout=300
    )

    if result.returncode == 0:
        import pandas as pd
        df = pd.read_csv("/tmp/demo_out.csv")
        st.success(f"Ranked {len(df)} candidates")
        st.dataframe(df)
        st.download_button("Download CSV", df.to_csv(index=False), "submission.csv")
    else:
        st.error(f"Error: {result.stderr}")
    os.unlink(tmp_path)
```

- [ ] **Step 3: Push app.py to HuggingFace Space**

```bash
git add app.py
git commit -m "feat: streamlit sandbox demo for HuggingFace Spaces"
git push
```

---

## Self-Review Checklist

### Spec Coverage

| Requirement | Task |
|---|---|
| Read JD and understand what role needs | Task 2 (jd_parser.py) |
| Full picture — career, skills, behavioral, platform activity | Task 6 (scorer.py — 5 components) |
| Deliverable top-100 shortlist recruiter can trust | Task 10 (rank.py) |
| Honeypot detection | Task 4 (honeypot_detector.py) |
| No network during ranking | Task 10 (rank.py loads only local artifacts) |
| ≤5 min CPU runtime | Task 10 (FAISS ANN + precomputed features = ~30s) |
| ≤16 GB RAM | embeddings (100K×384×4 = ~150MB) + FAISS (~150MB) + features (~2MB) |
| Pre-computation documented | Task 9 (precompute.py) + README |
| Fact-grounded reasoning column | Task 7 (reasoning_generator.py) |
| Submit CSV with exact 100 rows, ranks 1-100, non-increasing scores | Task 8 (submission_writer.py + validator) |
| GitHub repo + sandbox link | Task 12 |
| submission_metadata.yaml | Task 11 |

### Anti-Pattern Checks

- Consulting-only career → 0.20× penalty (Task 6)
- CV/speech primary focus → 0.25× penalty (Task 6)
- Title-chaser (avg tenure <18mo) → 0.55× penalty (Task 6)
- LangChain-only, no retrieval depth → 0.40× penalty (Task 6)
- Not open to work + stale profile → behavioral score near 0 (Task 6)
- Honeypots → score = 0, hard excluded from top-100 (Task 4 + Task 10)

### Score Weights Alignment with NDCG@10 (50% of total score)

Top-10 matters most. Design ensures:
- Semantic score (35%) catches implicit fit even when exact keywords absent
- Disqualifier penalties (×0.15–0.25) push consulting/CV candidates far down
- Honeypots excluded entirely → can't contaminate top-10
