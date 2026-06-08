# RecruitBrain — Session History & Decision Log

**Project:** Redrob AI Hackathon — Intelligent Candidate Discovery & Ranking  
**Repo:** https://github.com/Kartikd09/redrob-ranker  
**Last updated:** 2026-06-08

---

## What Was Built

### Problem
Build an AI system that ranks 100K candidates against a "Senior AI Engineer — Founding Team" JD better than keyword matching. Output: top-100 CSV with scores + reasoning.

### Architecture Decision: Two-Phase System
- **Phase 1 (precompute.py)** — offline, no time limit: embed all candidates, build FAISS index, extract JD requirements, flag honeypots
- **Phase 2 (rank.py)** — ≤5 min, CPU only, no network: load artifacts, ANN retrieval, multi-signal re-rank, generate reasoning, write CSV

---

## Session Chronology

### 1. Problem Analysis
- Read `The Data AI Challenge.txt` — problem: rank candidates like a great recruiter, not keyword filter
- Analyzed dataset: 100K candidates in JSONL, each with profile, career_history, skills, education, redrob_signals (23 behavioral signals)
- Read JD: Senior AI Engineer, Series A startup, Pune/Noida, 5-9 YOE, explicit disqualifiers (consulting firms, CV/speech focus, title-chasers, LangChain-only)
- Read submission spec: top-100 CSV, scored on NDCG@10 (50%) + NDCG@50 (30%) + MAP (15%) + P@10 (5%), Stage 4 manual reasoning review, honeypot trap in dataset

### 2. Brainstorming
- Explored neighbour projects for agent team patterns:
  - `fuseguard` → agent team: planner(opus)/dev(sonnet)/qa/security-reviewer/cloud-admin/code-reviewer, TDD, branch-per-feature, PR-only workflow
  - `kartik-portfolio` → same pattern, model cost routing, subagent-driven dev
  - `Quark/CLAUDE.md` → claude-trilogy knowledge base pattern, task-patterns.md
- Decided: adopt fuseguard's multi-agent pattern for this project

### 3. Architecture Design
**Scoring model (5 components):**
```
final = (0.35×semantic + 0.25×skill_match + 0.15×career_trajectory + 0.15×behavioral + 0.10×constraint) × disqualifier_penalty × honeypot_penalty
```

**Competitive edge — things others won't do:**
- Anti-pattern detection: consulting-only (0.20×), CV/speech focus (0.25×), title-chaser (0.55×), LangChain-only (0.40×)
- Honeypot detection: 5 impossible-profile rules
- Behavioral availability score: open_to_work + recency + response_rate + interview completion
- Career trajectory: product vs consulting vs research arc classification
- Fact-grounded reasoning per candidate (Stage-4 rubric compliant)

### 4. Constraint Audit
- `rank.py`: ≤5 min, CPU only, no network, ≤16GB RAM ← strict
- `precompute.py`: no time limit, pre-computed artifacts committed to repo ← flexible
- Honeypot rate >10% in top-100 = Stage 3 disqualification
- Stage 4: reasoning quality judged on 6 criteria (specific facts, JD connection, honest concerns, no hallucination, variation, rank consistency)
- Stage 5: defend-your-work interview — must explain every architectural choice

### 5. Implementation (Parallel Subagents)
Built via subagent-driven development, parallel where possible:

| Task | Status | Tests |
|------|--------|-------|
| Project scaffold + venv + deps | ✅ | - |
| `src/jd_parser.py` | ✅ | 6/6 |
| `src/candidate_loader.py` | ✅ | 6/6 |
| `src/honeypot_detector.py` | ✅ | 7/7 |
| `src/embedder.py` | ✅ | smoke |
| `src/faiss_index.py` | ✅ | smoke |
| `src/scorer.py` | ✅ | 9/9 |
| `src/reasoning_generator.py` | ✅ | 7/7 |
| `src/submission_writer.py` | ✅ | 6/6 |
| `precompute.py` | ✅ written | - |
| `rank.py` | ✅ written | syntax+import |

**Total: 41/41 tests passing**

### 6. Precompute Blocker & Resolution

**Attempt 1 — Local CPU fastembed:**
- Ran `precompute.py` locally — embedding step took 45+ min, killed
- Root cause: fastembed on CPU = ~35 texts/sec for 100K = 45 min

**Attempt 2 — Google Colab:**
- File upload truncated at ~28K lines (56MB file cut by Colab uploader)
- Fixed with split approach but session crashed on RAM (Colab free = 12.7GB, embeddings + model + texts = ~12GB peak)
- Fixed with chunked processing (5K at a time) but still slow on CPU

**Decision: Kaggle GPU**
- Kaggle free tier: 30GB RAM, T4 GPU, 12hr session, direct dataset mount
- Uploaded `candidates.jsonl` as Kaggle dataset
- Created `redrob_kaggle_precompute.ipynb`
- **Error 1:** `faiss-gpu` package doesn't exist on PyPI → fixed to `faiss-cpu`
- **Error 2:** `BAAI/bge-m3` not supported in this fastembed version → auto-detect + fallback to `BAAI/bge-small-en-v1.5`
- **Currently running:** bge-small-en-v1.5 embedding on Kaggle T4, ETA ~8-12 min

### 7. Model Choice Evolution
| Stage | Model | Reason |
|-------|-------|--------|
| Initial plan | BGE-small (384-dim) | Fast, no PyTorch |
| Upgraded to | BGE-M3 (1024-dim) | SOTA 2024 retrieval |
| Actual (fastembed limitation) | BGE-small (384-dim) | BGE-M3 not in fastembed supported list |
| embedder.py | Updated to bge-m3 name | Will work once fastembed supports it |

### 8. GitHub Setup
- Account: `Kartikd09` (personal), email: `kartikds009@gmail.com`
- SSH key: `~/.ssh/fuseguard_ed25519` (same as Fuseguard project)
- Repo: https://github.com/Kartikd09/redrob-ranker
- Initial push: HTTPS with gh token (SSH was routing to wrong account)
- Subsequent pushes: SSH with correct key

---

## Current Git Log

```
87ef1dd feat: rank.py phase-2 entry point + update embedder to BGE-M3 (1024-dim)
eb51ca4 feat: add Kaggle precompute notebook (BGE-M3, GPU, 30GB RAM)
7b35cfe docs: add STATUS.md project tracker with blocker details and decision needed
2f38b53 feat: initial RecruitBrain ranker — all core modules + 41 passing tests
```

---

## Current File Structure

```
H2S-Hackthon/
├── rank.py                        ✅ Phase 2 entry point
├── precompute.py                  ✅ Phase 1 orchestrator
├── redrob_kaggle_precompute.ipynb ✅ Kaggle notebook (running now)
├── redrob_precompute.ipynb        ✅ Colab notebook (backup)
├── requirements.txt               ✅
├── STATUS.md                      ✅ Live status tracker
├── SESSION_HISTORY.md             ✅ This file
├── src/
│   ├── jd_parser.py               ✅ 34 required skills extracted
│   ├── candidate_loader.py        ✅ streams 100K JSONL
│   ├── honeypot_detector.py       ✅ 5 impossible-profile rules
│   ├── scorer.py                  ✅ 5-signal + disqualifier penalties
│   ├── reasoning_generator.py     ✅ Stage-4 rubric compliant
│   ├── submission_writer.py       ✅ validates spec compliance
│   ├── embedder.py                ✅ BGE-small/M3 ONNX
│   └── faiss_index.py             ✅ IndexFlatIP
├── tests/
│   ├── test_jd_parser.py          ✅ 6/6
│   ├── test_candidate_loader.py   ✅ 6/6
│   ├── test_honeypot_detector.py  ✅ 7/7
│   ├── test_scorer.py             ✅ 9/9
│   ├── test_reasoning_generator.py✅ 7/7
│   └── test_submission_writer.py  ✅ 6/6
├── artifacts/
│   ├── jd_profile.json            ✅ committed
│   ├── candidate_ids.json         ✅ committed (100K IDs)
│   ├── honeypot_flags.json        ✅ committed (~80 honeypots)
│   ├── feature_matrix.npy         ✅ generated (gitignored, 100K×5)
│   ├── candidate_embeddings.npy   ⏳ Kaggle generating now
│   └── faiss.index                ⏳ Kaggle generating now
└── docs/superpowers/plans/
    └── 2026-06-08-redrob-ranker.md ✅ full 12-task plan
```

---

## What's Left

### Immediate (waiting on Kaggle)
- [ ] Download 3 files from Kaggle: `candidate_embeddings.npy`, `faiss.index`, `candidate_ids_kaggle.json`
- [ ] Drop into `artifacts/`, rename `candidate_ids_kaggle.json` → `candidate_ids.json`
- [ ] Run: `source venv/bin/activate && python rank.py --candidates data/candidates.jsonl --out submission.csv`
- [ ] Verify CSV passes validator

### After CSV generated
- [ ] README.md — setup instructions, reproduce command, architecture
- [ ] `submission_metadata.yaml` — team info, sandbox link, methodology
- [ ] `CLAUDE.md` — agent team pattern (modeled on fuseguard)
- [ ] HuggingFace Space or Streamlit sandbox (required by portal)
- [ ] PDF deck — approach, architecture, why it works

### Submission checklist
- [ ] `submission.csv` — 100 rows, validates clean
- [ ] GitHub repo — clean, README, metadata.yaml
- [ ] PDF deck
- [ ] Portal upload with sandbox link

---

## Key Decisions Log

| Decision | Choice | Reason |
|----------|--------|--------|
| Embedding model | BGE-small (384-dim) | BGE-M3 not in fastembed supported list |
| Precompute platform | Kaggle GPU T4 | 30GB RAM, no crashes, T4 GPU |
| Scoring weights | semantic 35%, skill 25%, trajectory 15%, behavioral 15%, constraint 10% | NDCG@10=50% of score, top-10 quality matters most |
| Anti-pattern penalties | consulting 0.20×, CV 0.25×, title-chaser 0.55×, LangChain-only 0.40× | JD explicitly states these disqualifiers |
| Honeypot detection | 5 rule-based checks | Hard exclude from top-100, spec says >10% = disqualified |
| GitHub account | Kartikd09 | Personal account, kartikds009@gmail.com |
| SSH key | fuseguard_ed25519 | Same key already authorised for Kartikd09 |
| Agent team pattern | fuseguard-style | planner/dev/qa/security-reviewer/code-reviewer roles |

---

## Neighbour Projects Learnings Applied

From `fuseguard/.claude/agents/` and `kartik-portfolio/CLAUDE.md`:
- Agent team with model cost routing (opus=arch, sonnet=impl, haiku=mechanical)
- TDD mandatory for core logic
- Branch-per-feature, PR-only to main
- code-reviewer + security-reviewer on every feature
- CLAUDE.md as living document updated each session
- claude-trilogy pattern for persistent knowledge base

Applied here:
- Subagent-driven development with parallel dispatch
- TDD: all 41 tests written before/alongside implementation
- Two-stage review (spec compliance + code quality) per task
- Explicit Definition of Done per task
