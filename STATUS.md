# RecruitBrain — Project Status Tracker

**Last updated:** 2026-06-08  
**Hackathon:** Redrob Intelligent Candidate Discovery & Ranking Challenge  
**Submission deadline:** TBD — check portal  
**Git:** local `main` branch, commit `2f38b53`

---

## Overall Progress

```
[██████████░░░░░░░░░░] ~55% complete
```

| Phase | Status | Notes |
|-------|--------|-------|
| 1 — Scaffold + deps | ✅ Done | venv, requirements.txt, src/, tests/, data symlinks |
| 2 — Core modules | ✅ Done | 8 modules, 41/41 tests passing |
| 3 — Pre-computation | 🔴 BLOCKED | Embedding step too slow on local CPU (45min+) |
| 4 — rank.py | ⬜ Pending | Blocked by Phase 3 |
| 5 — README + metadata + CLAUDE.md | ⬜ Pending | |
| 6 — GitHub repo + push | ⬜ Pending | gh CLI authenticated as Kartikd09 |
| 7 — Submission CSV | ⬜ Pending | Final output |
| 8 — HuggingFace sandbox | ⬜ Pending | Required for submission portal |

---

## What's Done ✅

### Core Modules (all in `src/`)

| File | Purpose | Tests | Status |
|------|---------|-------|--------|
| `src/jd_parser.py` | Parse JD docx → JDProfile dataclass, save/load JSON | 6/6 ✅ | Done |
| `src/candidate_loader.py` | Stream 100K JSONL → CandidateProfile, build embedding text | 6/6 ✅ | Done |
| `src/honeypot_detector.py` | Flag impossible profiles (5 rules) | 7/7 ✅ | Done |
| `src/scorer.py` | Multi-signal scorer: skill+career+behavioral+constraint+disqualifier | 9/9 ✅ | Done |
| `src/reasoning_generator.py` | Fact-grounded per-candidate reasoning (Stage-4 rubric) | 7/7 ✅ | Done |
| `src/submission_writer.py` | Write + validate spec-compliant CSV | 6/6 ✅ | Done |
| `src/embedder.py` | fastembed BGE-small ONNX wrapper, L2-normalized output | smoke ✅ | Done |
| `src/faiss_index.py` | FAISS IndexFlatIP build/save/load/search | smoke ✅ | Done |

**Total tests: 41/41 passing**

### Artifacts (in `artifacts/`)

| File | Status | Contents |
|------|--------|---------|
| `jd_profile.json` | ✅ | 34 required skills, disqualifiers, locations |
| `candidate_ids.json` | ✅ | 100K ordered IDs |
| `honeypot_flags.json` | ✅ | ~80 honeypot candidate_ids flagged |
| `feature_matrix.npy` | ✅ | 100K × 5 float32 (skill/traj/behavioral/constraint/disqualifier) — gitignored |
| `candidate_embeddings.npy` | ❌ MISSING | Blocked — see below |
| `faiss.index` | ❌ MISSING | Blocked — depends on embeddings |

### Plan doc
`docs/superpowers/plans/2026-06-08-redrob-ranker.md` — full 12-task plan with code

---

## 🔴 Current Blocker

### Problem: Embedding 100K candidates is too slow on local CPU

**What happened:** `precompute.py` ran for 45+ minutes without finishing the embedding step. fastembed downloads BGE-small-en-v1.5 (~33MB ONNX model) then encodes 100K texts. On this machine (16GB RAM, CPU only), it's processing ~35-40 texts/sec = ~45 min for 100K.

**Constraint check:**
- `rank.py` (Phase 2) = must run ≤5 min, CPU only, no network ← **strict**
- `precompute.py` (Phase 1) = no time limit stated, no network restriction on THIS step ← **flexible**

**This means we CAN:**
- Use GPU cloud (Colab, Kaggle, Lambda) for precompute — run embeddings there, download .npy, commit
- Use Claude/Anthropic API during precompute (not during rank.py)
- Use any approach that produces the artifacts — only rank.py is constrained

---

## 🗂 Decision Needed: Precompute Strategy

Three options. Pick one and we proceed immediately.

### Option A: TF-IDF (fastest, ~60s, already installed)
- scikit-learn TF-IDF vectorizer on all 100K texts
- Cosine sim via sparse matrix — no FAISS needed
- Installs: already have scikit-learn ✅
- Quality: good for keyword-heavy profiles, misses semantic nuance
- Time: ~60 seconds total

### Option B: Google Colab / Kaggle (best quality, ~5-10 min on GPU)
- Upload candidates.jsonl + requirements.txt to Colab
- Run embedding there (GPU = 30x faster)
- Download candidate_embeddings.npy + faiss.index (~150MB)
- Drop in artifacts/, commit
- Quality: best — full BGE-small semantic embeddings
- Time: 10-15 min total including upload/download

### Option C: Claude API during precompute (innovative, no GPU needed)
- Use Anthropic Batch API to score top-2K candidates (pre-filtered by feature_matrix)
- Claude reads each profile + JD, returns structured score + reasoning
- Commit scores as artifact → rank.py loads static scores
- Quality: highest possible — actual LLM judgment
- Cost: ~$2-5 for 2K candidates at batch pricing
- Time: ~20-30 min (batch API async)
- Network constraint: only in precompute (allowed), rank.py stays offline ✅

**Recommended: Option B (Colab) for embeddings + Option C (Claude API) for top-500 reasoning enrichment.**

---

## What's Left ⬜

### Task 9 (BLOCKED): Complete precompute artifacts
- [ ] Resolve embedding strategy (see Decision above)
- [ ] Generate `candidate_embeddings.npy` (100K × 384)
- [ ] Generate `artifacts/faiss.index`
- [ ] Verify all 6 artifacts present

### Task 10: rank.py — Phase 2 entry point
- [ ] Write `rank.py` — loads artifacts, ANN retrieval, re-rank, output CSV
- [ ] Must run ≤5 min, CPU only, no network
- [ ] Validate output with `submission_writer.validate_submission()`
- [ ] Run hackathon's own `validate_submission.py`

### Task 11: Docs + metadata
- [ ] `README.md` — setup, reproduce command, architecture
- [ ] `submission_metadata.yaml` — team info, repo URL, sandbox link
- [ ] `CLAUDE.md` — agent team pattern (modeled on fuseguard pattern)

### Task 12: GitHub + sandbox
- [ ] `gh repo create redrob-ranker --public` under Kartikd09
- [ ] Push all code + committed artifacts
- [ ] HuggingFace Space or Streamlit sandbox (required by submission portal)

### Final: Generate submission CSV
- [ ] Run `rank.py` end-to-end
- [ ] Validate CSV passes both internal + hackathon validators
- [ ] `submission.csv` ready to upload

---

## Architecture Summary

```
PHASE 1 — precompute.py (offline, no time limit)
  JD docx → jd_profile.json
  candidates.jsonl → candidate_ids.json + feature_matrix.npy (✅ done)
  candidates.jsonl → honeypot_flags.json (✅ done)
  candidates.jsonl → candidate_embeddings.npy → faiss.index  ← BLOCKED

PHASE 2 — rank.py (≤5 min, CPU, no network)
  Load artifacts → FAISS ANN top-1000 → multi-signal re-rank
  → honeypot disqualify → generate reasoning → write CSV
```

## Scoring Target
```
Final = 0.50×NDCG@10 + 0.30×NDCG@50 + 0.15×MAP + 0.05×P@10
```
Top-10 quality = 50% of score. Architecture tuned for this.

## Key Anti-Patterns We Detect (Competitive Edge)
- Consulting-only career (TCS/Infosys/Wipro etc.) → 0.20× penalty
- CV/speech primary focus → 0.25× penalty  
- Title-chaser (avg tenure <18mo) → 0.55× penalty
- LangChain-only, no retrieval depth → 0.40× penalty
- Honeypots → hard excluded from top-100

---

## Quick Commands

```bash
# Activate venv
source venv/bin/activate

# Run all tests
python -m pytest tests/ -v

# Run precompute (when blocker resolved)
python precompute.py

# Run ranker
python rank.py --candidates data/candidates.jsonl --out submission.csv

# Validate submission
python -c "from src.submission_writer import validate_submission; print(validate_submission('submission.csv'))"
```

---

## Files Committed (git log)

```
commit 2f38b53 — feat: initial RecruitBrain ranker — all core modules + 41 passing tests
  .gitignore
  artifacts/candidate_ids.json
  artifacts/honeypot_flags.json
  artifacts/jd_profile.json
  docs/superpowers/plans/2026-06-08-redrob-ranker.md
  precompute.py
  requirements.txt
  src/__init__.py + 8 module files
  tests/__init__.py + 6 test files
```
