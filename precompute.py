#!/usr/bin/env python3
# precompute.py
"""
Phase 1: Offline pre-computation. Run once before submission.
Runtime: ~5-10 minutes for 100K candidates on CPU.
"""
import argparse
import json
import os
import time
import numpy as np
from rich.console import Console
from rich.progress import track

from src.jd_parser import parse_jd, save_jd_profile, load_jd_profile
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

    console.print("[bold cyan]Step 1/5: Parsing job description...")
    jd_profile = parse_jd(args.jd)
    save_jd_profile(jd_profile, os.path.join(args.artifacts, "jd_profile.json"))
    console.print(f"  JD parsed: {len(jd_profile.required_skills)} required skills")

    console.print("[bold cyan]Step 2/5: Detecting honeypots...")
    honeypot_ids = build_honeypot_flags(args.candidates)
    with open(os.path.join(args.artifacts, "honeypot_flags.json"), "w") as f:
        json.dump(list(honeypot_ids), f)
    console.print(f"  Honeypots flagged: {len(honeypot_ids)}")

    console.print("[bold cyan]Step 3/5: Loading candidates and extracting features...")
    candidate_ids = []
    embedding_texts = []
    feature_rows = []

    for c in track(load_candidates(args.candidates), description="Loading...", total=100000):
        candidate_ids.append(c.candidate_id)
        embedding_texts.append(build_embedding_text(c))
        feature_rows.append(build_feature_vector(c, jd_profile))

    with open(os.path.join(args.artifacts, "candidate_ids.json"), "w") as f:
        json.dump(candidate_ids, f)
    feature_matrix = np.array(feature_rows, dtype=np.float32)
    np.save(os.path.join(args.artifacts, "feature_matrix.npy"), feature_matrix)
    console.print(f"  Features extracted: {feature_matrix.shape}")

    console.print("[bold cyan]Step 4/5: Embedding candidates (takes ~5-8 min)...")
    embeddings = embed_texts(embedding_texts, batch_size=args.batch_size)
    np.save(os.path.join(args.artifacts, "candidate_embeddings.npy"), embeddings)
    console.print(f"  Embeddings shape: {embeddings.shape}")

    console.print("[bold cyan]Step 5/5: Building FAISS index...")
    index = build_index(embeddings)
    save_index(index, os.path.join(args.artifacts, "faiss.index"))
    console.print(f"  FAISS index built: {index.ntotal} vectors")

    elapsed = time.time() - t0
    console.print(f"\n[bold green]Pre-computation complete in {elapsed:.1f}s")
    console.print(f"Artifacts saved to: {args.artifacts}/")


if __name__ == "__main__":
    main()
