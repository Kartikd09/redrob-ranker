#!/usr/bin/env python3
# rank.py
"""
Phase 2: Ranking step. CPU only, no network, ≤5 min.
Usage: python rank.py --candidates ./data/candidates.jsonl --out ./submission.csv
"""
import argparse
import json
import os
import time
import numpy as np
from rich.console import Console

from src.jd_parser import load_jd_profile
from src.embedder import embed_texts
from src.faiss_index import load_index, search_index
from src.candidate_loader import load_candidates
from src.reasoning_generator import generate_reasoning
from src.submission_writer import write_submission, validate_submission

console = Console()

TOP_K_RETRIEVE = 1000  # ANN retrieval pool
TOP_N_SUBMIT = 100     # final submission size


def load_candidate_ids(artifacts_dir: str) -> list:
    """Load candidate IDs — handle both naming conventions."""
    for name in ["candidate_ids.json", "candidate_ids_kaggle.json"]:
        path = os.path.join(artifacts_dir, name)
        if os.path.exists(path):
            with open(path) as f:
                return json.load(f)
    raise FileNotFoundError("No candidate_ids.json found in artifacts/")


def main():
    parser = argparse.ArgumentParser(description="RecruitBrain — Phase 2 ranker")
    parser.add_argument("--candidates", default="data/candidates.jsonl")
    parser.add_argument("--artifacts", default="artifacts")
    parser.add_argument("--out", default="submission.csv")
    args = parser.parse_args()

    t0 = time.time()
    console.print("[bold cyan]RecruitBrain Ranker — Phase 2[/bold cyan]")
    console.print(f"Candidates : {args.candidates}")
    console.print(f"Artifacts  : {args.artifacts}")
    console.print(f"Output     : {args.out}")

    # ── Load artifacts ──────────────────────────────────────────────────────
    console.print("\n[cyan]Loading artifacts...[/cyan]")

    jd_profile = load_jd_profile(os.path.join(args.artifacts, "jd_profile.json"))

    candidate_ids = load_candidate_ids(args.artifacts)
    id_to_idx = {cid: i for i, cid in enumerate(candidate_ids)}

    embeddings = np.load(os.path.join(args.artifacts, "candidate_embeddings.npy"))
    feature_matrix = np.load(os.path.join(args.artifacts, "feature_matrix.npy"))
    faiss_index = load_index(os.path.join(args.artifacts, "faiss.index"))

    with open(os.path.join(args.artifacts, "honeypot_flags.json")) as f:
        honeypot_ids = set(json.load(f))

    console.print(f"  Candidates  : {len(candidate_ids):,}")
    console.print(f"  Embeddings  : {embeddings.shape}")
    console.print(f"  Features    : {feature_matrix.shape}")
    console.print(f"  FAISS       : {faiss_index.ntotal} vectors")
    console.print(f"  Honeypots   : {len(honeypot_ids)}")
    console.print(f"  Load time   : {time.time()-t0:.1f}s")

    # ── Embed JD query ───────────────────────────────────────────────────────
    console.print("\n[cyan]Embedding JD query...[/cyan]")
    t1 = time.time()
    jd_embedding = embed_texts([jd_profile.jd_embedding_text])[0]
    console.print(f"  JD embedding shape: {jd_embedding.shape}, time: {time.time()-t1:.1f}s")

    # ── ANN retrieval: top-K candidates ──────────────────────────────────────
    console.print(f"\n[cyan]ANN retrieval: top {TOP_K_RETRIEVE}...[/cyan]")
    t2 = time.time()
    semantic_scores, ann_indices = search_index(faiss_index, jd_embedding, top_k=TOP_K_RETRIEVE)
    console.print(f"  Retrieved {len(ann_indices)} candidates in {time.time()-t2:.2f}s")
    console.print(f"  Score range: {semantic_scores.min():.4f} – {semantic_scores.max():.4f}")

    # ── Load candidate profiles for top-K pool ───────────────────────────────
    console.print("\n[cyan]Loading candidate profiles for re-ranking pool...[/cyan]")
    t3 = time.time()
    pool_idxs = set(int(i) for i in ann_indices if i >= 0)
    pool_candidates = {}

    for c in load_candidates(args.candidates):
        idx = id_to_idx.get(c.candidate_id)
        if idx is not None and idx in pool_idxs:
            pool_candidates[c.candidate_id] = c
        if len(pool_candidates) >= TOP_K_RETRIEVE:
            break

    console.print(f"  Loaded {len(pool_candidates)} profiles in {time.time()-t3:.1f}s")

    # ── Multi-signal re-rank ──────────────────────────────────────────────────
    console.print("\n[cyan]Re-ranking with multi-signal scorer...[/cyan]")
    t4 = time.time()
    ranked = []

    for i, idx in enumerate(ann_indices):
        if idx < 0:
            continue
        idx = int(idx)
        cid = candidate_ids[idx]

        # Hard disqualify honeypots
        if cid in honeypot_ids:
            continue

        c = pool_candidates.get(cid)
        if c is None:
            continue

        sem_score = float(semantic_scores[i])
        features = feature_matrix[idx]
        skill_s, traj_s, behav_s, constr_s, disq_mult = (
            float(features[0]), float(features[1]),
            float(features[2]), float(features[3]), float(features[4])
        )

        final_score = (
            0.35 * sem_score
            + 0.25 * skill_s
            + 0.15 * traj_s
            + 0.15 * behav_s
            + 0.10 * constr_s
        ) * disq_mult

        ranked.append((final_score, cid, c))

    # Sort descending by score
    ranked.sort(key=lambda x: -x[0])
    top_100 = ranked[:TOP_N_SUBMIT]

    console.print(f"  Re-ranked {len(ranked)} candidates in {time.time()-t4:.2f}s")
    console.print(f"  Top score: {top_100[0][0]:.4f}, #100 score: {top_100[-1][0]:.4f}")

    # ── Generate reasoning ────────────────────────────────────────────────────
    console.print("\n[cyan]Generating reasoning strings...[/cyan]")
    rows = []
    for rank_pos, (score, cid, candidate) in enumerate(top_100, start=1):
        reasoning = generate_reasoning(candidate, rank=rank_pos, score=score)
        rows.append({
            "candidate_id": cid,
            "rank": rank_pos,
            "score": round(score, 4),
            "reasoning": reasoning,
        })

    # ── Write + validate ──────────────────────────────────────────────────────
    console.print(f"\n[cyan]Writing {args.out}...[/cyan]")
    write_submission(rows, args.out)
    errors = validate_submission(args.out)

    if errors:
        console.print("[bold red]VALIDATION ERRORS:[/bold red]")
        for e in errors:
            console.print(f"  [red]{e}[/red]")
        raise SystemExit(1)

    elapsed = time.time() - t0
    console.print(f"\n[bold green]Done in {elapsed:.1f}s[/bold green]")
    console.print(f"Output: {args.out} — {len(rows)} candidates ranked")

    # Preview top 5
    console.print("\n[bold]Top 5 candidates:[/bold]")
    for r in rows[:5]:
        console.print(f"  #{r['rank']} {r['candidate_id']} score={r['score']:.4f} — {r['reasoning'][:80]}")

    console.print("\n[bold]Bottom 3 candidates:[/bold]")
    for r in rows[-3:]:
        console.print(f"  #{r['rank']} {r['candidate_id']} score={r['score']:.4f} — {r['reasoning'][:80]}")


if __name__ == "__main__":
    main()
