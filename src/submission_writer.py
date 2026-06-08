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
        errors.append(f"Ranks must be exactly 1-100 each appearing once.")

    if len(set(r["candidate_id"] for r in rows)) != len(rows):
        errors.append("Duplicate candidate_ids found")

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
