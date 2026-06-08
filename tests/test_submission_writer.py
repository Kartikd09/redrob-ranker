import pytest, csv, os, tempfile
from src.submission_writer import write_submission, validate_submission


def _make_rows(n=100):
    return [
        {
            "candidate_id": f"CAND_{str(i).zfill(7)}",
            "rank": i,
            "score": round(1.0 - (i - 1) * 0.009, 4),
            "reasoning": f"Candidate {i} reasoning.",
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
    assert errors == [], f"Unexpected errors: {errors}"
    os.unlink(path)


def test_validate_catches_duplicate_rank():
    rows = _make_rows()
    rows[5]["rank"] = 1
    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode="w") as f:
        path = f.name
    write_submission(rows, path)
    errors = validate_submission(path)
    assert any("rank" in e.lower() or "1-100" in e for e in errors)
    os.unlink(path)


def test_validate_catches_non_monotonic_score():
    rows = _make_rows()
    rows[5]["score"] = 0.999
    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode="w") as f:
        path = f.name
    write_submission(rows, path)
    errors = validate_submission(path)
    assert any("score" in e.lower() or "monoton" in e.lower() for e in errors)
    os.unlink(path)


def test_validate_catches_wrong_row_count():
    rows = _make_rows(99)  # only 99 rows
    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode="w") as f:
        path = f.name
    write_submission(rows, path)
    errors = validate_submission(path)
    assert any("100" in e for e in errors)
    os.unlink(path)
