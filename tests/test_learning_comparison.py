import json
import runpy
from pathlib import Path


def test_actual_snapshot_comparison_is_deterministic_and_observations_are_synthetic():
    compare = runpy.run_path("scripts/compare_learning_readiness.py")["compare"]
    path = Path("tests/fixtures/ranking_v2/scale50_snapshot.json")
    before = path.read_bytes()
    report = compare(path)
    assert report == compare(path)
    assert path.read_bytes() == before
    assert report["bookCount"] == 50
    assert report["readerEvidence"] == "synthetic_scenarios_not_real_assessments"
    assert report["accuracyMeasured"] is False
    assert sum(row["mappedBooks"] for row in report["topics"].values()) > 0
    for topic in report["topics"].values():
        for book in topic["books"]:
            assert book["v2Wrong"]["status"] in {"foundation-gap", "check-first"}
            assert book["v2Correct"]["status"] in {"ready-to-explore", "check-first"}
            assert book["v2Wrong"]["coveredConcepts"] == book["v2Correct"]["coveredConcepts"]
    json.dumps(report)
