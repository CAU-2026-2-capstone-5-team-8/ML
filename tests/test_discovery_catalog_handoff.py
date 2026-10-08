"""Network-free coverage semantics and executable response bindings."""

import json
import runpy
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
try:
    PREPARE = runpy.run_path(str(ROOT / "scripts/prepare_discovery_catalog_handoff.py"))
finally:
    sys.path.pop(0)
VERIFY = runpy.run_path(str(ROOT / "scripts/verify_linear_algebra_live.py"))


def row(**values):
    return SimpleNamespace(**values)


def test_coverage_counts_canonical_ownership_and_preserves_zero():
    dataset = row(
        books=[row(book_id="b"), row(book_id="a")],
        toc=[row(book_id="a"), row(book_id="a")],
        documents=[
            row(book_id="a", document_type="description"),
            row(book_id="b", document_type="toc"),
        ],
        sources=[row(book_id="a"), row(book_id="a"), row(book_id="b")],
    )
    assert PREPARE["coverage_rows"](dataset) == [
        {"book_id": "a", "toc_entry_count": 2, "description_count": 1, "source_count": 2},
        {"book_id": "b", "toc_entry_count": 0, "description_count": 0, "source_count": 1},
    ]


def test_existing_output_is_never_overwritten(tmp_path):
    sentinel = tmp_path / "keep"
    sentinel.write_text("existing result")
    with pytest.raises(ValueError, match="output-dir exists"):
        PREPARE["prepare"](tmp_path, tmp_path, tmp_path, tmp_path)
    assert sentinel.read_text() == "existing result"


def test_shared_http_builder_binds_current_profile_instead_of_copying_scores():
    http = (ROOT / "examples/linear-algebra-live.http").read_text()
    assessment = json.loads(
        http.split("# @name laProfile\n", 1)[1].split("\n\n", 1)[1].split("\n\n###", 1)[0]
    )
    wire = [{"bookId": "synthetic-book", "coveredConcepts": [], "lexicalDifficulty": None}]
    rendered = PREPARE["build_ml_http"](assessment, wire)
    concepts = sorted({r["conceptId"] for r in assessment["responses"]})
    profile = {
        "userId": 13,
        "topicId": "linear-algebra",
        "vocabulary": 0.4,
        "backgroundKnowledge": 0.2,
        "comprehension": 0.6,
        "profileVersion": "reader-v1",
        "configVersion": "actual-config",
        "configHash": "sha256:" + "a" * 64,
        "conceptReadiness": [{"conceptId": c, "score": 0.2} for c in concepts],
    }
    assert VERIFY["resolve_ml_http"](rendered, profile) == VERIFY["rank_request"](profile, wire)
