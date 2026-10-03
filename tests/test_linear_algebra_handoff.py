"""Network-free regression of the executable handoff and actual profile-to-rank contract."""

import asyncio
import hashlib
import json
import runpy
from pathlib import Path

import httpx
import pytest

from bookmatch_ml.api import create_app

ROOT = Path(__file__).resolve().parents[1]
VERIFY = runpy.run_path(str(ROOT / "scripts/verify_linear_algebra_live.py"))
PREPARE = runpy.run_path(str(ROOT / "scripts/prepare_linear_algebra_handoff.py"))
HTTP = (ROOT / "examples/linear-algebra-live.http").read_text()
APP = create_app(
    ROOT / "configs/reader.yaml", ROOT / "configs/ranking.yaml", ROOT / "configs/ranking_v2.yaml"
)


def post(path: str, body: dict) -> dict:
    async def execute():
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=APP), base_url="http://offline-test"
        ) as client:
            response = await client.post(path, json=body)
            assert response.status_code == 200, response.text
            return response.json()

    return asyncio.run(execute())


def assessment() -> dict:
    return json.loads(
        HTTP.split("# @name laProfile\n", 1)[1].split("\n\n", 1)[1].split("\n\n###", 1)[0]
    )


def test_rest_client_bindings_use_actual_response_and_original_candidate_values():
    profile = post("/ml/reader-profile", assessment())
    rank = VERIFY["resolve_ml_http"](HTTP, profile)
    assert rank["readerProfile"] == VERIFY["matching_profile"](profile)
    assert len(rank["readerProfile"]["conceptReadiness"]) == 8
    assert "earnedWeight" not in rank["readerProfile"]["conceptReadiness"][0]
    assert {c["bookId"] for c in rank["candidateBooks"]} == set(
        json.loads((ROOT / "configs/experiments/linear-algebra-live-handoff-v1.json").read_text())[
            "selected_book_ids"
        ]
    )
    assert all(
        c["lexicalDifficulty"] is None
        and c["prerequisiteConcepts"] == []
        and c["featureVersion"] == "book-v1"
        for c in rank["candidateBooks"]
    )
    result = post("/ml/rank", rank)
    assert result == post("/ml/rank", rank)
    assert len(result["items"]) == 3
    assert result["diagnostics"]["personalizedCandidateShortage"] == 2
    assert any(item["directLearningOpportunity"] is None for item in result["items"])


def test_changed_answers_change_bound_readiness_and_missing_readiness_returns_empty():
    known = assessment()
    profile = post("/ml/reader-profile", known)
    for row in known["responses"]:
        row["correct"] = False
    changed = post("/ml/reader-profile", known)
    original_rank = VERIFY["resolve_ml_http"](HTTP, profile)
    changed_rank = VERIFY["resolve_ml_http"](HTTP, changed)
    assert original_rank["readerProfile"] != changed_rank["readerProfile"]
    assert original_rank["candidateBooks"] == changed_rank["candidateBooks"]
    changed_rank["readerProfile"]["conceptReadiness"] = []
    result = post("/ml/rank", changed_rank)
    assert result["items"] == []
    assert result["diagnostics"]["personalizedCandidateShortage"] == 5


def test_snapshot_gate_rejects_missing_or_changed_input_without_overwriting(tmp_path):
    path = tmp_path / "snapshot.jsonl"
    original = b'{"synthetic":true}\n'
    expected = "sha256:" + hashlib.sha256(original).hexdigest()
    with pytest.raises(ValueError, match="missing snapshot file"):
        PREPARE["checked"](path, expected)
    path.write_bytes(original)
    assert PREPARE["checked"](path, expected) == original
    path.write_bytes(b"changed")
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        PREPARE["checked"](path, expected)
    assert path.read_bytes() == b"changed"
