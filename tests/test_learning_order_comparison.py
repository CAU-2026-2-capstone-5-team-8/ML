import csv
import importlib.util
import json
from pathlib import Path

import pytest

from bookmatch_ml.concept_v2.learning_fit import LearningFitRequest


def module():
    spec = importlib.util.spec_from_file_location(
        "comparison", Path("scripts/compare_learning_order.py")
    )
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def test_comparison_uses_pinned_real_inputs_and_labels_all_responses_synthetic(tmp_path):
    tool = module()
    # Small synthetic fixture with the same handoff contract, not a substitute for the real run.
    books = [{"book_id": "one", "title": "Synthetic book", "topics": ["operating-systems"]}]
    candidates = [
        {
            "book_id": "one",
            "topic_distribution": {"operating-systems": 1},
            "covered_concepts": [{"concept": "scheduling", "weight": 1}],
        }
    ]
    for name, rows in [("books.jsonl", books), ("candidates.jsonl", candidates)]:
        (tmp_path / name).write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    manifest = {
        "contract_version": "local-catalog-import-v1",
        "snapshot_id": "synthetic",
        "books_path": "books.jsonl",
        "candidates_path": "candidates.jsonl",
        "books_sha256": tool.digest((tmp_path / "books.jsonl").read_bytes()),
        "candidates_sha256": tool.digest((tmp_path / "candidates.jsonl").read_bytes()),
        "selected_book_ids": ["one"],
        "topic": {"ml_topic_id": "operating-systems"},
    }
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    first = tool.compare(path, Path("configs/ranking_v2.yaml"))
    assert tool.compare(path, Path("configs/ranking_v2.yaml")) == first
    assert first["readerEvidence"] == "synthetic_scenarios_not_real_assessments"
    assert first["accuracyMeasured"] is False
    assert first["bookCount"] == 1
    assert first["mappedBookCount"] == 1
    assert first["scenarios"][0]["observations"] == []
    assert all(row["after"][0]["title"] == "Synthetic book" for row in first["scenarios"])
    output = tmp_path / "comparison"
    tool.write_outputs(first, output)
    with (output / "human-review.csv").open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 6
    assert all(row["suitability_1_to_5"] == "" and row["reviewer"] == "" for row in rows)
    with pytest.raises(FileExistsError):
        tool.write_outputs(first, output)
    (tmp_path / "candidates.jsonl").write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="hash mismatch"):
        tool.compare(path, Path("configs/ranking_v2.yaml"))


def test_v3_duplicate_book_ids_and_unknown_concepts_still_fail():
    payload = {
        "modelVersion": "concept-learning-v3",
        "topicId": "operating-systems",
        "observations": [],
        "candidateBooks": [
            {
                "bookId": "one",
                "coveredConcepts": ["not-a-concept"],
                "sourceArtifactVersion": "fixture",
                "sourceArtifactHash": "sha256:" + "a" * 64,
            }
        ],
    }
    from bookmatch_ml.concept_v2.learning_fit import recommend_learning
    from bookmatch_ml.config import load_ranking_v2_config

    config = load_ranking_v2_config(Path("configs/ranking_v2.yaml"))
    with pytest.raises(ValueError, match="unknown"):
        recommend_learning(LearningFitRequest.model_validate(payload), config)
    payload["candidateBooks"][0]["coveredConcepts"] = ["scheduling"]
    payload["candidateBooks"] *= 2
    with pytest.raises(ValueError, match="duplicate candidate"):
        recommend_learning(LearningFitRequest.model_validate(payload), config)
