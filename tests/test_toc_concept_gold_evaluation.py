"""Human gold sampling and multilabel evaluation stay deterministic and strict."""

import json
from pathlib import Path

import pytest

from bookmatch_ml.concept_v2.gold_evaluation import (
    LoadedTocConceptGoldReview,
    PredictionMatch,
    TocConceptGoldReview,
    build_toc_concept_gold_review,
    calculate_gold_metrics,
    evaluate_toc_concept_gold,
    load_toc_concept_gold_review,
)
from bookmatch_ml.concept_v2.graph import load_concept_graph
from bookmatch_ml.concept_v2.profile import load_concept_matching_config
from bookmatch_ml.config import ConfigError, load_feature_config
from bookmatch_ml.data.evidence import calculate_evidence_coverage
from bookmatch_ml.io import write_json
from bookmatch_ml.schemas import Book, BookEvidence, TocEntry

ROOT = Path(__file__).resolve().parents[1]
FEATURES = load_feature_config(ROOT / "configs/features.yaml")
GRAPH = load_concept_graph(ROOT / "configs/concept_graph.yaml", FEATURES)
MATCHING = load_concept_matching_config(ROOT / "configs/concept_matching.yaml")
HASH = "sha256:" + "a" * 64
OS_BOOK_1 = "book_00000000000000000001"
OS_BOOK_2 = "book_00000000000000000002"
LA_BOOK_1 = "book_00000000000000000003"
LA_BOOK_2 = "book_00000000000000000004"


def _book(book_id: str, topic: str, title: str, toc_titles: list[str]) -> BookEvidence:
    metadata = Book(
        book_id=book_id,
        isbn_10=None,
        isbn_13=None,
        title=title,
        subtitle=None,
        authors=[],
        publisher=None,
        published_year=None,
        language="en",
        topics=[topic],
    )
    toc = [
        TocEntry(
            toc_entry_id=f"{book_id}_toc_{index}",
            book_id=book_id,
            parent_entry_id=None,
            level=1,
            order_index=index,
            label=str(index + 1),
            title=toc_title,
            source_id=f"{book_id}_source",
        )
        for index, toc_title in enumerate(toc_titles)
    ]
    return BookEvidence(
        book_id=book_id,
        metadata=metadata,
        toc=toc,
        documents=[],
        coverage=calculate_evidence_coverage([], toc),
    )


def _evidence() -> list[BookEvidence]:
    return [
        _book(OS_BOOK_1, "operating-systems", "OS One", ["Processes", "Appendix"]),
        _book(OS_BOOK_2, "operating-systems", "OS Two", ["CPU Scheduling", "Notes"]),
        _book(LA_BOOK_1, "linear-algebra", "LA One", ["Matrices", "Exercises"]),
        _book(
            LA_BOOK_2,
            "linear-algebra",
            "LA Two",
            ["Systems of Linear Equations", "Preface"],
        ),
    ]


def _review(evidence: list[BookEvidence] | None = None) -> TocConceptGoldReview:
    return build_toc_concept_gold_review(
        evidence=evidence or _evidence(),
        graph=GRAPH,
        features=FEATURES,
        matching=MATCHING,
        toc_file_hash=HASH,
        canonical_input_hashes={"toc.jsonl": HASH},
        sample_size_by_topic={"linear-algebra": 4, "operating-systems": 4},
    )


def _write_payload(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def test_sampling_is_deterministic_and_uses_matched_and_unmatched_population():
    first = _review()
    second = _review(list(reversed(_evidence())))
    assert first.model_dump_json() == second.model_dump_json()
    assert len(first.entries) == 8
    assert {row.book_id for row in first.entries} == {
        OS_BOOK_1,
        OS_BOOK_2,
        LA_BOOK_1,
        LA_BOOK_2,
    }
    assert any(row.predicted_concept_ids for row in first.entries)
    assert any(not row.predicted_concept_ids for row in first.entries)
    assert {row.toc_entry_id for row in first.entries} == {
        entry.toc_entry_id for book in _evidence() for entry in book.toc
    }
    first_row = first.entries[0]
    assert first_row.source_id
    assert first_row.toc_path == [first_row.toc_title]
    assert first_row.traversal_position in {0, 1}


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("unknown_book", "unknown topic/book/TOC entries"),
        ("unknown_toc", "unknown topic/book/TOC entries"),
        ("wrong_topic", "entries do not match sample_size_by_topic"),
        ("identity_mismatch", "identity or prediction differs"),
        ("duplicate", "duplicate TOC gold review row"),
        ("unknown_concept", "unknown operating-systems concepts"),
        ("invalid_status", "Input should be 'unreviewed' or 'reviewed'"),
    ],
)
def test_review_validation_rejects_invalid_rows(tmp_path: Path, mutation: str, message: str):
    expected = _review()
    payload = expected.model_dump(mode="json")
    if mutation == "unknown_book":
        payload["entries"][4]["book_id"] = "missing_book"
    elif mutation == "unknown_toc":
        payload["entries"][4]["toc_entry_id"] = "missing_toc"
    elif mutation == "wrong_topic":
        payload["entries"][4]["topic"] = "linear-algebra"
    elif mutation == "identity_mismatch":
        payload["entries"][4]["toc_title"] = "Changed title"
    elif mutation == "duplicate":
        payload["entries"][1] = dict(payload["entries"][0])
    elif mutation == "unknown_concept":
        payload["entries"][4]["review_status"] = "reviewed"
        payload["entries"][4]["human_gold_concept_ids"] = ["missing concept"]
    else:
        payload["entries"][4]["review_status"] = "accepted"
    path = tmp_path / "review.json"
    _write_payload(path, payload)
    with pytest.raises(ConfigError, match=message):
        load_toc_concept_gold_review(path, expected, GRAPH)


def test_evaluation_rejects_incomplete_reviews(tmp_path: Path):
    expected = _review()
    path = tmp_path / "review.json"
    write_json(expected, path)
    loaded = load_toc_concept_gold_review(path, expected, GRAPH)
    with pytest.raises(ValueError, match="requires complete reviews"):
        evaluate_toc_concept_gold(loaded, GRAPH)


def test_multilabel_metrics_and_empty_gold_are_entry_set_based():
    rows = _review().entries[:4]

    def reviewed(row, predicted: list[str], gold: list[str]):
        return row.model_copy(
            update={
                "predicted_concept_ids": predicted,
                "prediction_matches": [
                    PredictionMatch(
                        concept_id=concept,
                        matching_alias=concept,
                        match_method="test",
                    )
                    for concept in predicted
                ],
                "human_gold_concept_ids": gold,
                "review_status": "reviewed",
            }
        )

    first = reviewed(rows[0], ["process", "thread"], ["process", "scheduling"])
    second = reviewed(rows[1], [], [])
    predicted_only = reviewed(rows[2], ["matrix"], [])
    gold_only = reviewed(rows[3], [], ["vector"])
    metrics = calculate_gold_metrics([first, second, predicted_only, gold_only], "test")
    assert metrics.reviewed_entry_count == 4
    assert metrics.exact_set_match_accuracy == 0.25
    assert metrics.micro_precision == pytest.approx(1 / 3)
    assert metrics.micro_recall == pytest.approx(1 / 3)
    assert metrics.micro_f1 == pytest.approx(1 / 3)
    assert metrics.false_positive_assignment_count == 2
    assert metrics.false_negative_assignment_count == 2
    assert metrics.predicted_unmatched_entry_count == 2
    assert metrics.human_unmatched_entry_count == 2


def test_review_and_evaluation_artifacts_are_byte_stable(tmp_path: Path):
    blank = _review()
    reviewed = blank.model_copy(
        update={
            "entries": [
                row.model_copy(
                    update={
                        "human_gold_concept_ids": row.predicted_concept_ids,
                        "review_status": "reviewed",
                    }
                )
                for row in blank.entries
            ]
        }
    )
    review_path = tmp_path / "review.json"
    write_json(reviewed, review_path)
    loaded = load_toc_concept_gold_review(review_path, blank, GRAPH)
    report = evaluate_toc_concept_gold(loaded, GRAPH)
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"
    write_json(report, first)
    write_json(report, second)
    assert first.read_bytes() == second.read_bytes()
    assert report.combined.exact_set_match_accuracy == 1
    assert report.review_file_hash == loaded.content_hash
    assert isinstance(loaded, LoadedTocConceptGoldReview)
