"""Fresh matcher holdout sampling and blinded-review regression tests."""

from datetime import UTC, datetime
from pathlib import Path

import pytest

from bookmatch_ml.concept_v2.evidence_evaluation import build_evidence_concept_gold_review
from bookmatch_ml.concept_v2.graph import load_concept_graph
from bookmatch_ml.concept_v2.holdout import (
    _challenge_categories,
    _population,
    build_holdout_manifests,
    build_holdout_predictions,
    build_holdout_review,
    load_holdout_review,
    review_packet,
    update_holdout_review_entry,
)
from bookmatch_ml.concept_v2.holdout_evaluation import build_holdout_evaluation_report
from bookmatch_ml.concept_v2.matcher_experiments import load_matcher_v2_experiment_config
from bookmatch_ml.concept_v2.profile import load_concept_matching_config
from bookmatch_ml.config import load_feature_config
from bookmatch_ml.data.book_evidence import ImportedBookEvidence, ImportedEvidenceItem
from bookmatch_ml.schemas import Book

ROOT = Path(__file__).resolve().parents[1]
FEATURES = load_feature_config(ROOT / "configs/features.yaml")
GRAPH = load_concept_graph(ROOT / "configs/concept_graph.yaml", FEATURES)
MATCHING = load_concept_matching_config(ROOT / "configs/concept_matching.yaml")
EXPERIMENT = load_matcher_v2_experiment_config(ROOT / "configs/matcher_v2_experiments.yaml")
HASH = "sha256:" + "a" * 64


def _item(number: int, evidence_type: str, text: str, path: list[str] | None = None):
    values = {
        "evidence_id": f"evidence_{number:020x}",
        "evidence_type": evidence_type,
        "text": text,
        "source_id": f"source_{number}",
        "provider": "fixture",
        "source_type": "metadata_api",
        "source_url": f"https://example.test/{number}",
        "source_retrieved_at": datetime(2026, 9, 22, tzinfo=UTC),
        "source_content_hash": HASH,
        "provenance_hash": HASH,
    }
    if evidence_type.startswith("toc_"):
        values.update(
            edition_relation="exact",
            toc_entry_id=f"toc_{number}",
            level=len(path or [text]),
            order_index=number,
            toc_path=path or [text],
        )
    elif evidence_type == "description":
        values.update(
            edition_relation="unspecified",
            document_id=f"doc_{number}",
            document_type="description",
            document_content_hash=HASH,
        )
    else:
        values.update(
            edition_relation="canonical_record",
            metadata_field="topics" if evidence_type == "subject" else "title",
        )
    return ImportedEvidenceItem.model_validate(values)


def _records() -> list[ImportedBookEvidence]:
    records = []
    number = 1
    for index, topic in enumerate(("linear-algebra", "operating-systems") * 3, start=1):
        book = Book(
            book_id=f"book_{index:020x}",
            title=f"Fixture {index}",
            authors=["Author"],
            language="en",
            topics=[topic],
        )
        concept = "vector space" if topic == "linear-algebra" else "memory management"
        path = [concept.title(), "Exercises"]
        evidence = [
            _item(number, "toc_exact", "Exercises", path),
            _item(number + 1, "toc_public_web_exact", concept.title()),
            _item(number + 2, "description", f"Covers {concept} and programming exercises."),
            _item(number + 3, "subject", topic),
            _item(number + 4, "metadata_minimal", book.title),
        ]
        number += 5
        records.append(
            ImportedBookEvidence(
                schema_version=1,
                contract_version="book-evidence-v1",
                book=book,
                book_content_hash=HASH,
                evidence=evidence,
            )
        )
    return records


def _frozen(records: list[ImportedBookEvidence]):
    return build_evidence_concept_gold_review(
        records,
        GRAPH,
        FEATURES,
        MATCHING,
        HASH,
        sample_size_by_evidence_type={"metadata_minimal": 2},
        per_book_cap=1,
    )


def test_fresh_membership_is_deterministic_excluded_and_prediction_blind() -> None:
    records = _records()
    frozen = _frozen(records)
    kwargs = {
        "graph": GRAPH,
        "frozen_review": frozen,
        "book_evidence_hash": HASH,
        "frozen_gold_hash": HASH,
        "general_sizes": {
            "toc_exact": 4,
            "toc_public_web_exact": 4,
            "description": 4,
            "subject": 4,
            "metadata_minimal": 4,
        },
        "per_book_cap": 5,
        "generic_toc_cap": 2,
    }
    first = build_holdout_manifests(records, **kwargs)
    second = build_holdout_manifests(list(reversed(records)), **kwargs)

    assert first == second
    general, challenge = first
    frozen_ids = {row.evidence_id for row in frozen.entries}
    general_ids = {row.evidence_id for row in general.entries}
    challenge_ids = {row.evidence_id for row in challenge.entries}
    assert not (frozen_ids & (general_ids | challenge_ids))
    assert not (general_ids & challenge_ids)
    assert {row.topic for row in general.entries} == {"linear-algebra", "operating-systems"}
    assert all(not hasattr(row, "predicted_concept_ids") for row in general.entries)


def test_predictions_are_separate_and_review_packet_is_blinded() -> None:
    records = _records()
    general, _ = build_holdout_manifests(
        records,
        GRAPH,
        _frozen(records),
        HASH,
        HASH,
        general_sizes={"toc_exact": 4, "description": 4, "subject": 4},
        per_book_cap=4,
    )
    predictions = build_holdout_predictions(general, HASH, FEATURES, GRAPH, MATCHING, EXPERIMENT)
    review = build_holdout_review(general, HASH, HASH, GRAPH)
    packet = review_packet(review, GRAPH, 10)

    assert len(predictions.entries) == len(general.entries)
    assert all(len(row.predictions) == 5 for row in predictions.entries)
    assert packet["prediction_visibility"] == "hidden_during_human_review"
    assert "predicted_concept_ids" not in str(packet)
    assert all(row.review_status == "unreviewed" for row in review.entries)


def test_challenge_categories_cover_path_and_overlap_without_predictions() -> None:
    records = _records()
    rows = [
        item
        for item in build_holdout_manifests(
            records,
            GRAPH,
            _frozen(records),
            HASH,
            HASH,
            general_sizes={"subject": 1},
            per_book_cap=8,
        )[1].entries
    ]

    categories = {category for row in rows for category in row.challenge_categories}
    assert "nested_overlap" in categories
    assert "path_context_candidate" in categories
    assert "parent_overinheritance_counterexample" in categories
    assert all(_challenge_categories(row) == row.challenge_categories for row in rows)


def test_multi_topic_evidence_uses_one_deterministic_artifact_identity() -> None:
    record = _records()[0]
    multi_topic = record.model_copy(
        update={
            "book": record.book.model_copy(
                update={"topics": ["operating-systems", "linear-algebra"]}
            )
        }
    )

    rows = _population([multi_topic], GRAPH, set())

    assert len(rows) == len(record.evidence)
    assert len({row.evidence_id for row in rows}) == len(rows)
    assert {row.topic for row in rows} == {"linear-algebra"}


def test_holdout_review_rejects_duplicate_evidence_ids(tmp_path: Path) -> None:
    records = _records()
    general, _ = build_holdout_manifests(
        records,
        GRAPH,
        _frozen(records),
        HASH,
        HASH,
        general_sizes={"subject": 2},
        per_book_cap=2,
    )
    review = build_holdout_review(general, HASH, HASH, GRAPH)
    duplicate = review.model_copy(update={"entries": [*review.entries, review.entries[0]]})
    path = tmp_path / "duplicate-review.json"
    path.write_text(duplicate.model_dump_json(), encoding="utf-8")

    with pytest.raises(ValueError, match="duplicate holdout review evidence ID"):
        load_holdout_review(path, general, HASH, HASH, GRAPH)


def _completed_review(manifest, predictions):
    review = build_holdout_review(manifest, HASH, HASH, GRAPH)
    v1_by_id = {
        row.evidence_id: row.predictions[0].predicted_concept_ids for row in predictions.entries
    }
    for row in review.entries:
        gold = v1_by_id[row.evidence_id]
        review = update_holdout_review_entry(
            review,
            GRAPH,
            row.evidence_id,
            "labeled" if gold else "no_concept",
            gold,
            "fixture review",
        )
    return review


def test_holdout_evaluation_requires_complete_review_and_has_general_scopes() -> None:
    general, _ = build_holdout_manifests(
        _records(),
        GRAPH,
        _frozen(_records()),
        HASH,
        HASH,
        general_sizes={"toc_exact": 4, "description": 4, "subject": 4},
        per_book_cap=4,
    )
    predictions = build_holdout_predictions(general, HASH, FEATURES, GRAPH, MATCHING, EXPERIMENT)
    incomplete = build_holdout_review(general, HASH, HASH, GRAPH)

    try:
        build_holdout_evaluation_report(general, predictions, incomplete, HASH, HASH, HASH)
    except ValueError as exc:
        assert "complete human review" in str(exc)
    else:
        raise AssertionError("incomplete holdout review must be rejected")

    report = build_holdout_evaluation_report(
        general,
        predictions,
        _completed_review(general, predictions),
        HASH,
        HASH,
        HASH,
    )
    assert len(report.variants) == 5
    scopes = {metric.scope for metric in report.variants[0].metrics}
    assert {"all", "family:toc", "family:metadata"} <= scopes
    assert any(scope.startswith("topic:") for scope in scopes)
    assert any(scope.startswith("evidence_type:") for scope in scopes)


def test_challenge_category_metrics_allow_overlapping_membership() -> None:
    records = _records()
    _, challenge = build_holdout_manifests(
        records,
        GRAPH,
        _frozen(records),
        HASH,
        HASH,
        general_sizes={"subject": 1},
        per_book_cap=8,
    )
    predictions = build_holdout_predictions(challenge, HASH, FEATURES, GRAPH, MATCHING, EXPERIMENT)
    report = build_holdout_evaluation_report(
        challenge,
        predictions,
        _completed_review(challenge, predictions),
        HASH,
        HASH,
        HASH,
    )

    assert report.category_membership_is_overlapping is True
    metrics = {metric.scope: metric for metric in report.variants[0].metrics}
    category_total = sum(
        metric.entry_count
        for scope, metric in metrics.items()
        if scope.startswith("challenge_category:")
    )
    assert category_total > metrics["all"].entry_count
