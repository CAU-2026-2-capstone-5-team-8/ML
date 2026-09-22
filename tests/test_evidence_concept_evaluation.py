"""Source-aware concept review sampling, validation, and metrics."""

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest
from pydantic import ValidationError

from bookmatch_ml.concept_v2.evidence_evaluation import (
    EvidenceConceptGoldReview,
    LoadedEvidenceConceptGoldReview,
    build_evidence_concept_gold_review,
    evaluate_evidence_concept_gold,
    load_evidence_concept_gold_review,
    review_json_bytes,
    update_review_entry,
)
from bookmatch_ml.concept_v2.graph import load_concept_graph
from bookmatch_ml.concept_v2.profile import load_concept_matching_config
from bookmatch_ml.config import ConfigError, load_feature_config
from bookmatch_ml.data.book_evidence import ImportedBookEvidence, ImportedEvidenceItem
from bookmatch_ml.io import write_json
from bookmatch_ml.schemas import Book

ROOT = Path(__file__).resolve().parents[1]
FEATURES = load_feature_config(ROOT / "configs/features.yaml")
GRAPH = load_concept_graph(ROOT / "configs/concept_graph.yaml", FEATURES)
MATCHING = load_concept_matching_config(ROOT / "configs/concept_matching.yaml")
HASH = "sha256:" + "a" * 64


def _item(number: int, evidence_type: str, text: str) -> ImportedEvidenceItem:
    base = {
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
        base.update(
            edition_relation="same_work" if evidence_type == "toc_same_work" else "exact",
            toc_entry_id=f"toc_{number}",
            level=1,
            order_index=number,
            toc_path=[text],
        )
        if evidence_type == "toc_same_work":
            base.update(
                source_evidence_tier="same_work_alternate_edition_toc",
                source_evidence={
                    "evidence_type": "toc",
                    "tier": "same_work_alternate_edition_toc",
                    "target_title": "Fixture",
                    "target_authors": ["Author"],
                    "discovery_method": "provider_work_relation",
                    "match_basis": ["provider_work_relation"],
                    "validation_status": "strong",
                },
            )
    elif evidence_type == "description":
        base.update(
            edition_relation="unspecified",
            document_id=f"doc_{number}",
            document_type="description",
            document_content_hash=HASH,
        )
    else:
        base.update(
            edition_relation="canonical_record",
            metadata_field="topics" if evidence_type == "subject" else "title",
        )
    return ImportedEvidenceItem.model_validate(base)


def _records() -> list[ImportedBookEvidence]:
    records = []
    number = 1
    for index, topic in enumerate(
        ["operating-systems", "operating-systems", "linear-algebra", "linear-algebra"],
        start=1,
    ):
        concept = "virtual memory" if topic == "operating-systems" else "determinant"
        book = Book(
            book_id=f"book_{index:020x}",
            title=f"{concept.title()} Textbook {index}",
            authors=["Author"],
            language="en",
            topics=[topic],
        )
        evidence = []
        for evidence_type in (
            "toc_exact",
            "toc_public_web_exact",
            "toc_same_work",
            "description",
            "subject",
            "metadata_minimal",
        ):
            text = concept
            if evidence_type == "subject":
                text = topic
            elif evidence_type == "description":
                text = f"This book covers {concept}."
            elif evidence_type == "metadata_minimal":
                text = book.title
            evidence.append(_item(number, evidence_type, text))
            number += 1
        evidence.append(_item(number, "toc_exact", "Introduction"))
        number += 1
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


def _review() -> EvidenceConceptGoldReview:
    return build_evidence_concept_gold_review(
        records=_records(),
        graph=GRAPH,
        features=FEATURES,
        matching=MATCHING,
        book_evidence_hash=HASH,
        sample_size_by_evidence_type={
            "toc_same_work": 2,
            "toc_exact": 4,
            "toc_public_web_exact": 2,
            "description": 2,
            "subject": 2,
            "metadata_minimal": 2,
        },
        per_book_cap=4,
        generic_toc_cap=1,
    )


def test_sampling_is_deterministic_stratified_and_capped() -> None:
    first = _review()
    reversed_records = build_evidence_concept_gold_review(
        records=list(reversed(_records())),
        graph=GRAPH,
        features=FEATURES,
        matching=MATCHING,
        book_evidence_hash=HASH,
        sample_size_by_evidence_type=first.requested_sample_size_by_evidence_type,
        per_book_cap=4,
        generic_toc_cap=1,
    )

    assert review_json_bytes(first) == review_json_bytes(reversed_records)
    assert {row.evidence_type for row in first.entries} == set(
        first.requested_sample_size_by_evidence_type
    )
    assert {row.topic for row in first.entries} == {"operating-systems", "linear-algebra"}
    assert (
        max(
            sum(candidate.book_id == row.book_id for candidate in first.entries)
            for row in first.entries
        )
        <= 4
    )
    assert sum(row.evidence_text == "Introduction" for row in first.entries) <= 1
    assert all(row.provider == "fixture" for row in first.entries)
    assert all(row.edition_relation for row in first.entries)


def test_unreviewed_rows_never_become_gold_or_metrics(tmp_path: Path) -> None:
    review = _review()
    path = tmp_path / "review.json"
    write_json(review, path)
    loaded = load_evidence_concept_gold_review(path, review, GRAPH)

    report = evaluate_evidence_concept_gold(loaded)

    assert report.metric_status == "unavailable"
    assert report.reviewed_entry_count == 0
    assert report.remaining_entry_count == len(review.entries)
    assert report.metrics == []
    assert report.policy_metrics == []


def test_human_update_rejects_invalid_concepts_and_outcome_shapes() -> None:
    review = _review()
    row = review.entries[0]
    with pytest.raises(ValueError, match="unknown"):
        update_review_entry(review, GRAPH, row.evidence_id, "labeled", ["not-a-node"], None)
    with pytest.raises(ValidationError, match="require human gold"):
        update_review_entry(review, GRAPH, row.evidence_id, "labeled", [], None)
    with pytest.raises(ValidationError, match="cannot contain human gold"):
        update_review_entry(
            review, GRAPH, row.evidence_id, "no_concept", [GRAPH.graph.nodes[row.topic][0]], None
        )


def test_loader_rejects_duplicate_rows_and_prediction_edits(tmp_path: Path) -> None:
    review = _review()
    payload = review.model_dump(mode="json")
    payload["entries"][1] = dict(payload["entries"][0])
    path = tmp_path / "duplicate.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ConfigError, match="duplicate evidence concept review row"):
        load_evidence_concept_gold_review(path, review, GRAPH)

    payload = review.model_dump(mode="json")
    payload["entries"][0]["evidence_text"] = "tampered"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ConfigError, match="identity or prediction differs"):
        load_evidence_concept_gold_review(path, review, GRAPH)


def test_partial_metrics_and_policy_presence_deduplicate_row_occurrences(tmp_path: Path) -> None:
    review = _review()
    topic = "operating-systems"
    chosen = [
        row
        for row in review.entries
        if row.topic == topic and row.predicted_concept_ids == ["virtual memory"]
    ][:3]
    assert len(chosen) == 3
    chosen_ids = {row.evidence_id for row in chosen}
    reviewed_entries = [
        row.model_copy(
            update={
                "human_gold_concept_ids": ["virtual memory"],
                "review_status": "reviewed",
                "review_outcome": "labeled",
            }
        )
        if row.evidence_id in chosen_ids
        else row
        for row in review.entries
    ]
    reviewed = review.model_copy(update={"entries": reviewed_entries})
    path = tmp_path / "review.json"
    write_json(reviewed, path)
    loaded = LoadedEvidenceConceptGoldReview(review=reviewed, content_hash=HASH)

    report = evaluate_evidence_concept_gold(loaded)

    assert report.metric_status == "partial"
    assert report.evaluable_entry_count == 3
    assert report.metrics[0].micro_f1 == 1
    combined = next(item for item in report.policy_metrics if item.policy == "combined_unweighted")
    assert combined.unique_book_concept_assignment_count == 1
    assert combined.raw_prediction_occurrence_count == 3
    assert combined.micro_f1 == 1


def test_not_judgable_is_reviewed_but_excluded_from_metrics() -> None:
    review = _review()
    row = review.entries[0]
    updated = update_review_entry(
        review,
        GRAPH,
        row.evidence_id,
        "not_judgable",
        [],
        "Evidence is too generic.",
    )
    report = evaluate_evidence_concept_gold(
        LoadedEvidenceConceptGoldReview(review=updated, content_hash=HASH)
    )

    assert report.reviewed_entry_count == 1
    assert report.not_judgable_entry_count == 1
    assert report.evaluable_entry_count == 0
    assert report.metric_status == "unavailable"
