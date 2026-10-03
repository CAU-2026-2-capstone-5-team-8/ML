"""Deterministic comprehension grounding contract tests."""

import hashlib
from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from bookmatch_ml.assessment.grounding import (
    DISPLAY_NORMALIZATION_POLICY,
    MAX_PASSAGE_CHARACTERS,
    GenerationGrounding,
    GenerationGroundingV2,
    GroundingError,
    build_generation_grounding,
    build_generation_grounding_v2,
    extract_passage,
    normalize_display_passage,
)
from bookmatch_ml.assessment.schemas import AssessmentEvidenceRef, QuestionSpec
from bookmatch_ml.cli import _validate_grounding_output_path
from bookmatch_ml.schemas import Book, CanonicalDataset, Document, Source

HASH = "sha256:" + "1" * 64
CANONICAL_HASHES = {
    "books.jsonl": HASH,
    "documents.jsonl": HASH,
    "toc.jsonl": HASH,
    "sources.jsonl": HASH,
}
HEFFERON_DOCUMENT_ID = "doc_de934d33d551223812e8"
HEFFERON_SOURCE_PASSAGE = """2.6 DeﬁnitionAnm×n matrix is a rectangular array of numbers withm rows
andn columns. Each number in the matrix is anentry.
We usually denote a matrix with an upper case roman letter. For instance,
A =
(
1 2.2 5
3 4 −7
)
has2 rows and3 columns and so is a2×3 matrix. Read that aloud as “two-by-
three”; the number of rows is always stated ﬁrst. (The matrix has parentheses
around it so that when two matrices are adjacent we can tell where one ends and
the other begins.) We name matrix entries with the corresponding lower-case
letter, so that the entry in the second row and ﬁrst column of the above array
isa2,1 =3."""
HEFFERON_DISPLAY_PASSAGE = (
    "2.6 Definition An m×n matrix is a rectangular array of numbers with m rows\n"
    "and n columns. Each number in the matrix is an entry.\n"
    "We usually denote a matrix with an upper case roman letter. For instance,\n"
    "A =\n"
    "(\n"
    "1 2.2 5\n"
    "3 4 −7\n"
    ")\n"
    "has 2 rows and 3 columns and so is a 2×3 matrix. Read that aloud as “"
    "two-by-three”; the number of rows is always stated first. "
    "(The matrix has parentheses\n"
    "around it so that when two matrices are adjacent we can tell where one ends and\n"
    "the other begins.) We name matrix entries with the corresponding lower-case\n"
    "letter, so that the entry in the second row and first column of the above array\n"
    "is a2,1 = 3."
)


def _hash_text(value: str) -> str:
    return "sha256:" + hashlib.sha256(value.encode("utf-8")).hexdigest()


def _text(concept: str = "matrix", *, repeat: int = 28) -> str:
    return (
        f"A {concept} is introduced here as the object used in the worked situation. "
        + "This collected textbook sentence explains how the object is interpreted and applied. "
        * repeat
    ).strip()


def _spec() -> QuestionSpec:
    return QuestionSpec(
        question_id="q_" + "a" * 20,
        topic_id="linear-algebra",
        question_type="comprehension",
        cognitive_operation="apply",
        concept_role="covered",
        primary_concept="matrix",
        related_concepts=[],
        prerequisite_concepts=[],
        target_difficulty=2,
        difficulty_version="question-difficulty-v1",
        difficulty_rationale="Apply the concept in one prose-grounded situation.",
        prerequisite_depth=None,
        primary_concept_count=1,
        related_concept_count=0,
        relationship_reasoning_required=False,
        multi_step_required=False,
        abstraction_level="concrete",
        source_text_complexity=0.4,
        assessment_priority=0.8,
        relation_candidate_source=None,
        evidence_summary="Grounded in one analyzed sample chapter.",
        supporting_book_ids=["book_" + "b" * 20],
        supporting_evidence=[
            AssessmentEvidenceRef(
                book_id="book_" + "b" * 20,
                concept_id="matrix",
                evidence_id="doc_grounding",
                evidence_type="sample_chapter",
                mention_count=2,
            )
        ],
        source_document_ids=["doc_grounding"],
        question_spec_version="question-spec-v1",
        config_version="assessment-config-v2-reviewed",
        config_hash=HASH,
    )


def _dataset(text: str | None = None) -> CanonicalDataset:
    document_text = text if text is not None else _text()
    book_id = "book_" + "b" * 20
    return CanonicalDataset(
        books=[
            Book(
                book_id=book_id,
                title="Open Linear Algebra",
                authors=["Example Author"],
                language="en",
                topics=["mathematics", "linear-algebra"],
            )
        ],
        documents=[
            Document(
                document_id="doc_grounding",
                book_id=book_id,
                document_type="sample_chapter",
                text=document_text,
                source_id="source_grounding",
                content_hash=_hash_text(document_text),
            )
        ],
        toc=[],
        sources=[
            Source(
                source_id="source_grounding",
                book_id=book_id,
                provider="open_textbook",
                source_type="open_textbook",
                url="https://example.test/open-textbook",
                retrieved_at=datetime(2026, 9, 1, tzinfo=UTC),
                license="Creative Commons Attribution 4.0 International License",
                rights_note="Explicitly licensed by the author.",
                content_hash=HASH,
            )
        ],
    )


def _build(spec: QuestionSpec | None = None, dataset: CanonicalDataset | None = None):
    return build_generation_grounding(
        spec or _spec(),
        dataset or _dataset(),
        canonical_file_hashes=CANONICAL_HASHES,
        blueprint_hash=HASH,
    )


def _hefferon_spec() -> QuestionSpec:
    spec = _spec()
    evidence = spec.supporting_evidence[0].model_copy(update={"evidence_id": HEFFERON_DOCUMENT_ID})
    return spec.model_copy(
        update={
            "supporting_evidence": [evidence],
            "source_document_ids": [HEFFERON_DOCUMENT_ID],
        }
    )


def _hefferon_dataset() -> CanonicalDataset:
    dataset = _dataset(HEFFERON_SOURCE_PASSAGE)
    document = dataset.documents[0].model_copy(update={"document_id": HEFFERON_DOCUMENT_ID})
    return dataset.model_copy(update={"documents": [document]})


def _build_v2() -> GenerationGroundingV2:
    return build_generation_grounding_v2(
        _hefferon_spec(),
        _hefferon_dataset(),
        canonical_file_hashes=CANONICAL_HASHES,
        blueprint_hash=HASH,
    )


def test_grounding_is_deterministic_bounded_and_preserves_source_identity() -> None:
    first = _build()
    second = _build()

    assert first == second
    assert first.model_dump_json() == second.model_dump_json()
    assert 600 <= len(first.passage_text) <= MAX_PASSAGE_CHARACTERS
    assert first.passage_text in _dataset().documents[0].text
    assert first.passage_hash == _hash_text(first.passage_text)
    assert first.source_document_id == "doc_grounding"
    assert first.book_id == "book_" + "b" * 20
    assert first.source_url == "https://example.test/open-textbook"
    assert first.license == "Creative Commons Attribution 4.0 International License"
    assert first.edition_relation == "unspecified"


def test_grounding_retains_original_passage_when_english_is_available() -> None:
    dataset = _dataset()
    document = dataset.documents[0]
    document.en_text = _text("translated concept")
    grounding = _build(dataset=dataset)
    assert grounding.passage_text in document.text
    assert grounding.passage_text not in document.en_text
    assert grounding.document_content_hash == _hash_text(document.text)
    document.content_hash = _hash_text(document.en_text)
    with pytest.raises(GroundingError, match="content hash"):
        _build(dataset=dataset)


def test_grounding_v2_preserves_exact_source_and_builds_reviewed_display() -> None:
    first = _build_v2()
    second = _build_v2()

    assert first == second
    assert first.model_dump_json() == second.model_dump_json()
    assert first.source_passage_text == HEFFERON_SOURCE_PASSAGE
    assert first.source_passage_text in _hefferon_dataset().documents[0].text
    assert first.source_passage_hash == _hash_text(HEFFERON_SOURCE_PASSAGE)
    assert first.display_passage_text == HEFFERON_DISPLAY_PASSAGE
    assert first.display_passage_hash == _hash_text(HEFFERON_DISPLAY_PASSAGE)
    assert first.display_normalization_policy == DISPLAY_NORMALIZATION_POLICY
    assert "1 2.2 5\n3 4 −7" in first.display_passage_text
    assert "a2,1" in first.display_passage_text


def test_display_normalization_fails_closed_for_unknown_policy_hash_or_passage() -> None:
    source_hash = _hash_text(HEFFERON_SOURCE_PASSAGE)
    with pytest.raises(GroundingError, match="unsupported display normalization policy"):
        normalize_display_passage(
            HEFFERON_SOURCE_PASSAGE,
            source_document_id=HEFFERON_DOCUMENT_ID,
            source_passage_hash=source_hash,
            policy="pdf-display-normalization-v2",
        )
    with pytest.raises(GroundingError, match="source passage hash"):
        normalize_display_passage(
            HEFFERON_SOURCE_PASSAGE + " changed",
            source_document_id=HEFFERON_DOCUMENT_ID,
            source_passage_hash=source_hash,
        )
    with pytest.raises(GroundingError, match="no reviewed rules"):
        normalize_display_passage(
            HEFFERON_SOURCE_PASSAGE,
            source_document_id="doc_unreviewed",
            source_passage_hash=source_hash,
        )


def test_grounding_v2_contract_rejects_altered_display_or_source_hash() -> None:
    grounding = _build_v2()
    display_payload = grounding.model_dump()
    display_payload["display_passage_text"] += " altered"
    with pytest.raises(ValidationError, match="display passage hash"):
        GenerationGroundingV2.model_validate(display_payload)

    source_payload = grounding.model_dump()
    source_payload["source_passage_hash"] = HASH
    with pytest.raises(ValidationError, match="source passage hash"):
        GenerationGroundingV2.model_validate(source_payload)


def test_grounding_rejects_missing_mismatched_or_unlicensed_source() -> None:
    missing = _dataset().model_copy(update={"documents": []})
    with pytest.raises(GroundingError, match="missing or duplicated"):
        _build(dataset=missing)

    dataset = _dataset()
    bad_document = dataset.documents[0].model_copy(update={"content_hash": HASH})
    with pytest.raises(GroundingError, match="content hash"):
        _build(dataset=dataset.model_copy(update={"documents": [bad_document]}))

    wrong_type = dataset.documents[0].model_copy(update={"document_type": "description"})
    with pytest.raises(GroundingError, match="allowed prose"):
        _build(dataset=dataset.model_copy(update={"documents": [wrong_type]}))

    unlicensed = dataset.sources[0].model_copy(update={"license": None})
    with pytest.raises(GroundingError, match="explicit reuse license"):
        _build(dataset=dataset.model_copy(update={"sources": [unlicensed]}))

    ambiguous = dataset.sources[0].model_copy(update={"license": "Publicly available"})
    with pytest.raises(GroundingError, match="explicit reuse license"):
        _build(dataset=dataset.model_copy(update={"sources": [ambiguous]}))

    denied = dataset.sources[0].model_copy(update={"license": "All rights reserved; no reuse"})
    with pytest.raises(GroundingError, match="explicit reuse license"):
        _build(dataset=dataset.model_copy(update={"sources": [denied]}))


def test_passage_extraction_fails_closed_on_irrelevant_or_unsafe_lengths() -> None:
    with pytest.raises(GroundingError, match="no concept-relevant"):
        extract_passage(_text("vector"), "matrix")
    with pytest.raises(GroundingError, match="too short"):
        extract_passage("A matrix is a rectangular array. Short context ends here.", "matrix")
    with pytest.raises(GroundingError, match="sentence exceeds"):
        extract_passage("A matrix " + "x" * MAX_PASSAGE_CHARACTERS + ".", "matrix")


def test_grounding_rejects_wrong_spec_operation_and_difficulty() -> None:
    non_comprehension = _spec().model_copy(update={"question_type": "vocabulary"})
    with pytest.raises(GroundingError, match="comprehension/apply/Level 2"):
        _build(spec=non_comprehension)

    integrate = _spec().model_copy(
        update={"cognitive_operation": "integrate", "target_difficulty": 3}
    )
    with pytest.raises(GroundingError, match="comprehension/apply/Level 2"):
        _build(spec=integrate)

    wrong_document = _spec().model_copy(update={"source_document_ids": ["doc_other"]})
    with pytest.raises(GroundingError, match="missing or duplicated"):
        _build(spec=wrong_document)


def test_grounding_contract_rejects_altered_passage_or_malformed_dataset_hashes() -> None:
    grounding = _build()
    payload = grounding.model_dump()
    payload["passage_text"] += " altered"
    with pytest.raises(ValidationError, match="passage hash"):
        GenerationGrounding.model_validate(payload)

    with pytest.raises(GroundingError, match="all four JSONL"):
        build_generation_grounding(
            _spec(),
            _dataset(),
            canonical_file_hashes={"books.jsonl": HASH},
            blueprint_hash=HASH,
        )


def test_grounding_output_cannot_overwrite_an_input(tmp_path) -> None:
    blueprint = tmp_path / "blueprint.json"
    canonical = tmp_path / "documents.jsonl"
    output = tmp_path / "grounding.json"

    with pytest.raises(GroundingError, match="differ from every input"):
        _validate_grounding_output_path(blueprint, [blueprint, canonical])
    assert _validate_grounding_output_path(output, [blueprint, canonical]) == output.resolve()
