"""Scope is collection metadata, not a provider heuristic or a readability score."""

import hashlib
import json
from pathlib import Path

import pytest

from bookmatch_ml.book.difficulty import build_difficulty_profile
from bookmatch_ml.concept_v2.book_evidence_mapping import build_book_evidence_concept_mapping_report
from bookmatch_ml.concept_v2.evidence_evaluation import build_evidence_concept_gold_review
from bookmatch_ml.concept_v2.graph import load_concept_graph
from bookmatch_ml.concept_v2.holdout import build_holdout_manifests
from bookmatch_ml.concept_v2.profile import load_concept_matching_config
from bookmatch_ml.config import load_feature_config
from bookmatch_ml.data.book_evidence import (
    BookEvidenceImportError,
    build_concept_candidate_inputs,
    load_book_evidence,
)
from bookmatch_ml.data.evidence import assemble_book_evidence, calculate_evidence_coverage
from bookmatch_ml.data.loader import load_canonical_dataset
from bookmatch_ml.schemas import TextExtent

ROOT = Path(__file__).parents[1]
FIXTURE = Path(__file__).parent / "fixtures/book-evidence-v2.jsonl"
CONFIG = load_feature_config(ROOT / "configs/features.yaml")
GRAPH = load_concept_graph(ROOT / "configs/concept_graph.yaml", CONFIG)
MATCHING = load_concept_matching_config(ROOT / "configs/concept_matching_v2.yaml")


def test_mapping_retains_v2_scope_and_rights_and_legacy_review_rejects_v2():
    rows = load_book_evidence(FIXTURE)
    digest = "sha256:" + hashlib.sha256(FIXTURE.read_bytes()).hexdigest()
    result = build_book_evidence_concept_mapping_report(rows, digest, CONFIG, GRAPH, MATCHING)
    assert result.book_evidence_contract_version == "book-evidence-v2"
    supports = [s for c in result.books[0].concepts for s in c.supporting_evidence]
    excerpt = next(s for s in supports if s.document_id == "doc_excerpt")
    assert excerpt.text_extent.scope == "excerpt"
    assert excerpt.source_rights_note == rows[0].evidence[0].source_rights_note
    assert excerpt.source_url == "https://example.test/book"
    with pytest.raises(ValueError, match="requires book-evidence-v1"):
        build_evidence_concept_gold_review(rows, GRAPH, CONFIG, MATCHING, digest)
    blank_review = build_evidence_concept_gold_review([], GRAPH, CONFIG, MATCHING, digest)
    with pytest.raises(ValueError, match="requires book-evidence-v1"):
        build_holdout_manifests(rows, GRAPH, blank_review, digest, digest)


def test_upstream_shared_fixture_loads_and_candidate_projection_preserves_scope_and_rights():
    rows = load_book_evidence(FIXTURE)
    candidates = build_concept_candidate_inputs(rows)[0].candidates
    docs = {c.document_id: c for c in candidates if c.document_id}
    assert rows[0].contract_version == "book-evidence-v2"
    assert docs["doc_excerpt"].text_extent.scope == "excerpt"
    assert docs["doc_complete"].text_extent.scope == "complete_section"
    assert docs["doc_unknown"].text_extent is None
    assert (
        docs["doc_excerpt"].source_rights_note
        == "Synthetic test source. No reuse permission inferred."
    )
    assert docs["doc_excerpt"].source_license is None
    assert docs["doc_excerpt"].source_url == "https://example.test/book"


@pytest.mark.parametrize(
    "field,value",
    [
        ("source_rights_note", "changed"),
        ("source_license", "new license"),
        ("text_extent", {"scope": "complete_section", "basis": "invented completeness"}),
    ],
)
def test_extent_and_rights_tampering_invalidates_provenance(tmp_path, field, value):
    row = json.loads(FIXTURE.read_text())
    item = next(i for i in row["evidence"] if i["document_id"] == "doc_excerpt")
    item[field] = value
    path = tmp_path / "changed.jsonl"
    path.write_text(json.dumps(row) + "\n")
    with pytest.raises(BookEvidenceImportError, match="provenance hash mismatch"):
        load_book_evidence(path)


def test_same_source_cannot_claim_conflicting_rights_even_with_recomputed_hash(tmp_path):
    row = json.loads(FIXTURE.read_text())
    item = row["evidence"][0]
    item["source_rights_note"] = "conflicting statement"
    payload = {k: v for k, v in item.items() if k not in {"provenance_hash", "evidence_id"}}
    item["provenance_hash"] = (
        "sha256:"
        + hashlib.sha256(
            json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
    )
    path = tmp_path / "inconsistent.jsonl"
    path.write_text(json.dumps(row) + "\n")
    with pytest.raises(BookEvidenceImportError, match="inconsistent source snapshot"):
        load_book_evidence(path)


def test_v2_requires_extent_and_rights_fields_and_rejects_v1_disguise(tmp_path):
    row = json.loads(FIXTURE.read_text())
    del row["evidence"][0]["source_rights_note"]
    path = tmp_path / "missing.jsonl"
    path.write_text(json.dumps(row) + "\n")
    with pytest.raises(BookEvidenceImportError, match="source_rights_note"):
        load_book_evidence(path)
    row = json.loads(FIXTURE.read_text())
    row.update(schema_version=1, contract_version="book-evidence-v1")
    path.write_text(json.dumps(row) + "\n")
    with pytest.raises(BookEvidenceImportError, match="Extra inputs"):
        load_book_evidence(path)


def evidence_with_scope(scope):
    evidence = assemble_book_evidence(load_canonical_dataset(ROOT / "tests/fixtures/canonical"))[0]
    extent = TextExtent(scope=scope, basis="Synthetic fixture section boundary.") if scope else None
    documents = [
        d.model_copy(update={"text_extent": extent}) if d.document_type == "preface" else d
        for d in evidence.documents
    ]
    return evidence.model_copy(
        update={
            "documents": documents,
            "coverage": calculate_evidence_coverage(documents, evidence.toc),
        }
    )


@pytest.mark.parametrize(
    "scope,expected",
    [
        (None, "mixed_or_unknown"),
        ("excerpt", "excerpt_only"),
        ("complete_section", "complete_sections_only"),
    ],
)
def test_difficulty_keeps_observed_scope_without_changing_formula(scope, expected):
    evidence = evidence_with_scope(scope)
    profile = build_difficulty_profile(evidence, CONFIG)
    assert profile.analyzed_text_scope == expected
    assert profile.text_scope_version == "text-extent-v1"
    if scope:
        assert profile.documents[0].text_extent.scope == scope
    else:
        assert profile.documents[0].text_extent is None
    legacy = build_difficulty_profile(evidence_with_scope(None), CONFIG)
    assert profile.lexical_difficulty == legacy.lexical_difficulty
    c = evidence.coverage
    assert c.prose_document_count == (
        c.excerpt_prose_document_count
        + c.complete_section_prose_document_count
        + c.unknown_extent_prose_document_count
    )


def test_short_excerpt_retains_scope_when_excluded_and_scores_stay_unavailable():
    evidence = evidence_with_scope("excerpt")
    docs = [d.model_copy(update={"text": "Too short."}) for d in evidence.documents]
    evidence = evidence.model_copy(
        update={"documents": docs, "coverage": calculate_evidence_coverage(docs, evidence.toc)}
    )
    result = build_difficulty_profile(evidence, CONFIG)
    assert result.analyzed_text_scope == "unavailable"
    assert result.lexical_difficulty is None
    assert result.excluded_documents[0].text_extent.scope == "excerpt"


def test_scope_is_unknown_for_legacy_canonical_and_invalid_basis_is_rejected():
    evidence = evidence_with_scope(None)
    assert evidence.coverage.unknown_extent_prose_document_count == 1
    legacy = evidence.coverage.model_dump()
    for k in [
        "text_scope_version",
        "excerpt_prose_document_count",
        "complete_section_prose_document_count",
        "unknown_extent_prose_document_count",
    ]:
        legacy.pop(k)
    assert type(evidence.coverage).model_validate(legacy).unknown_extent_prose_document_count == 1
    with pytest.raises(ValueError):
        TextExtent(scope="excerpt", basis=" ")
