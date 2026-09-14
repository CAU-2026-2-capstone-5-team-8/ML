from pathlib import Path

from bookmatch_ml.data.evidence import assemble_book_evidence
from bookmatch_ml.data.loader import load_canonical_dataset

FIXTURE_DIR = Path(__file__).parent / "fixtures" / "canonical"


def test_documents_attach_to_correct_book_and_keep_types() -> None:
    evidence = assemble_book_evidence(load_canonical_dataset(FIXTURE_DIR))
    operating_systems = evidence[0]

    assert operating_systems.book_id == "book_11111111111111111111"
    assert [document.document_type for document in operating_systems.documents] == [
        "description",
        "preface",
    ]
    assert evidence[1].documents == []


def test_toc_hierarchy_and_canonical_order_are_preserved() -> None:
    evidence = assemble_book_evidence(load_canonical_dataset(FIXTURE_DIR))[0]

    assert [entry.toc_entry_id for entry in evidence.toc] == ["toc_root", "toc_child"]
    assert evidence.toc[1].parent_entry_id == evidence.toc[0].toc_entry_id
    assert evidence.toc[1].level == 2


def test_coverage_distinguishes_description_from_prose() -> None:
    evidence = assemble_book_evidence(load_canonical_dataset(FIXTURE_DIR))
    coverage = evidence[0].coverage

    assert coverage.has_metadata is True
    assert coverage.has_toc is True
    assert coverage.has_description is True
    assert coverage.has_preface is True
    assert coverage.toc_entry_count == 2
    assert coverage.document_count == 2
    assert coverage.prose_document_count == 1
    assert coverage.prose_character_count == len(evidence[0].documents[1].text)


def test_missing_optional_evidence_is_explicit_not_an_error() -> None:
    evidence = assemble_book_evidence(load_canonical_dataset(FIXTURE_DIR))[1]

    assert evidence.coverage.has_metadata is True
    assert evidence.coverage.has_toc is False
    assert evidence.coverage.has_description is False
    assert evidence.coverage.prose_document_count == 0
    assert evidence.coverage.prose_character_count == 0


def test_provider_fields_do_not_leak_into_book_evidence() -> None:
    evidence = assemble_book_evidence(load_canonical_dataset(FIXTURE_DIR))[0]
    serialized = evidence.model_dump_json()

    assert "fixture_provider" not in serialized
    assert '"provider"' not in serialized
    assert '"url"' not in serialized


def test_assembly_is_deterministic() -> None:
    dataset = load_canonical_dataset(FIXTURE_DIR)

    first = [item.model_dump_json() for item in assemble_book_evidence(dataset)]
    dataset.books.reverse()
    dataset.documents.reverse()
    dataset.toc.reverse()
    second = [item.model_dump_json() for item in assemble_book_evidence(dataset)]

    assert first == second
