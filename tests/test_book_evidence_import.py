import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

import pytest
from typer.testing import CliRunner

from bookmatch_ml.cli import app
from bookmatch_ml.data.book_evidence import (
    BookEvidenceImportError,
    ImportedBookEvidence,
    ImportedEvidenceItem,
    build_concept_candidate_inputs,
    load_book_evidence,
    summarize_book_evidence,
)
from bookmatch_ml.schemas import Book


def _sha256_json(value: object) -> str:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return f"sha256:{hashlib.sha256(encoded.encode()).hexdigest()}"


def _book(book_id: str, title: str) -> Book:
    return Book(
        book_id=book_id,
        title=title,
        authors=["Fixture Author"],
        language="en",
        topics=["operating-systems"],
    )


def _item(evidence_id: str, **values: object) -> ImportedEvidenceItem:
    draft = ImportedEvidenceItem.model_validate(
        {
            "evidence_id": evidence_id,
            "provider": "fixture",
            "source_type": "metadata_api",
            "source_url": "https://example.test/book",
            "source_retrieved_at": datetime(2026, 9, 22, tzinfo=UTC),
            "source_content_hash": "sha256:" + "a" * 64,
            "source_evidence_tier": None,
            "source_evidence": None,
            "provenance_hash": "sha256:" + "0" * 64,
            **values,
        }
    )
    payload = draft.model_dump(mode="json")
    payload.pop("evidence_id")
    payload.pop("provenance_hash")
    return draft.model_copy(update={"provenance_hash": _sha256_json(payload)})


def _record(book: Book, evidence: list[ImportedEvidenceItem]) -> ImportedBookEvidence:
    return ImportedBookEvidence(
        schema_version=1,
        contract_version="book-evidence-v1",
        book=book,
        book_content_hash=_sha256_json(book.model_dump(mode="json")),
        evidence=evidence,
    )


def _records() -> list[ImportedBookEvidence]:
    toc_book = _book("book_11111111111111111111", "TOC Book")
    metadata_book = _book("book_22222222222222222222", "Metadata Book")
    title = _item(
        "evidence_11111111111111111111",
        evidence_type="metadata_minimal",
        text=toc_book.title,
        source_id="source_toc_metadata",
        edition_relation="canonical_record",
        metadata_field="title",
    )
    toc = _item(
        "evidence_22222222222222222222",
        evidence_type="toc_public_web_exact",
        text="Virtual Memory",
        source_id="source_toc_public",
        source_type="publisher_page",
        source_url="https://example.test/toc",
        source_content_hash="sha256:" + "b" * 64,
        source_evidence_tier="validated_public_web_toc",
        source_evidence={
            "evidence_type": "toc",
            "tier": "validated_public_web_toc",
            "target_title": toc_book.title,
            "target_authors": toc_book.authors,
            "source_isbns": [],
            "source_author_ids": [],
            "same_edition": True,
            "source_document_type": "public_html",
            "discovery_method": "reviewed_public_page",
            "match_basis": ["exact_title", "author"],
            "validation_status": "strong",
        },
        edition_relation="exact",
        toc_entry_id="toc_virtual_memory",
        level=2,
        order_index=0,
        label="9.2",
        toc_path=["Memory Management", "Virtual Memory"],
    )
    metadata_title = _item(
        "evidence_33333333333333333333",
        evidence_type="metadata_minimal",
        text=metadata_book.title,
        source_id="source_metadata",
        edition_relation="canonical_record",
        metadata_field="title",
    )
    subject = _item(
        "evidence_44444444444444444444",
        evidence_type="subject",
        text="operating-systems",
        source_id="source_metadata",
        edition_relation="canonical_record",
        metadata_field="topics",
    )
    description = _item(
        "evidence_55555555555555555555",
        evidence_type="description",
        text="Covers processes, scheduling, and memory management.",
        source_id="source_metadata",
        edition_relation="unspecified",
        document_id="doc_metadata_description",
        document_type="description",
        document_content_hash="sha256:" + "c" * 64,
    )
    return [
        _record(toc_book, [title, toc]),
        _record(metadata_book, [metadata_title, subject, description]),
    ]


def _write_artifact(path: Path, records: list[ImportedBookEvidence] | None = None) -> Path:
    records = records or _records()
    path.write_text(
        "".join(record.model_dump_json(exclude_none=False) + "\n" for record in records),
        encoding="utf-8",
    )
    return path


def test_loads_source_aware_contract_without_weights(tmp_path: Path) -> None:
    records = load_book_evidence(_write_artifact(tmp_path / "evidence.jsonl"))
    toc = records[0].evidence[1]

    assert toc.evidence_type == "toc_public_web_exact"
    assert toc.edition_relation == "exact"
    assert toc.toc_path == ["Memory Management", "Virtual Memory"]
    assert toc.source_evidence is not None
    assert toc.source_evidence.match_basis == ["exact_title", "author"]
    assert "confidence" not in toc.model_dump()
    assert "weight" not in toc.model_dump()


def test_metadata_only_book_still_produces_concept_candidate_inputs(tmp_path: Path) -> None:
    records = load_book_evidence(_write_artifact(tmp_path / "evidence.jsonl"))
    candidate_books = build_concept_candidate_inputs(records)
    metadata = next(item for item in candidate_books if item.book.title == "Metadata Book")

    assert {candidate.evidence_type for candidate in metadata.candidates} == {
        "description",
        "metadata_minimal",
        "subject",
    }
    assert any("scheduling" in candidate.text for candidate in metadata.candidates)
    assert all("weight" not in candidate.model_dump() for candidate in metadata.candidates)


def test_summary_keeps_toc_and_metadata_coverage_separate(tmp_path: Path) -> None:
    records = load_book_evidence(_write_artifact(tmp_path / "evidence.jsonl"))

    summary = summarize_book_evidence(records)

    assert summary["total_books"] == 2
    assert summary["books_with_toc_evidence"] == 1
    assert summary["books_with_exact_toc"] == 1
    assert summary["books_with_public_web_toc"] == 1
    assert summary["books_with_same_work_alternate_toc"] == 0
    assert summary["books_with_unspecified_toc"] == 0
    assert summary["metadata_fallback_only"] == 1
    assert summary["books_with_zero_evidence"] == 0
    assert summary["evidence_type_rows"]["description"] == 1
    assert summary["evidence_type_rows"]["toc_public_web_exact"] == 1


def test_legacy_toc_without_provenance_stays_unspecified() -> None:
    toc = _records()[0].evidence[1]
    payload = toc.model_dump()
    payload.update(
        evidence_type="toc_unspecified",
        source_evidence_tier=None,
        source_evidence=None,
        edition_relation="unspecified",
    )

    unspecified = ImportedEvidenceItem.model_validate(payload)

    assert unspecified.evidence_type == "toc_unspecified"
    assert unspecified.edition_relation == "unspecified"


def test_rejects_changed_provenance_hash(tmp_path: Path) -> None:
    path = _write_artifact(tmp_path / "evidence.jsonl")
    lines = path.read_text(encoding="utf-8").splitlines()
    payload = json.loads(lines[0])
    payload["evidence"][0]["text"] = "Changed after export"
    lines[0] = json.dumps(payload)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    with pytest.raises(BookEvidenceImportError, match="provenance hash mismatch"):
        load_book_evidence(path)


def test_rejects_provenance_type_that_conflicts_with_category(tmp_path: Path) -> None:
    path = _write_artifact(tmp_path / "evidence.jsonl")
    lines = path.read_text(encoding="utf-8").splitlines()
    payload = json.loads(lines[0])
    payload["evidence"][1]["source_evidence"]["evidence_type"] = "metadata"
    lines[0] = json.dumps(payload)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    with pytest.raises(BookEvidenceImportError, match="type does not match evidence category"):
        load_book_evidence(path)


@pytest.mark.parametrize("foreign_field", ["toc_path", "document_content_hash"])
def test_rejects_fields_from_another_evidence_category(
    tmp_path: Path,
    foreign_field: str,
) -> None:
    path = _write_artifact(tmp_path / "evidence.jsonl")
    lines = path.read_text(encoding="utf-8").splitlines()
    payload = json.loads(lines[1])
    payload["evidence"][1][foreign_field] = (
        ["Not a subject path"] if foreign_field == "toc_path" else "sha256:" + "f" * 64
    )
    lines[1] = json.dumps(payload)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    with pytest.raises(BookEvidenceImportError, match="metadata evidence cannot carry"):
        load_book_evidence(path)


def test_rejects_duplicate_evidence_identity(tmp_path: Path) -> None:
    records = _records()
    duplicate = records[0].evidence[0]
    records[1] = records[1].model_copy(update={"evidence": [duplicate, *records[1].evidence]})

    with pytest.raises(BookEvidenceImportError, match="duplicate evidence ID"):
        load_book_evidence(_write_artifact(tmp_path / "evidence.jsonl", records))


def test_rejects_inconsistent_source_snapshot(tmp_path: Path) -> None:
    records = _records()
    changed = records[1].evidence[1].model_copy(update={"provider": "changed-provider"})
    payload = changed.model_dump(mode="json")
    payload.pop("evidence_id")
    payload.pop("provenance_hash")
    changed = changed.model_copy(update={"provenance_hash": _sha256_json(payload)})
    records[1] = records[1].model_copy(
        update={"evidence": [records[1].evidence[0], changed, records[1].evidence[2]]}
    )

    with pytest.raises(BookEvidenceImportError, match="inconsistent source snapshot"):
        load_book_evidence(_write_artifact(tmp_path / "evidence.jsonl", records))


def test_cli_validates_and_reports_artifact(tmp_path: Path) -> None:
    path = _write_artifact(tmp_path / "evidence.jsonl")

    result = CliRunner().invoke(app, ["inspect-book-evidence", "--input", str(path)])

    assert result.exit_code == 0, result.output
    assert '"total_books": 2' in result.output
    assert '"books_with_toc_evidence": 1' in result.output
    assert '"metadata_fallback_only": 1' in result.output


def test_rejects_empty_artifact(tmp_path: Path) -> None:
    path = tmp_path / "empty.jsonl"
    path.write_text("\n", encoding="utf-8")

    with pytest.raises(BookEvidenceImportError, match="contains no records"):
        load_book_evidence(path)
