"""Strict loader for the Data-Pipeline book-evidence-v1/v2/v3 handoff."""

import hashlib
import json
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

from pydantic import Field, ValidationError, field_validator, model_validator

from bookmatch_ml.schemas import (
    Book,
    DocumentType,
    EvidenceTier,
    SourceEvidenceProvenance,
    SourceType,
    StrictModel,
    TextExtent,
)

EvidenceType = Literal[
    "toc_exact",
    "toc_same_work",
    "toc_public_web_exact",
    "toc_unspecified",
    "description",
    "document",
    "subject",
    "metadata_minimal",
]
EditionRelation = Literal["exact", "same_work", "canonical_record", "unspecified"]


# EvidenceTier and SourceEvidenceProvenance now live in schemas.py so the canonical
# four-file loader and this handoff loader cannot drift apart. They stay importable
# from here for existing callers.
__all__ = [
    "EvidenceTier",
    "EvidenceType",
    "EditionRelation",
    "SourceEvidenceProvenance",
    "BookEvidenceImportError",
]


class BookEvidenceImportError(ValueError):
    """The book-evidence handoff is malformed or internally inconsistent."""


class ImportedEvidenceItem(StrictModel):
    """One evidence row with provenance retained and no ML weight."""

    evidence_id: str = Field(pattern=r"^evidence_[0-9a-f]{20}$")
    evidence_type: EvidenceType
    text: str = Field(min_length=1)
    source_id: str = Field(min_length=1)
    provider: str = Field(min_length=1)
    source_type: SourceType
    source_url: str = Field(min_length=1)
    source_retrieved_at: datetime
    source_content_hash: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    source_evidence_tier: EvidenceTier | None = None
    source_evidence: SourceEvidenceProvenance | None = None
    edition_relation: EditionRelation
    provenance_hash: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    toc_entry_id: str | None = None
    parent_entry_id: str | None = None
    level: int | None = Field(default=None, ge=1)
    order_index: int | None = Field(default=None, ge=0)
    label: str | None = None
    toc_path: list[str] | None = None
    document_id: str | None = None
    document_type: DocumentType | None = None
    document_content_hash: str | None = Field(
        default=None,
        pattern=r"^sha256:[0-9a-f]{64}$",
    )
    metadata_field: Literal["title", "topics"] | None = None

    @field_validator("text")
    @classmethod
    def text_must_not_be_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("evidence text must not be blank")
        return value

    @field_validator("source_retrieved_at")
    @classmethod
    def retrieval_time_must_include_timezone(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("source_retrieved_at must include a timezone")
        return value

    @model_validator(mode="after")
    def evidence_shape_and_category_must_agree(self) -> "ImportedEvidenceItem":
        if self.source_evidence is not None:
            if self.source_evidence_tier != self.source_evidence.tier:
                raise ValueError("source evidence tier does not match provenance")
            expected_provenance_type = (
                "toc"
                if self.evidence_type.startswith("toc_")
                else "metadata"
                if self.evidence_type in {"subject", "metadata_minimal"}
                else None
            )
            if (
                expected_provenance_type is not None
                and self.source_evidence.evidence_type != expected_provenance_type
            ):
                raise ValueError("source evidence type does not match evidence category")
        elif self.source_evidence_tier is not None:
            raise ValueError("source evidence tier requires source evidence provenance")

        toc_fields = (
            self.toc_entry_id,
            self.parent_entry_id,
            self.level,
            self.order_index,
            self.label,
            self.toc_path,
        )
        document_fields = (
            self.document_id,
            self.document_type,
            self.document_content_hash,
        )
        if self.evidence_type.startswith("toc_"):
            if self.toc_entry_id is None or self.level is None or not self.toc_path:
                raise ValueError("TOC evidence requires entry identity, level, and path")
            if (
                any(value is not None for value in document_fields)
                or self.metadata_field is not None
            ):
                raise ValueError("TOC evidence cannot carry document or metadata identity")
        elif self.evidence_type in {"description", "document"}:
            if (
                self.document_id is None
                or self.document_type is None
                or self.document_content_hash is None
            ):
                raise ValueError("document evidence requires identity, type, and content hash")
            if any(value is not None for value in toc_fields) or self.metadata_field is not None:
                raise ValueError("document evidence cannot carry TOC or metadata identity")
        else:
            if self.metadata_field is None:
                raise ValueError("metadata evidence requires its canonical field")
            if any(value is not None for value in (*toc_fields, *document_fields)):
                raise ValueError("metadata evidence cannot carry TOC or document identity")
            if self.evidence_type == "subject" and self.metadata_field != "topics":
                raise ValueError("subject evidence must identify topics")
            if self.evidence_type == "metadata_minimal" and self.metadata_field != "title":
                raise ValueError("minimal metadata evidence must identify title")

        if self.evidence_type == "toc_same_work":
            if self.edition_relation != "same_work":
                raise ValueError("same-Work TOC must retain same_work relation")
            if self.source_evidence_tier != "same_work_alternate_edition_toc":
                raise ValueError("same-Work TOC must retain alternate-edition provenance")
        elif self.evidence_type in {"toc_exact", "toc_public_web_exact"}:
            if self.edition_relation != "exact":
                raise ValueError("exact TOC must retain exact relation")
        elif self.evidence_type == "toc_unspecified":
            if self.edition_relation != "unspecified":
                raise ValueError("unverified TOC must retain unspecified relation")
        elif self.evidence_type in {"subject", "metadata_minimal"}:
            if self.edition_relation != "canonical_record":
                raise ValueError("canonical metadata must retain canonical_record relation")
        return self


class ImportedBookEvidence(StrictModel):
    """All source-aware evidence available for one canonical book."""

    schema_version: Literal[1]
    contract_version: Literal["book-evidence-v1"]
    book: Book
    book_content_hash: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    evidence: list[ImportedEvidenceItem] = Field(min_length=1)

    @model_validator(mode="after")
    def english_book_requires_v3(self) -> "ImportedBookEvidence":
        if self.schema_version < 3 and (
            self.book.en_title is not None or self.book.en_subtitle is not None
        ):
            raise ValueError("English book fields require book-evidence-v3")
        return self


class ImportedEvidenceItemV2(ImportedEvidenceItem):
    source_external_id: str | None
    source_license: str | None
    source_rights_note: str | None
    text_extent: TextExtent | None

    @model_validator(mode="after")
    def extent_requires_document(self) -> "ImportedEvidenceItemV2":
        if self.text_extent is not None and self.document_id is None:
            raise ValueError("text extent requires document evidence")
        return self


class ImportedBookEvidenceV2(ImportedBookEvidence):
    schema_version: Literal[2]
    contract_version: Literal["book-evidence-v2"]
    evidence: list[ImportedEvidenceItemV2] = Field(min_length=1)


class ImportedEvidenceItemV3(ImportedEvidenceItemV2):
    en_text: str | None = Field(min_length=1)

    @field_validator("en_text")
    @classmethod
    def english_text_not_blank(cls, value: str | None) -> str | None:
        if value is not None and not value.strip():
            raise ValueError("English analysis text must not be blank")
        return value


class ImportedBookEvidenceV3(ImportedBookEvidence):
    schema_version: Literal[3]
    contract_version: Literal["book-evidence-v3"]
    evidence: list[ImportedEvidenceItemV3] = Field(min_length=1)


class ConceptCandidateInput(StrictModel):
    """Unweighted text candidate for a later source-aware concept experiment."""

    book_id: str = Field(min_length=1)
    evidence_id: str = Field(min_length=1)
    evidence_type: EvidenceType
    text: str = Field(min_length=1)
    en_text: str | None = None
    provider: str = Field(min_length=1)
    source_type: SourceType
    source_evidence_tier: EvidenceTier | None = None
    edition_relation: EditionRelation
    toc_path: list[str] | None = None
    document_id: str | None = None
    document_type: DocumentType | None = None
    text_extent: TextExtent | None = None
    source_url: str | None = None
    source_external_id: str | None = None
    source_license: str | None = None
    source_rights_note: str | None = None


class BookConceptCandidateInputs(StrictModel):
    """Candidate text grouped by book without assigning concepts or weights."""

    book: Book
    candidates: list[ConceptCandidateInput] = Field(min_length=1)


def _sha256_json(value: Any) -> str:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return f"sha256:{hashlib.sha256(encoded.encode('utf-8')).hexdigest()}"


def _provenance_hash(item: ImportedEvidenceItem) -> str:
    payload = item.model_dump(mode="json")
    payload.pop("evidence_id")
    payload.pop("provenance_hash")
    return _sha256_json(payload)


def _source_snapshot(item: ImportedEvidenceItem) -> dict[str, Any]:
    return {
        **(
            {
                "source_external_id": item.source_external_id,
                "source_license": item.source_license,
                "source_rights_note": item.source_rights_note,
            }
            if isinstance(item, ImportedEvidenceItemV2)
            else {}
        ),
        "provider": item.provider,
        "source_type": item.source_type,
        "source_url": item.source_url,
        "source_retrieved_at": item.source_retrieved_at.isoformat(),
        "source_content_hash": item.source_content_hash,
        "source_evidence_tier": item.source_evidence_tier,
        "source_evidence": (
            item.source_evidence.model_dump(mode="json")
            if item.source_evidence is not None
            else None
        ),
    }


def _validate_records(records: list[ImportedBookEvidence]) -> list[str]:
    errors: list[str] = []
    if len({record.contract_version for record in records}) > 1:
        errors.append("mixed book evidence contract versions")
    book_ids = [record.book.book_id for record in records]
    evidence_ids = [item.evidence_id for record in records for item in record.evidence]
    for kind, values in (("book", book_ids), ("evidence", evidence_ids)):
        errors.extend(
            f"duplicate {kind} ID: {value}"
            for value, count in sorted(Counter(values).items())
            if count > 1
        )

    sources: dict[str, tuple[str, dict[str, Any]]] = {}
    for record in records:
        expected_book_hash = _sha256_json(record.book.model_dump(mode="json"))
        if record.book_content_hash != expected_book_hash:
            errors.append(f"book content hash mismatch: {record.book.book_id}")
        for item in record.evidence:
            if _provenance_hash(item) != item.provenance_hash:
                errors.append(f"evidence provenance hash mismatch: {item.evidence_id}")
            snapshot = _source_snapshot(item)
            existing = sources.get(item.source_id)
            if existing is None:
                sources[item.source_id] = (record.book.book_id, snapshot)
            elif existing != (record.book.book_id, snapshot):
                errors.append(f"inconsistent source snapshot: {item.source_id}")
    return errors


def load_book_evidence(path: Path) -> list[ImportedBookEvidence]:
    """Load and validate a complete, single-version book evidence JSONL artifact."""

    path = path.expanduser()
    if not path.is_file():
        raise BookEvidenceImportError(f"missing book evidence artifact: {path}")
    records: list[ImportedBookEvidence] = []
    try:
        stream = path.open(encoding="utf-8")
    except OSError as exc:
        raise BookEvidenceImportError(f"cannot read book evidence artifact: {path}: {exc}") from exc
    with stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                decoded = json.loads(line)
                model = (
                    ImportedBookEvidenceV3
                    if isinstance(decoded, dict) and decoded.get("schema_version") == 3
                    else ImportedBookEvidenceV2
                    if isinstance(decoded, dict) and decoded.get("schema_version") == 2
                    else ImportedBookEvidence
                )
                records.append(model.model_validate_json(line))
            except (ValidationError, ValueError) as exc:
                raise BookEvidenceImportError(
                    f"invalid book evidence line {line_number}: {exc}"
                ) from exc
    if not records:
        raise BookEvidenceImportError("book evidence artifact contains no records")
    errors = _validate_records(records)
    if errors:
        details = "\n".join(f"- {error}" for error in errors)
        raise BookEvidenceImportError(f"book evidence validation failed:\n{details}")
    return sorted(records, key=lambda record: record.book.book_id)


def build_concept_candidate_inputs(
    records: list[ImportedBookEvidence],
) -> list[BookConceptCandidateInputs]:
    """Expose all collected text and source labels without choosing concepts or weights."""

    return [
        BookConceptCandidateInputs(
            book=record.book,
            candidates=[
                ConceptCandidateInput(
                    book_id=record.book.book_id,
                    evidence_id=item.evidence_id,
                    evidence_type=item.evidence_type,
                    text=item.text,
                    en_text=getattr(item, "en_text", None),
                    provider=item.provider,
                    source_type=item.source_type,
                    source_evidence_tier=item.source_evidence_tier,
                    edition_relation=item.edition_relation,
                    toc_path=item.toc_path,
                    document_id=item.document_id,
                    document_type=item.document_type,
                    source_url=item.source_url,
                    text_extent=getattr(item, "text_extent", None),
                    source_external_id=getattr(item, "source_external_id", None),
                    source_license=getattr(item, "source_license", None),
                    source_rights_note=getattr(item, "source_rights_note", None),
                )
                for item in record.evidence
            ],
        )
        for record in sorted(records, key=lambda item: item.book.book_id)
    ]


def summarize_book_evidence(records: list[ImportedBookEvidence]) -> dict[str, Any]:
    """Report source-aware availability without interpreting evidence quality numerically."""

    toc_types = {"toc_exact", "toc_same_work", "toc_public_web_exact", "toc_unspecified"}
    rows = Counter(item.evidence_type for record in records for item in record.evidence)
    referenced_sources: dict[str, str] = {}
    for record in records:
        for item in record.evidence:
            referenced_sources[item.source_id] = item.provider
    return {
        "contract_version": records[0].contract_version if records else "book-evidence-v1",
        "document_extent_rows": dict(
            sorted(
                Counter(
                    item.text_extent.scope if getattr(item, "text_extent", None) else "unknown"
                    for record in records
                    for item in record.evidence
                    if item.document_id is not None
                ).items()
            )
        ),
        "total_books": len(records),
        "books_with_toc_evidence": sum(
            any(item.evidence_type in toc_types for item in record.evidence) for record in records
        ),
        "books_with_exact_toc": sum(
            any(
                item.evidence_type in {"toc_exact", "toc_public_web_exact"}
                for item in record.evidence
            )
            for record in records
        ),
        "books_with_public_web_toc": sum(
            any(item.evidence_type == "toc_public_web_exact" for item in record.evidence)
            for record in records
        ),
        "books_with_same_work_alternate_toc": sum(
            any(item.evidence_type == "toc_same_work" for item in record.evidence)
            for record in records
        ),
        "books_with_unspecified_toc": sum(
            any(item.evidence_type == "toc_unspecified" for item in record.evidence)
            for record in records
        ),
        "metadata_fallback_only": sum(
            all(item.evidence_type not in toc_types for item in record.evidence)
            for record in records
        ),
        "books_with_zero_evidence": sum(not record.evidence for record in records),
        "evidence_type_rows": dict(sorted(rows.items())),
        "provider_source_counts": dict(sorted(Counter(referenced_sources.values()).items())),
    }
