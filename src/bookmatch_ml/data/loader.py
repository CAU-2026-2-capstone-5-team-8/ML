"""Strict JSONL loader and visible cross-record validation."""

from collections import Counter
from pathlib import Path

from pydantic import BaseModel, ValidationError

from bookmatch_ml.schemas import Book, CanonicalDataset, Document, Source, TocEntry

_CANONICAL_FILES = ("books.jsonl", "documents.jsonl", "toc.jsonl", "sources.jsonl")


class CanonicalDataError(ValueError):
    """Canonical input is malformed, incomplete, or internally inconsistent."""


def _read_jsonl[ModelT: BaseModel](path: Path, model: type[ModelT]) -> list[ModelT]:
    if not path.is_file():
        raise CanonicalDataError(f"missing canonical dataset file: {path}")

    records: list[ModelT] = []
    try:
        stream = path.open(encoding="utf-8")
    except OSError as exc:
        raise CanonicalDataError(f"cannot read canonical dataset file: {path}: {exc}") from exc

    with stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                records.append(model.model_validate_json(line))
            except (ValidationError, ValueError) as exc:
                raise CanonicalDataError(f"invalid {path.name} line {line_number}: {exc}") from exc
    return records


def _duplicates(values: list[str]) -> list[str]:
    return sorted(value for value, count in Counter(values).items() if count > 1)


def _validate_references(dataset: CanonicalDataset) -> list[str]:
    errors: list[str] = []
    book_ids = {book.book_id for book in dataset.books}
    sources_by_id = {source.source_id: source for source in dataset.sources}
    toc_by_id = {entry.toc_entry_id: entry for entry in dataset.toc}
    sourced_book_ids = {source.book_id for source in dataset.sources}

    identifiers = (
        ("book_id", [book.book_id for book in dataset.books]),
        ("source_id", [source.source_id for source in dataset.sources]),
        ("document_id", [document.document_id for document in dataset.documents]),
        ("toc_entry_id", [entry.toc_entry_id for entry in dataset.toc]),
        ("ISBN-13", [book.isbn_13 for book in dataset.books if book.isbn_13]),
        ("ISBN-10", [book.isbn_10 for book in dataset.books if book.isbn_10]),
    )
    for kind, values in identifiers:
        errors.extend(f"duplicate {kind}: {value}" for value in _duplicates(values))

    for source in dataset.sources:
        if source.book_id not in book_ids:
            errors.append(f"source {source.source_id} references missing book {source.book_id}")
    for book in dataset.books:
        if book.book_id not in sourced_book_ids:
            errors.append(f"book {book.book_id} has no source")

    for document in dataset.documents:
        if document.book_id not in book_ids:
            errors.append(
                f"document {document.document_id} references missing book {document.book_id}"
            )
        source = sources_by_id.get(document.source_id)
        if source is None:
            errors.append(
                f"document {document.document_id} references missing source {document.source_id}"
            )
        elif source.book_id != document.book_id:
            errors.append(
                f"document {document.document_id} references source {document.source_id} "
                f"owned by different book {source.book_id}"
            )

    for entry in dataset.toc:
        if entry.book_id not in book_ids:
            errors.append(f"TOC {entry.toc_entry_id} references missing book {entry.book_id}")
        source = sources_by_id.get(entry.source_id)
        if source is None:
            errors.append(f"TOC {entry.toc_entry_id} references missing source {entry.source_id}")
        elif source.book_id != entry.book_id:
            errors.append(
                f"TOC {entry.toc_entry_id} references source {entry.source_id} "
                f"owned by different book {source.book_id}"
            )
        if entry.parent_entry_id is not None:
            parent = toc_by_id.get(entry.parent_entry_id)
            if parent is None:
                errors.append(
                    f"TOC {entry.toc_entry_id} references missing parent {entry.parent_entry_id}"
                )
            elif parent.book_id != entry.book_id:
                errors.append(
                    f"TOC {entry.toc_entry_id} references parent {entry.parent_entry_id} "
                    f"owned by different book {parent.book_id}"
                )
            elif parent.level >= entry.level:
                errors.append(
                    f"TOC {entry.toc_entry_id} level {entry.level} must be below parent "
                    f"{parent.toc_entry_id} level {parent.level}"
                )
    return errors


def load_canonical_dataset(directory: Path) -> CanonicalDataset:
    """Load and validate one complete four-file canonical dataset."""

    directory = directory.expanduser()
    missing = [name for name in _CANONICAL_FILES if not (directory / name).is_file()]
    if missing:
        raise CanonicalDataError(
            f"incomplete canonical dataset at {directory}; missing: {', '.join(missing)}"
        )

    dataset = CanonicalDataset(
        books=_read_jsonl(directory / "books.jsonl", Book),
        documents=_read_jsonl(directory / "documents.jsonl", Document),
        toc=_read_jsonl(directory / "toc.jsonl", TocEntry),
        sources=_read_jsonl(directory / "sources.jsonl", Source),
    )
    errors = _validate_references(dataset)
    if errors:
        details = "\n".join(f"- {error}" for error in sorted(errors))
        raise CanonicalDataError(f"canonical dataset validation failed:\n{details}")
    return dataset
