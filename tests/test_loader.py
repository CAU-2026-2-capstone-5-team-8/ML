import json
import shutil
from pathlib import Path

import pytest

from bookmatch_ml.data.loader import CanonicalDataError, load_canonical_dataset

FIXTURE_DIR = Path(__file__).parent / "fixtures" / "canonical"


def _dataset_copy(tmp_path: Path) -> Path:
    target = tmp_path / "canonical"
    shutil.copytree(FIXTURE_DIR, target)
    return target


def test_loads_all_four_canonical_files() -> None:
    dataset = load_canonical_dataset(FIXTURE_DIR)

    assert len(dataset.books) == 2
    assert len(dataset.documents) == 2
    assert len(dataset.toc) == 2
    assert len(dataset.sources) == 2


def test_loads_additive_source_evidence_provenance(tmp_path: Path) -> None:
    directory = _dataset_copy(tmp_path)
    path = directory / "sources.jsonl"
    lines = path.read_text(encoding="utf-8").splitlines()
    record = json.loads(lines[0])
    record["evidence"] = {
        "evidence_type": "toc",
        "tier": "exact_edition_toc",
        "target_isbn": None,
        "target_title": "Fixture Operating Systems",
        "target_authors": ["Fixture Author"],
        "source_edition_id": "edition-1",
        "source_isbns": [],
        "source_title": "Fixture Operating Systems",
        "source_author_ids": [],
        "same_edition": True,
        "source_document_type": "fixture",
        "discovery_method": "fixture_import",
        "match_basis": ["exact_title"],
        "validation_status": "strong",
    }
    lines[0] = json.dumps(record)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    dataset = load_canonical_dataset(directory)

    assert dataset.sources[0].evidence is not None
    assert dataset.sources[0].evidence.tier == "exact_edition_toc"


def test_missing_file_fails_visibly(tmp_path: Path) -> None:
    directory = _dataset_copy(tmp_path)
    (directory / "sources.jsonl").unlink()

    with pytest.raises(CanonicalDataError, match=r"missing: sources\.jsonl"):
        load_canonical_dataset(directory)


def test_invalid_record_reports_file_and_line(tmp_path: Path) -> None:
    directory = _dataset_copy(tmp_path)
    path = directory / "documents.jsonl"
    lines = path.read_text(encoding="utf-8").splitlines()
    record = json.loads(lines[1])
    record["document_type"] = "web_page"
    lines[1] = json.dumps(record)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    with pytest.raises(CanonicalDataError, match=r"documents\.jsonl line 2"):
        load_canonical_dataset(directory)


def test_duplicate_identifiers_fail_visibly(tmp_path: Path) -> None:
    directory = _dataset_copy(tmp_path)
    path = directory / "documents.jsonl"
    lines = path.read_text(encoding="utf-8").splitlines()
    duplicate = json.loads(lines[1])
    duplicate["document_id"] = "doc_description"
    lines[1] = json.dumps(duplicate)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    with pytest.raises(CanonicalDataError, match="duplicate document_id: doc_description"):
        load_canonical_dataset(directory)


def test_broken_cross_file_reference_fails_visibly(tmp_path: Path) -> None:
    directory = _dataset_copy(tmp_path)
    path = directory / "documents.jsonl"
    lines = path.read_text(encoding="utf-8").splitlines()
    record = json.loads(lines[0])
    record["source_id"] = "missing_source"
    lines[0] = json.dumps(record)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    with pytest.raises(CanonicalDataError, match="references missing source missing_source"):
        load_canonical_dataset(directory)


def test_source_must_belong_to_the_referenced_book(tmp_path: Path) -> None:
    directory = _dataset_copy(tmp_path)
    path = directory / "documents.jsonl"
    lines = path.read_text(encoding="utf-8").splitlines()
    record = json.loads(lines[0])
    record["source_id"] = "source_222"
    lines[0] = json.dumps(record)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    with pytest.raises(CanonicalDataError, match="owned by different book"):
        load_canonical_dataset(directory)


def test_toc_parent_must_exist(tmp_path: Path) -> None:
    directory = _dataset_copy(tmp_path)
    path = directory / "toc.jsonl"
    lines = path.read_text(encoding="utf-8").splitlines()
    record = json.loads(lines[1])
    record["parent_entry_id"] = "missing_parent"
    lines[1] = json.dumps(record)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    with pytest.raises(CanonicalDataError, match="references missing parent missing_parent"):
        load_canonical_dataset(directory)


def test_conflicting_canonical_book_identity_fails_visibly(tmp_path: Path) -> None:
    directory = _dataset_copy(tmp_path)
    path = directory / "books.jsonl"
    lines = path.read_text(encoding="utf-8").splitlines()
    record = json.loads(lines[0])
    record["isbn_13"] = "9780130319999"
    lines[0] = json.dumps(record)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    with pytest.raises(CanonicalDataError, match="book_id must use the record's ISBN-13"):
        load_canonical_dataset(directory)


@pytest.mark.parametrize(
    ("book_id", "message"),
    [
        ("isbn13:9780130319999", "ISBN-13 book_id requires isbn_13"),
        ("isbn10:0131103628", "ISBN-10 book_id requires isbn_10"),
    ],
)
def test_isbn_book_id_requires_its_canonical_field(
    tmp_path: Path,
    book_id: str,
    message: str,
) -> None:
    directory = _dataset_copy(tmp_path)
    path = directory / "books.jsonl"
    lines = path.read_text(encoding="utf-8").splitlines()
    record = json.loads(lines[0])
    record["book_id"] = book_id
    lines[0] = json.dumps(record)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    with pytest.raises(CanonicalDataError, match=message):
        load_canonical_dataset(directory)
