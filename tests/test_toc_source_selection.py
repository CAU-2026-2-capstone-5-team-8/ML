from pathlib import Path

from bookmatch_ml.concept_v2.toc_source_selection import select_toc_source
from bookmatch_ml.data.evidence import assemble_book_evidence
from bookmatch_ml.data.loader import load_canonical_dataset


def example():
    rows = assemble_book_evidence(
        load_canonical_dataset(Path(__file__).parent / "fixtures/canonical")
    )
    return next(book for book in rows if book.toc)


def test_refresh_duplicates_do_not_inflate_coverage_and_inputs_are_preserved():
    book = example()
    source = book.toc[0].source_id
    entries = [e for e in book.toc if e.source_id == source]
    duplicate = [
        e.model_copy(
            update={
                "source_id": "zzz-refresh",
                "toc_entry_id": "repeat-" + e.toc_entry_id,
                "parent_entry_id": None
                if e.parent_entry_id is None
                else "repeat-" + e.parent_entry_id,
            }
        )
        for e in entries
    ]
    mixed = book.model_copy(update={"toc": entries + duplicate})
    before = mixed.model_dump()
    selected, report = select_toc_source(mixed)
    assert len(selected.toc) == len(entries)
    assert len(report["availableSourceIds"]) == 2
    assert mixed.model_dump() == before


def test_bad_source_is_explicitly_excluded_without_fabricating_missing_toc():
    book = example()
    bad = book.toc[0].model_copy(
        update={
            "source_id": "broken-source",
            "toc_entry_id": "bad-parent",
            "parent_entry_id": "absent",
        }
    )
    mixed = book.model_copy(update={"toc": [bad]})
    selected, report = select_toc_source(mixed)
    assert not selected.toc and not selected.coverage.has_toc
    assert report["invalidSourceIds"] == ["broken-source"] and report["selectedSourceId"] is None
