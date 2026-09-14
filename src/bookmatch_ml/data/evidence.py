"""Deterministic assembly of canonical records into per-book evidence."""

from collections import defaultdict

from bookmatch_ml.schemas import (
    BookEvidence,
    CanonicalDataset,
    Document,
    EvidenceCoverage,
    TocEntry,
)

PROSE_DOCUMENT_TYPES = frozenset({"preface", "introduction", "preview", "sample_chapter", "other"})


def _hierarchical_toc_order(entries: list[TocEntry]) -> list[TocEntry]:
    """Return a stable depth-first order using parent links and sibling order indexes."""

    children_by_parent: dict[str | None, list[TocEntry]] = defaultdict(list)
    for entry in entries:
        children_by_parent[entry.parent_entry_id].append(entry)
    for siblings in children_by_parent.values():
        siblings.sort(key=lambda item: (item.order_index, item.toc_entry_id))

    ordered: list[TocEntry] = []

    def visit(entry: TocEntry) -> None:
        ordered.append(entry)
        for child in children_by_parent[entry.toc_entry_id]:
            visit(child)

    for root in children_by_parent[None]:
        visit(root)
    return ordered


def calculate_evidence_coverage(
    documents: list[Document],
    toc: list[TocEntry],
) -> EvidenceCoverage:
    """Calculate availability without substituting one evidence type for another."""

    document_types = {document.document_type for document in documents}
    prose_documents = [
        document for document in documents if document.document_type in PROSE_DOCUMENT_TYPES
    ]
    return EvidenceCoverage(
        has_metadata=True,
        has_toc=bool(toc),
        has_description=bool(document_types & {"description", "publisher_summary"}),
        has_preface="preface" in document_types,
        has_introduction="introduction" in document_types,
        has_preview="preview" in document_types,
        has_sample_chapter="sample_chapter" in document_types,
        has_other_text="other" in document_types,
        toc_entry_count=len(toc),
        document_count=len(documents),
        prose_document_count=len(prose_documents),
        prose_character_count=sum(len(document.text) for document in prose_documents),
    )


def assemble_book_evidence(dataset: CanonicalDataset) -> list[BookEvidence]:
    """Attach documents and TOC entries without leaking provider-specific source fields."""

    documents_by_book = defaultdict(list)
    for document in dataset.documents:
        documents_by_book[document.book_id].append(document)

    toc_by_book = defaultdict(list)
    for entry in dataset.toc:
        toc_by_book[entry.book_id].append(entry)

    evidence: list[BookEvidence] = []
    for book in sorted(dataset.books, key=lambda item: item.book_id):
        documents = sorted(documents_by_book[book.book_id], key=lambda item: item.document_id)
        # Parent IDs retain hierarchy; depth-first traversal preserves logical section order.
        toc = _hierarchical_toc_order(toc_by_book[book.book_id])
        evidence.append(
            BookEvidence(
                book_id=book.book_id,
                metadata=book,
                toc=toc,
                documents=documents,
                coverage=calculate_evidence_coverage(documents, toc),
            )
        )
    return evidence
