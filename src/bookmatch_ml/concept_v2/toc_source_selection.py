"""Choose one coherent source TOC per book without rewriting canonical evidence."""

from collections import defaultdict

from bookmatch_ml.concept_v2.toc import TocTreeError, reconstruct_toc
from bookmatch_ml.data.evidence import calculate_evidence_coverage

POLICY = "single-source-complete-toc-v1"


def select_toc_source(book):
    sources = defaultdict(list)
    for entry in book.toc:
        sources[entry.source_id].append(entry)
    valid = {}
    invalid = []
    for source, entries in sorted(sources.items()):
        try:
            tree = reconstruct_toc(entries)
            valid[source] = [visit.entry for visit in tree.traversal]
        except TocTreeError:
            invalid.append(source)
    chosen = min(valid, key=lambda source: (-len(valid[source]), source)) if valid else None
    selected = valid[chosen] if chosen is not None else []
    result = book.model_copy(
        update={"toc": selected, "coverage": calculate_evidence_coverage(book.documents, selected)}
    )
    return result, {
        "bookId": book.book_id,
        "selectedSourceId": chosen,
        "availableSourceIds": sorted(sources),
        "invalidSourceIds": invalid,
        "selectedEntryCount": len(selected),
        "policy": POLICY,
    }
