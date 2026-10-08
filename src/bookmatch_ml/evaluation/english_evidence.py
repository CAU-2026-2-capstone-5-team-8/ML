"""Paired offline comparison of original and stored English TOC evidence."""

import hashlib
from typing import Any

from bookmatch_ml.concept_v2.graph import LoadedConceptGraph
from bookmatch_ml.concept_v2.profile import (
    LoadedConceptMatchingConfig,
    match_concept_text_production,
)
from bookmatch_ml.config import LoadedFeatureConfig
from bookmatch_ml.data.book_evidence import ImportedBookEvidence
from bookmatch_ml.data.evidence import PROSE_DOCUMENT_TYPES


def compare_english_evidence(
    records: list[ImportedBookEvidence],
    features: LoadedFeatureConfig,
    graph: LoadedConceptGraph,
    matching: LoadedConceptMatchingConfig,
) -> dict[str, Any]:
    """Use identical settings and paired rows; missing English never falls back to original."""
    if not records or any(r.contract_version != "book-evidence-v3" for r in records):
        raise ValueError("English comparison requires nonempty book-evidence-v3 input")
    books = []
    for record in sorted(records, key=lambda r: r.book.book_id):
        topics = sorted(set(record.book.topics) & set(graph.graph.nodes))
        original_concepts: set[tuple[str, str]] = set()
        english_concepts: set[tuple[str, str]] = set()
        rows = []
        for item in sorted(record.evidence, key=lambda e: e.evidence_id):
            if item.toc_entry_id is None:
                continue
            row = {
                "evidence_id": item.evidence_id,
                "toc_entry_id": item.toc_entry_id,
                "source_id": item.source_id,
                "provenance_hash": item.provenance_hash,
                "original_text_hash": "sha256:" + hashlib.sha256(item.text.encode()).hexdigest(),
                "english_text_hash": "sha256:" + hashlib.sha256(item.en_text.encode()).hexdigest()
                if item.en_text is not None
                else None,
                "paired": item.en_text is not None and bool(topics),
                "unpaired_reason": "missing_english"
                if item.en_text is None
                else "no_configured_topic"
                if not topics
                else None,
                "original_matches": [],
                "english_matches": [],
                "original_ambiguous_topics": [],
                "english_ambiguous_topics": [],
                "human_translation_review": None,
            }
            if row["paired"]:
                for label, text, concepts in [
                    ("original", item.text, original_concepts),
                    ("english", item.en_text, english_concepts),
                ]:
                    for topic in topics:
                        matches, ambiguous = match_concept_text_production(
                            text, topic, features, graph, matching
                        )
                        if ambiguous:
                            row[label + "_ambiguous_topics"].append(topic)
                            continue
                        for match in matches:
                            concepts.add((topic, match.concept_id))
                            row[label + "_matches"].append(
                                {
                                    "topic": topic,
                                    "concept_id": match.concept_id,
                                    "alias": match.matching_alias,
                                    "method": match.match_method,
                                }
                            )
            rows.append(row)

        def encode(values):
            return [{"topic": t, "concept_id": c} for t, c in sorted(values)]

        books.append(
            {
                "book_id": record.book.book_id,
                "title": record.book.title,
                "toc_rows": len(rows),
                "paired_rows": sum(row["paired"] for row in rows),
                "original_concepts": encode(original_concepts),
                "english_concepts": encode(english_concepts),
                "added_concepts": encode(english_concepts - original_concepts),
                "removed_concepts": encode(original_concepts - english_concepts),
                "rows": rows,
            }
        )
    return {
        "report_version": "english-evidence-comparison-v1",
        "contract_version": "book-evidence-v3",
        "comparison_scope": "paired_toc_only",
        "evaluation_status": "not_evaluated",
        "translation_api_requests": 0,
        "english_field_origin": "not_asserted_by_contract",
        "feature_config_hash": features.content_hash,
        "graph_hash": graph.content_hash,
        "matching_config_hash": matching.content_hash,
        "summary": {
            "book_count": len(books),
            "toc_rows": sum(b["toc_rows"] for b in books),
            "paired_rows": sum(b["paired_rows"] for b in books),
            "original_matched_books": sum(bool(b["original_concepts"]) for b in books),
            "english_matched_books": sum(bool(b["english_concepts"]) for b in books),
            "original_book_concept_pairs": sum(len(b["original_concepts"]) for b in books),
            "english_book_concept_pairs": sum(len(b["english_concepts"]) for b in books),
            "stored_english_prose_documents": sum(
                e.document_type in PROSE_DOCUMENT_TYPES and e.en_text is not None
                for r in records
                for e in r.evidence
            ),
        },
        "books": books,
    }


def prepare_english_evidence_review(
    records: list[ImportedBookEvidence],
    features: LoadedFeatureConfig,
    graph: LoadedConceptGraph,
    matching: LoadedConceptMatchingConfig,
) -> dict[str, Any]:
    """Build a local review packet from validated evidence, never inferred review labels.

    Includes unchanged and unmatched TOCs so matching gains cannot hide translation
    omissions. Prose is listed separately and never scored as part of this comparison.
    """
    report = compare_english_evidence(records, features, graph, matching)
    by_book = {record.book.book_id: record for record in records}
    changed_rows = 0
    prose_count = 0

    def concept_keys(matches):
        return {(m["topic"], m["concept_id"]) for m in matches}

    def encode(values):
        return [{"topic": t, "concept_id": c} for t, c in sorted(values)]

    def review_evidence(item):
        # Retain the complete evidence envelope, including rights, extent, and exact text.
        return {
            "evidence": item.model_dump(mode="json"),
            "translation_status": "missing_english"
            if item.en_text is None
            else "stored_english_unreviewed",
            "human_translation_review": None,
            "human_match_review": None,
            "reviewer": None,
            "review_notes": None,
        }

    for book in report["books"]:
        record = by_book[book["book_id"]]
        evidence = {item.evidence_id: item for item in record.evidence}
        book_added = concept_keys(book["added_concepts"])
        book_removed = concept_keys(book["removed_concepts"])
        for row in book["rows"]:
            original = concept_keys(row["original_matches"])
            english = concept_keys(row["english_matches"])
            added, removed = english - original, original - english
            changed_rows += bool(added or removed)
            row.update(review_evidence(evidence[row["evidence_id"]]))
            row["added_concepts"] = encode(added)
            row["removed_concepts"] = encode(removed)
            row["supports_book_added_concepts"] = encode(added & book_added)
            row["supports_book_removed_concepts"] = encode(removed & book_removed)
        book["prose_samples"] = []
        for item in sorted(record.evidence, key=lambda e: e.evidence_id):
            if item.document_type not in PROSE_DOCUMENT_TYPES:
                continue
            sample = review_evidence(item)
            sample["original_text_hash"] = (
                "sha256:" + hashlib.sha256(item.text.encode()).hexdigest()
            )
            sample["english_text_hash"] = (
                "sha256:" + hashlib.sha256(item.en_text.encode()).hexdigest()
                if item.en_text is not None
                else None
            )
            book["prose_samples"].append(sample)
            prose_count += 1
    report["report_version"] = "english-evidence-review-v1"
    report["review_scope"] = "all_toc_rows_and_prose_samples"
    report["contains_source_text"] = True
    report["summary"]["changed_toc_rows"] = changed_rows
    report["summary"]["prose_samples"] = prose_count
    return report
