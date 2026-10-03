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
