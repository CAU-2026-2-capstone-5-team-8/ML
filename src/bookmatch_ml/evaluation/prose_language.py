"""Read-only language diagnostics for collected prose and the current baseline."""

import re
import unicodedata
from collections import Counter
from typing import Literal

from pydantic import Field

from bookmatch_ml.book.difficulty import build_difficulty_profile
from bookmatch_ml.config import LoadedFeatureConfig
from bookmatch_ml.data.evidence import PROSE_DOCUMENT_TYPES, assemble_book_evidence
from bookmatch_ml.schemas import (
    CanonicalDataset,
    DocumentDifficulty,
    ExcludedProseDocument,
    StrictModel,
    TextExtent,
)


class ProseLanguageRow(StrictModel):
    book_id: str
    title: str
    document_id: str
    document_type: str
    document_content_hash: str
    declared_book_language: str
    language_relation: Literal["matches_declared_baseline", "different_from_baseline", "unknown"]
    text_extent: TextExtent | None
    source_id: str
    source_url: str
    source_rights_note: str | None
    character_count: int
    letter_counts: dict[str, int]
    hangul_letter_share: float | None
    review_flags: list[str]
    baseline_result: DocumentDifficulty | ExcludedProseDocument
    # This tool does not create human labels or validate difficulty estimates.
    human_language_review: str | None = None
    human_prose_review: str | None = None
    human_relative_difficulty: float | None = None


class ProseLanguageAudit(StrictModel):
    report_version: Literal["prose-language-audit-v1"] = "prose-language-audit-v1"
    baseline_language: str
    evaluation_status: Literal["not_evaluated"] = "not_evaluated"
    analyzed_text_field: Literal["text"] = "text"
    feature_config_hash: str
    canonical_hashes: dict[str, str] = Field(default_factory=dict)
    total_books: int
    books_with_prose: int
    books_without_prose: int
    prose_document_count: int
    baseline_scored_document_count: int
    language_relation_counts: dict[str, int]
    documents: list[ProseLanguageRow]


def _language(value: str) -> str | None:
    return {"eng": "en", "kor": "ko", "und": None, "mul": None, "zxx": None}.get(value, value)


def build_prose_language_audit(
    dataset: CanonicalDataset,
    features: LoadedFeatureConfig,
    baseline_language: str,
) -> ProseLanguageAudit:
    """Observe script and declared-language mismatch without inferring reading ability."""
    if not re.fullmatch(r"[a-z]{2,3}", baseline_language) or _language(baseline_language) is None:
        raise ValueError(
            "baseline language requires an explicit two/three-letter code, not und/mul/zxx"
        )
    baseline_language = _language(baseline_language)
    sources = {source.source_id: source for source in dataset.sources}
    rows = []
    for evidence in assemble_book_evidence(dataset):
        difficulty = build_difficulty_profile(evidence, features)
        results = {
            d.document_id: d for d in [*difficulty.documents, *difficulty.excluded_documents]
        }
        for document in evidence.documents:
            if document.document_type not in PROSE_DOCUMENT_TYPES:
                continue
            letters = Counter({"hangul": 0, "latin": 0, "other": 0})
            for character in document.text:
                if character.isalpha():
                    name = unicodedata.name(character, "")
                    script = (
                        "hangul"
                        if name.startswith("HANGUL")
                        else "latin"
                        if "LATIN" in name
                        else "other"
                    )
                    letters[script] += 1
            declared = _language(evidence.metadata.language)
            relation = (
                "unknown"
                if declared is None
                else "matches_declared_baseline"
                if declared == baseline_language
                else "different_from_baseline"
            )
            result = results[document.document_id]
            flags = []
            if relation != "matches_declared_baseline":
                flags.append(
                    "declared_language_unknown"
                    if declared is None
                    else "declared_language_mismatch"
                )
            if baseline_language == "en" and letters["hangul"]:
                flags.append("hangul_present_in_english_baseline_input")
            if isinstance(result, DocumentDifficulty) and result.raw.concept_mention_count == 0:
                flags.append("no_configured_concept_alias_matches")
            source = sources[document.source_id]
            rows.append(
                ProseLanguageRow(
                    book_id=evidence.book_id,
                    title=evidence.metadata.title,
                    document_id=document.document_id,
                    document_type=document.document_type,
                    document_content_hash=document.content_hash,
                    declared_book_language=evidence.metadata.language,
                    language_relation=relation,
                    text_extent=document.text_extent,
                    source_id=source.source_id,
                    source_url=source.url,
                    source_rights_note=source.rights_note,
                    character_count=len(document.text),
                    letter_counts=dict(letters),
                    hangul_letter_share=letters["hangul"] / sum(letters.values())
                    if sum(letters.values())
                    else None,
                    review_flags=flags,
                    baseline_result=result,
                )
            )
    prose_books = len({row.book_id for row in rows})
    return ProseLanguageAudit(
        baseline_language=baseline_language,
        feature_config_hash=features.content_hash,
        total_books=len(dataset.books),
        books_with_prose=prose_books,
        books_without_prose=len(dataset.books) - prose_books,
        prose_document_count=len(rows),
        baseline_scored_document_count=sum(
            isinstance(row.baseline_result, DocumentDifficulty) for row in rows
        ),
        language_relation_counts=dict(
            sorted(Counter(row.language_relation for row in rows).items())
        ),
        documents=rows,
    )
