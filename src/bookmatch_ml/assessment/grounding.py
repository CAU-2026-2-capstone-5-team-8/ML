"""Deterministic prose grounding for one narrow comprehension generation slice."""

import hashlib
import json
import re
from datetime import datetime
from typing import Literal

from pydantic import Field, field_validator, model_validator

from bookmatch_ml.assessment.schemas import QuestionSpec
from bookmatch_ml.schemas import CanonicalDataset, StrictModel

GROUNDING_VERSION = "generation-grounding-v1"
DISPLAY_GROUNDING_VERSION = "generation-grounding-v2"
PASSAGE_EXTRACTION_POLICY = "first-concept-sentence-window-v1"
DISPLAY_NORMALIZATION_POLICY = "pdf-display-normalization-v1"
MIN_PASSAGE_CHARACTERS = 600
MAX_PASSAGE_CHARACTERS = 1800
MAX_DISPLAY_PASSAGE_CHARACTERS = 2000
_PROSE_TYPES = {"preface", "introduction", "preview", "sample_chapter", "other"}
_CANONICAL_FILES = {"books.jsonl", "documents.jsonl", "toc.jsonl", "sources.jsonl"}
_HASH_PATTERN = re.compile(r"^sha256:[0-9a-f]{64}$")
_APPROVED_REUSE_LICENSE_MARKERS = (
    "creative commons attribution",
    "cc by",
    "gnu free documentation license",
    "gfdl",
    "public domain",
)
_REUSE_DENIAL_MARKERS = ("all rights reserved", "no reuse", "no redistribution")

# A reviewed allowlist keeps display repair auditable and prevents broad spacing
# heuristics from changing identifiers or mathematical notation. Each rule set is
# bound to one exact canonical source passage and is intentionally fail closed.
_REVIEWED_DISPLAY_REPLACEMENTS = {
    (
        "doc_de934d33d551223812e8",
        "sha256:7220ee5501766f97edabc506e871470356fa2aeb40ec92acfd6f7e44d9ce4ce0",
    ): (
        ("DeﬁnitionAnm×n", "Definition An m×n"),
        ("withm rows\nandn columns", "with m rows\nand n columns"),
        ("anentry", "an entry"),
        (
            "has2 rows and3 columns and so is a2×3 matrix",
            "has 2 rows and 3 columns and so is a 2×3 matrix",
        ),
        ("two-by-\nthree", "two-by-three"),
        ("stated ﬁrst", "stated first"),
        ("row and ﬁrst column", "row and first column"),
        ("isa2,1 =3", "is a2,1 = 3"),
    ),
}


class GroundingError(ValueError):
    """A QuestionSpec cannot be safely grounded in the supplied canonical dataset."""


def _sha256_text(value: str) -> str:
    return "sha256:" + hashlib.sha256(value.encode("utf-8")).hexdigest()


def _sha256_json(value: object) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def _has_approved_reuse_license(value: str) -> bool:
    normalized = " ".join(value.casefold().split())
    return not any(marker in normalized for marker in _REUSE_DENIAL_MARKERS) and any(
        marker in normalized for marker in _APPROVED_REUSE_LICENSE_MARKERS
    )


class GenerationGrounding(StrictModel):
    """Bind one authoritative QuestionSpec to exact, provider-ready source prose."""

    schema_version: Literal[1] = 1
    grounding_version: Literal["generation-grounding-v1"] = GROUNDING_VERSION
    question_spec_id: str = Field(pattern=r"^q_[0-9a-f]{20}$")
    question_spec_hash: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    topic_id: str
    question_type: Literal["comprehension"]
    cognitive_operation: Literal["apply"]
    target_difficulty: Literal[2]
    primary_concept: str
    related_concepts: list[str]
    source_document_id: str
    book_id: str
    document_type: Literal["preface", "introduction", "preview", "sample_chapter", "other"]
    document_content_hash: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    passage_text: str = Field(
        min_length=MIN_PASSAGE_CHARACTERS,
        max_length=MAX_PASSAGE_CHARACTERS,
    )
    passage_hash: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    passage_extraction_policy: Literal["first-concept-sentence-window-v1"] = (
        PASSAGE_EXTRACTION_POLICY
    )
    source_id: str
    provider: str
    source_type: str
    source_url: str
    source_retrieved_at: datetime
    license: str
    rights_note: str | None = None
    source_content_hash: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    edition_relation: Literal["exact", "same_work", "unspecified"]
    canonical_file_hashes: dict[str, str]
    blueprint_hash: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")

    @field_validator(
        "topic_id",
        "primary_concept",
        "source_document_id",
        "book_id",
        "source_id",
        "provider",
        "source_type",
        "source_url",
        "license",
    )
    @classmethod
    def strings_must_be_nonblank_and_trimmed(cls, value: str) -> str:
        if not value or value != value.strip():
            raise ValueError(
                "grounding identity and provenance strings must be nonblank and trimmed"
            )
        return value

    @field_validator("canonical_file_hashes")
    @classmethod
    def canonical_hashes_must_cover_exact_inputs(cls, value: dict[str, str]) -> dict[str, str]:
        if set(value) != _CANONICAL_FILES or any(
            _HASH_PATTERN.fullmatch(digest) is None for digest in value.values()
        ):
            raise ValueError("canonical file hashes must cover all four JSONL inputs")
        return value

    @model_validator(mode="after")
    def hashes_and_target_must_match(self) -> "GenerationGrounding":
        if self.passage_hash != _sha256_text(self.passage_text):
            raise ValueError("passage hash does not match passage text")
        if self.related_concepts:
            raise ValueError("generation-grounding-v1 supports only single-concept apply targets")
        return self


def normalize_display_passage(
    source_passage_text: str,
    *,
    source_document_id: str,
    source_passage_hash: str,
    policy: str = DISPLAY_NORMALIZATION_POLICY,
) -> str:
    """Apply one reviewed, versioned display repair to an exact source passage."""

    if policy != DISPLAY_NORMALIZATION_POLICY:
        raise GroundingError(f"unsupported display normalization policy: {policy!r}")
    if source_passage_hash != _sha256_text(source_passage_text):
        raise GroundingError("source passage hash does not match source passage text")
    replacements = _REVIEWED_DISPLAY_REPLACEMENTS.get((source_document_id, source_passage_hash))
    if replacements is None:
        raise GroundingError(
            "display normalization policy has no reviewed rules for this source passage"
        )

    display_passage = source_passage_text
    for original, replacement in replacements:
        if display_passage.count(original) != 1:
            raise GroundingError(
                "reviewed display normalization input no longer matches its exact source fragment"
            )
        display_passage = display_passage.replace(original, replacement, 1)
    return display_passage


class GenerationGroundingV2(StrictModel):
    """Preserve exact source prose while carrying a reviewed display representation."""

    schema_version: Literal[2] = 2
    grounding_version: Literal["generation-grounding-v2"] = DISPLAY_GROUNDING_VERSION
    question_spec_id: str = Field(pattern=r"^q_[0-9a-f]{20}$")
    question_spec_hash: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    topic_id: str
    question_type: Literal["comprehension"]
    cognitive_operation: Literal["apply"]
    target_difficulty: Literal[2]
    primary_concept: str
    related_concepts: list[str]
    source_document_id: str
    book_id: str
    document_type: Literal["preface", "introduction", "preview", "sample_chapter", "other"]
    document_content_hash: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    source_passage_text: str = Field(
        min_length=MIN_PASSAGE_CHARACTERS,
        max_length=MAX_PASSAGE_CHARACTERS,
    )
    source_passage_hash: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    source_passage_extraction_policy: Literal["first-concept-sentence-window-v1"] = (
        PASSAGE_EXTRACTION_POLICY
    )
    display_passage_text: str = Field(
        min_length=MIN_PASSAGE_CHARACTERS,
        max_length=MAX_DISPLAY_PASSAGE_CHARACTERS,
    )
    display_passage_hash: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    display_normalization_policy: Literal["pdf-display-normalization-v1"] = (
        DISPLAY_NORMALIZATION_POLICY
    )
    source_id: str
    provider: str
    source_type: str
    source_url: str
    source_retrieved_at: datetime
    license: str
    rights_note: str | None = None
    source_content_hash: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    edition_relation: Literal["exact", "same_work", "unspecified"]
    canonical_file_hashes: dict[str, str]
    blueprint_hash: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")

    @field_validator(
        "topic_id",
        "primary_concept",
        "source_document_id",
        "book_id",
        "source_id",
        "provider",
        "source_type",
        "source_url",
        "license",
    )
    @classmethod
    def strings_must_be_nonblank_and_trimmed(cls, value: str) -> str:
        if not value or value != value.strip():
            raise ValueError(
                "grounding identity and provenance strings must be nonblank and trimmed"
            )
        return value

    @field_validator("canonical_file_hashes")
    @classmethod
    def canonical_hashes_must_cover_exact_inputs(cls, value: dict[str, str]) -> dict[str, str]:
        if set(value) != _CANONICAL_FILES or any(
            _HASH_PATTERN.fullmatch(digest) is None for digest in value.values()
        ):
            raise ValueError("canonical file hashes must cover all four JSONL inputs")
        return value

    @model_validator(mode="after")
    def hashes_policy_and_target_must_match(self) -> "GenerationGroundingV2":
        if self.source_passage_hash != _sha256_text(self.source_passage_text):
            raise ValueError("source passage hash does not match source passage text")
        if self.display_passage_hash != _sha256_text(self.display_passage_text):
            raise ValueError("display passage hash does not match display passage text")
        expected_display = normalize_display_passage(
            self.source_passage_text,
            source_document_id=self.source_document_id,
            source_passage_hash=self.source_passage_hash,
            policy=self.display_normalization_policy,
        )
        if self.display_passage_text != expected_display:
            raise ValueError(
                "display passage does not match the deterministic normalization policy"
            )
        if self.related_concepts:
            raise ValueError("generation-grounding-v2 supports only single-concept apply targets")
        return self


def _sentence_spans(text: str) -> list[tuple[int, int]]:
    """Return contiguous source-text spans ending at deterministic sentence punctuation."""

    spans: list[tuple[int, int]] = []
    start = 0
    for match in re.finditer(r"[.!?](?=\s)", text):
        end = match.end()
        if end > start:
            spans.append((start, end))
        start = end
        while start < len(text) and text[start].isspace():
            start += 1
    if start < len(text):
        spans.append((start, len(text)))
    return spans


def extract_passage(text: str, concept: str) -> str:
    """Select the first concept-bearing sentence and enough following source prose."""

    if not text or not text.strip():
        raise GroundingError("source document text is blank")
    if not concept or concept != concept.strip():
        raise GroundingError("primary concept must be nonblank and trimmed")
    pattern = re.compile(rf"(?<!\w){re.escape(concept)}(?!\w)", re.IGNORECASE)
    spans = _sentence_spans(text)
    try:
        index = next(
            index for index, (start, end) in enumerate(spans) if pattern.search(text[start:end])
        )
    except StopIteration as exc:
        raise GroundingError("source document has no concept-relevant passage") from exc

    start, end = spans[index]
    if end - start > MAX_PASSAGE_CHARACTERS:
        raise GroundingError("concept-bearing sentence exceeds the passage length limit")
    while end - start < MIN_PASSAGE_CHARACTERS and index + 1 < len(spans):
        next_end = spans[index + 1][1]
        if next_end - start > MAX_PASSAGE_CHARACTERS:
            break
        index += 1
        end = next_end

    passage = text[start:end].strip()
    if len(passage) < MIN_PASSAGE_CHARACTERS:
        raise GroundingError("concept-relevant passage is too short")
    if len(passage) > MAX_PASSAGE_CHARACTERS:
        raise GroundingError("extracted passage exceeds the passage length limit")
    if passage not in text:
        raise GroundingError("extracted passage is not an exact source-text substring")
    return passage


def _edition_relation(
    source_evidence: object | None,
) -> Literal["exact", "same_work", "unspecified"]:
    if source_evidence is None or getattr(source_evidence, "same_edition", None) is None:
        return "unspecified"
    return "exact" if source_evidence.same_edition else "same_work"


def build_generation_grounding(
    spec: QuestionSpec,
    dataset: CanonicalDataset,
    *,
    canonical_file_hashes: dict[str, str],
    blueprint_hash: str,
) -> GenerationGrounding:
    """Resolve and bind one ML-selected source document without provider-specific logic."""

    if (
        spec.question_type != "comprehension"
        or spec.cognitive_operation != "apply"
        or spec.target_difficulty != 2
    ):
        raise GroundingError("generation-grounding-v1 supports only comprehension/apply/Level 2")
    if spec.related_concepts or spec.relationship_reasoning_required or spec.multi_step_required:
        raise GroundingError("generation-grounding-v1 supports only single-concept apply targets")
    if len(spec.source_document_ids) != 1:
        raise GroundingError("generation-grounding-v1 requires exactly one source document")

    source_document_id = spec.source_document_ids[0]
    documents = [
        document for document in dataset.documents if document.document_id == source_document_id
    ]
    if len(documents) != 1:
        raise GroundingError("QuestionSpec source document is missing or duplicated")
    document = documents[0]
    if document.document_type not in _PROSE_TYPES:
        raise GroundingError("QuestionSpec source document is not an allowed prose type")
    if document.book_id not in spec.supporting_book_ids:
        raise GroundingError("source document book does not match QuestionSpec supporting books")
    matching_evidence = [
        item
        for item in spec.supporting_evidence
        if item.evidence_id == source_document_id
        and item.book_id == document.book_id
        and item.concept_id == spec.primary_concept
        and item.evidence_type == document.document_type
    ]
    if not matching_evidence:
        raise GroundingError("source document lacks matching QuestionSpec concept evidence")
    if document.content_hash != _sha256_text(document.text):
        raise GroundingError("source document content hash does not match its text")

    sources = [source for source in dataset.sources if source.source_id == document.source_id]
    if len(sources) != 1:
        raise GroundingError("source document provenance is missing or duplicated")
    source = sources[0]
    if source.book_id != document.book_id:
        raise GroundingError("source provenance book does not match the source document")
    if source.license is None or not _has_approved_reuse_license(source.license):
        raise GroundingError("generation-grounding-v1 requires an approved explicit reuse license")
    if set(canonical_file_hashes) != _CANONICAL_FILES or any(
        _HASH_PATTERN.fullmatch(digest) is None for digest in canonical_file_hashes.values()
    ):
        raise GroundingError("canonical file hashes must cover all four JSONL inputs")
    if _HASH_PATTERN.fullmatch(blueprint_hash) is None:
        raise GroundingError("blueprint hash is malformed")

    passage = extract_passage(document.text, spec.primary_concept)
    return GenerationGrounding(
        question_spec_id=spec.question_id,
        question_spec_hash=_sha256_json(spec.model_dump(mode="json")),
        topic_id=spec.topic_id,
        question_type="comprehension",
        cognitive_operation="apply",
        target_difficulty=2,
        primary_concept=spec.primary_concept,
        related_concepts=spec.related_concepts,
        source_document_id=document.document_id,
        book_id=document.book_id,
        document_type=document.document_type,
        document_content_hash=document.content_hash,
        passage_text=passage,
        passage_hash=_sha256_text(passage),
        source_id=source.source_id,
        provider=source.provider,
        source_type=source.source_type,
        source_url=source.url,
        source_retrieved_at=source.retrieved_at,
        license=source.license,
        rights_note=source.rights_note,
        source_content_hash=source.content_hash,
        edition_relation=_edition_relation(source.evidence),
        canonical_file_hashes=dict(sorted(canonical_file_hashes.items())),
        blueprint_hash=blueprint_hash,
    )


def build_generation_grounding_v2(
    spec: QuestionSpec,
    dataset: CanonicalDataset,
    *,
    canonical_file_hashes: dict[str, str],
    blueprint_hash: str,
    display_normalization_policy: str = DISPLAY_NORMALIZATION_POLICY,
) -> GenerationGroundingV2:
    """Add a reviewed display passage without changing v1 extraction semantics."""

    source = build_generation_grounding(
        spec,
        dataset,
        canonical_file_hashes=canonical_file_hashes,
        blueprint_hash=blueprint_hash,
    )
    display_passage = normalize_display_passage(
        source.passage_text,
        source_document_id=source.source_document_id,
        source_passage_hash=source.passage_hash,
        policy=display_normalization_policy,
    )
    return GenerationGroundingV2(
        question_spec_id=source.question_spec_id,
        question_spec_hash=source.question_spec_hash,
        topic_id=source.topic_id,
        question_type=source.question_type,
        cognitive_operation=source.cognitive_operation,
        target_difficulty=source.target_difficulty,
        primary_concept=source.primary_concept,
        related_concepts=source.related_concepts,
        source_document_id=source.source_document_id,
        book_id=source.book_id,
        document_type=source.document_type,
        document_content_hash=source.document_content_hash,
        source_passage_text=source.passage_text,
        source_passage_hash=source.passage_hash,
        source_passage_extraction_policy=source.passage_extraction_policy,
        display_passage_text=display_passage,
        display_passage_hash=_sha256_text(display_passage),
        display_normalization_policy=display_normalization_policy,
        source_id=source.source_id,
        provider=source.provider,
        source_type=source.source_type,
        source_url=source.source_url,
        source_retrieved_at=source.source_retrieved_at,
        license=source.license,
        rights_note=source.rights_note,
        source_content_hash=source.source_content_hash,
        edition_relation=source.edition_relation,
        canonical_file_hashes=source.canonical_file_hashes,
        blueprint_hash=source.blueprint_hash,
    )
