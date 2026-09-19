"""Deterministic human gold review and evaluation for TOC concept mappings."""

import hashlib
from collections import defaultdict
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

from bookmatch_ml.concept_v2.graph import LoadedConceptGraph
from bookmatch_ml.concept_v2.profile import (
    LoadedConceptMatchingConfig,
    build_book_concept_profile_v2,
)
from bookmatch_ml.concept_v2.toc import reconstruct_toc
from bookmatch_ml.config import ConfigError, LoadedFeatureConfig
from bookmatch_ml.schemas import BookEvidence

SAMPLE_VERSION = "toc-concept-gold-sample-v1"
REVIEW_VERSION = "toc-concept-gold-review-v1"
EVALUATION_VERSION = "toc-concept-gold-evaluation-v1"
DEFAULT_SAMPLE_SIZES = {"linear-algebra": 50, "operating-systems": 50}


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class PredictionMatch(_Strict):
    concept_id: str
    matching_alias: str
    match_method: str


class TocConceptGoldReviewRow(_Strict):
    topic: str
    book_id: str
    book_title: str
    toc_entry_id: str
    toc_title: str
    toc_path: list[str]
    source_id: str
    traversal_position: int = Field(ge=0)
    predicted_concept_ids: list[str]
    prediction_matches: list[PredictionMatch]
    human_gold_concept_ids: list[str]
    review_status: Literal["unreviewed", "reviewed"]
    review_note: str | None = None

    @field_validator("predicted_concept_ids", "human_gold_concept_ids")
    @classmethod
    def concept_sets_are_sorted_and_unique(cls, value: list[str]) -> list[str]:
        if value != sorted(set(value)):
            raise ValueError("concept ID lists must be sorted and unique")
        return value

    @model_validator(mode="after")
    def review_fields_are_consistent(self) -> "TocConceptGoldReviewRow":
        match_ids = sorted({match.concept_id for match in self.prediction_matches})
        if match_ids != self.predicted_concept_ids:
            raise ValueError("prediction matches disagree with predicted concept IDs")
        if self.review_status == "unreviewed" and self.human_gold_concept_ids:
            raise ValueError("unreviewed rows must not contain human gold concepts")
        if self.review_note is not None and not self.review_note.strip():
            raise ValueError("review_note must be null or non-blank")
        return self


class TocConceptGoldReview(_Strict):
    review_version: Literal["toc-concept-gold-review-v1"]
    sample_version: Literal["toc-concept-gold-sample-v1"]
    sample_size_by_topic: dict[str, int]
    canonical_input_hashes: dict[str, str]
    feature_config_hash: str
    graph_hash: str
    graph_version: str
    matching_config_hash: str
    matching_config_version: str
    entries: list[TocConceptGoldReviewRow]

    @model_validator(mode="after")
    def entries_are_unique_and_counts_match(self) -> "TocConceptGoldReview":
        keys = [(row.topic, row.book_id, row.toc_entry_id) for row in self.entries]
        if len(keys) != len(set(keys)):
            raise ValueError("duplicate TOC gold review row")
        if any(size <= 0 for size in self.sample_size_by_topic.values()):
            raise ValueError("sample sizes must be positive")
        counts: dict[str, int] = defaultdict(int)
        for row in self.entries:
            counts[row.topic] += 1
        if dict(sorted(counts.items())) != dict(sorted(self.sample_size_by_topic.items())):
            raise ValueError("review entries do not match sample_size_by_topic")
        return self


class LoadedTocConceptGoldReview(_Strict):
    review: TocConceptGoldReview
    content_hash: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")


class ConceptAssignmentError(_Strict):
    topic: str
    book_id: str
    toc_entry_id: str
    concept_id: str


class TocConceptGoldMetrics(_Strict):
    scope: str
    reviewed_entry_count: int = Field(ge=0)
    exact_set_match_count: int = Field(ge=0)
    exact_set_match_accuracy: float = Field(ge=0, le=1)
    true_positive_assignment_count: int = Field(ge=0)
    false_positive_assignment_count: int = Field(ge=0)
    false_negative_assignment_count: int = Field(ge=0)
    micro_precision: float = Field(ge=0, le=1)
    micro_recall: float = Field(ge=0, le=1)
    micro_f1: float = Field(ge=0, le=1)
    predicted_unmatched_entry_count: int = Field(ge=0)
    human_unmatched_entry_count: int = Field(ge=0)
    false_positive_assignments: list[ConceptAssignmentError]
    false_negative_assignments: list[ConceptAssignmentError]


class TocConceptGoldEvaluationReport(_Strict):
    evaluation_version: Literal["toc-concept-gold-evaluation-v1"]
    sample_version: str
    canonical_input_hashes: dict[str, str]
    feature_config_hash: str
    graph_hash: str
    graph_version: str
    matching_config_hash: str
    matching_config_version: str
    review_file_hash: str
    review_version: str
    topics: list[TocConceptGoldMetrics]
    combined: TocConceptGoldMetrics


def _selection_key(row: TocConceptGoldReviewRow) -> tuple[str, str]:
    payload = "\0".join((SAMPLE_VERSION, row.topic, row.book_id, row.toc_entry_id)).encode()
    return hashlib.sha256(payload).hexdigest(), row.toc_entry_id


def _allocate_sample(
    rows: list[TocConceptGoldReviewRow], sample_size: int
) -> list[TocConceptGoldReviewRow]:
    by_book: dict[str, list[TocConceptGoldReviewRow]] = defaultdict(list)
    for row in rows:
        by_book[row.book_id].append(row)
    if len(rows) < sample_size:
        raise ValueError(f"sample size {sample_size} exceeds TOC population {len(rows)}")
    book_ids = sorted(by_book)
    if not book_ids:
        raise ValueError("cannot sample a topic without books")
    base, remainder = divmod(sample_size, len(book_ids))
    allocations = {
        book_id: min(len(by_book[book_id]), base + (index < remainder))
        for index, book_id in enumerate(book_ids)
    }
    remaining = sample_size - sum(allocations.values())
    while remaining:
        progressed = False
        for book_id in book_ids:
            if allocations[book_id] < len(by_book[book_id]):
                allocations[book_id] += 1
                remaining -= 1
                progressed = True
                if remaining == 0:
                    break
        if not progressed:
            raise ValueError("unable to allocate deterministic TOC sample")
    selected = [
        row
        for book_id in book_ids
        for row in sorted(by_book[book_id], key=_selection_key)[: allocations[book_id]]
    ]
    return sorted(
        selected,
        key=lambda row: (row.topic, row.book_id, row.traversal_position, row.toc_entry_id),
    )


def build_toc_concept_gold_review(
    evidence: list[BookEvidence],
    graph: LoadedConceptGraph,
    features: LoadedFeatureConfig,
    matching: LoadedConceptMatchingConfig,
    toc_file_hash: str,
    canonical_input_hashes: dict[str, str],
    sample_size_by_topic: dict[str, int] | None = None,
) -> TocConceptGoldReview:
    """Build an unlabeled, deterministic sample from every topic's full TOC population."""

    sample_sizes = dict(sorted((sample_size_by_topic or DEFAULT_SAMPLE_SIZES).items()))
    unknown_topics = set(sample_sizes) - set(graph.graph.nodes)
    if unknown_topics:
        raise ValueError(f"sample references unsupported topics: {sorted(unknown_topics)}")
    population: dict[str, list[TocConceptGoldReviewRow]] = defaultdict(list)
    for book in sorted(evidence, key=lambda item: item.book_id):
        for topic in sorted(set(book.metadata.topics) & set(sample_sizes)):
            profile = build_book_concept_profile_v2(
                book, topic, features, graph, matching, toc_file_hash
            )
            predictions: dict[str, list[PredictionMatch]] = defaultdict(list)
            for mapping in profile.toc_mappings:
                predictions[mapping.toc_entry_id].append(
                    PredictionMatch(
                        concept_id=mapping.concept_id,
                        matching_alias=mapping.matching_alias,
                        match_method=mapping.match_method,
                    )
                )
            for visit in reconstruct_toc(book.toc).traversal:
                matches = sorted(
                    predictions[visit.entry.toc_entry_id],
                    key=lambda item: (item.concept_id, item.matching_alias, item.match_method),
                )
                population[topic].append(
                    TocConceptGoldReviewRow(
                        topic=topic,
                        book_id=book.book_id,
                        book_title=book.metadata.title,
                        toc_entry_id=visit.entry.toc_entry_id,
                        toc_title=visit.entry.title,
                        toc_path=list(visit.path),
                        source_id=visit.entry.source_id,
                        traversal_position=visit.traversal_position,
                        predicted_concept_ids=sorted({match.concept_id for match in matches}),
                        prediction_matches=matches,
                        human_gold_concept_ids=[],
                        review_status="unreviewed",
                    )
                )
    entries = [
        row
        for topic, size in sample_sizes.items()
        for row in _allocate_sample(population[topic], size)
    ]
    return TocConceptGoldReview(
        review_version=REVIEW_VERSION,
        sample_version=SAMPLE_VERSION,
        sample_size_by_topic=sample_sizes,
        canonical_input_hashes=dict(sorted(canonical_input_hashes.items())),
        feature_config_hash=features.content_hash,
        graph_hash=graph.content_hash,
        graph_version=graph.graph.graph_version,
        matching_config_hash=matching.content_hash,
        matching_config_version=matching.config.config_version,
        entries=entries,
    )


def load_toc_concept_gold_review(
    path: Path,
    expected: TocConceptGoldReview,
    graph: LoadedConceptGraph,
) -> LoadedTocConceptGoldReview:
    """Load human labels while verifying all sampled identities and predictions."""

    try:
        content = path.read_bytes()
        review = TocConceptGoldReview.model_validate_json(content)
        expected_by_id = {
            (row.topic, row.book_id, row.toc_entry_id): row for row in expected.entries
        }
        actual_by_id = {(row.topic, row.book_id, row.toc_entry_id): row for row in review.entries}
        unknown = set(actual_by_id) - set(expected_by_id)
        missing = set(expected_by_id) - set(actual_by_id)
        if unknown:
            raise ValueError(f"review references unknown topic/book/TOC entries: {sorted(unknown)}")
        if missing:
            raise ValueError(f"review is missing sampled TOC entries: {sorted(missing)}")
        metadata_fields = (
            "topic",
            "book_id",
            "book_title",
            "toc_entry_id",
            "toc_title",
            "toc_path",
            "source_id",
            "traversal_position",
            "predicted_concept_ids",
            "prediction_matches",
        )
        for key, actual in actual_by_id.items():
            reference = expected_by_id[key]
            if any(
                getattr(actual, field) != getattr(reference, field) for field in metadata_fields
            ):
                raise ValueError(f"review identity or prediction differs for {key}")
            allowed = set(graph.graph.nodes[actual.topic])
            unknown_concepts = set(actual.human_gold_concept_ids) - allowed
            if unknown_concepts:
                raise ValueError(
                    f"review references unknown {actual.topic} concepts: {sorted(unknown_concepts)}"
                )
        header_fields = (
            "review_version",
            "sample_version",
            "sample_size_by_topic",
            "canonical_input_hashes",
            "feature_config_hash",
            "graph_hash",
            "graph_version",
            "matching_config_hash",
            "matching_config_version",
        )
        if any(getattr(review, field) != getattr(expected, field) for field in header_fields):
            raise ValueError("review provenance differs from the current deterministic sample")
    except (OSError, ValidationError, ValueError) as exc:
        raise ConfigError(f"invalid TOC concept gold review: {path}: {exc}") from exc
    return LoadedTocConceptGoldReview(
        review=review, content_hash=f"sha256:{hashlib.sha256(content).hexdigest()}"
    )


def calculate_gold_metrics(
    rows: list[TocConceptGoldReviewRow], scope: str
) -> TocConceptGoldMetrics:
    """Calculate deterministic multilabel micro metrics over reviewed entry-level sets."""

    false_positives: list[ConceptAssignmentError] = []
    false_negatives: list[ConceptAssignmentError] = []
    true_positives = 0
    exact = 0
    for row in rows:
        predicted = set(row.predicted_concept_ids)
        gold = set(row.human_gold_concept_ids)
        true_positives += len(predicted & gold)
        exact += predicted == gold
        false_positives.extend(
            ConceptAssignmentError(
                topic=row.topic,
                book_id=row.book_id,
                toc_entry_id=row.toc_entry_id,
                concept_id=concept,
            )
            for concept in sorted(predicted - gold)
        )
        false_negatives.extend(
            ConceptAssignmentError(
                topic=row.topic,
                book_id=row.book_id,
                toc_entry_id=row.toc_entry_id,
                concept_id=concept,
            )
            for concept in sorted(gold - predicted)
        )
    count = len(rows)
    fp = len(false_positives)
    fn = len(false_negatives)
    precision = true_positives / (true_positives + fp) if true_positives + fp else 0.0
    recall = true_positives / (true_positives + fn) if true_positives + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return TocConceptGoldMetrics(
        scope=scope,
        reviewed_entry_count=count,
        exact_set_match_count=exact,
        exact_set_match_accuracy=exact / count if count else 0.0,
        true_positive_assignment_count=true_positives,
        false_positive_assignment_count=fp,
        false_negative_assignment_count=fn,
        micro_precision=precision,
        micro_recall=recall,
        micro_f1=f1,
        predicted_unmatched_entry_count=sum(not row.predicted_concept_ids for row in rows),
        human_unmatched_entry_count=sum(not row.human_gold_concept_ids for row in rows),
        false_positive_assignments=false_positives,
        false_negative_assignments=false_negatives,
    )


def evaluate_toc_concept_gold(
    loaded: LoadedTocConceptGoldReview,
    graph: LoadedConceptGraph,
) -> TocConceptGoldEvaluationReport:
    """Evaluate current stored predictions after every sampled row has human labels."""

    pending = [
        (row.book_id, row.toc_entry_id)
        for row in loaded.review.entries
        if row.review_status != "reviewed"
    ]
    if pending:
        raise ValueError(f"gold evaluation requires complete reviews; {len(pending)} remain")
    allowed_by_topic = {topic: set(nodes) for topic, nodes in graph.graph.nodes.items()}
    for row in loaded.review.entries:
        unknown = set(row.human_gold_concept_ids) - allowed_by_topic[row.topic]
        if unknown:
            raise ValueError(f"review references unknown {row.topic} concepts: {sorted(unknown)}")
    topics = [
        calculate_gold_metrics([row for row in loaded.review.entries if row.topic == topic], topic)
        for topic in sorted(loaded.review.sample_size_by_topic)
    ]
    return TocConceptGoldEvaluationReport(
        evaluation_version=EVALUATION_VERSION,
        sample_version=loaded.review.sample_version,
        canonical_input_hashes=loaded.review.canonical_input_hashes,
        feature_config_hash=loaded.review.feature_config_hash,
        graph_hash=loaded.review.graph_hash,
        graph_version=loaded.review.graph_version,
        matching_config_hash=loaded.review.matching_config_hash,
        matching_config_version=loaded.review.matching_config_version,
        review_file_hash=loaded.content_hash,
        review_version=loaded.review.review_version,
        topics=topics,
        combined=calculate_gold_metrics(loaded.review.entries, "combined"),
    )
