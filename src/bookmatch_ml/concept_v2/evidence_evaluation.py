"""Human-review evaluation for source-aware book evidence concept mappings."""

import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

from bookmatch_ml.book.text import normalize_text
from bookmatch_ml.concept_v2.graph import LoadedConceptGraph
from bookmatch_ml.concept_v2.profile import (
    LoadedConceptMatchingConfig,
    match_concept_text,
)
from bookmatch_ml.config import ConfigError, LoadedFeatureConfig
from bookmatch_ml.data.book_evidence import EvidenceType, ImportedBookEvidence

SAMPLE_VERSION = "evidence-concept-gold-sample-v1"
REVIEW_VERSION = "evidence-concept-gold-review-v1"
EVALUATION_VERSION = "evidence-concept-gold-evaluation-v1"
SUPPORTED_EVIDENCE_TYPES: tuple[EvidenceType, ...] = (
    "toc_exact",
    "toc_public_web_exact",
    "toc_same_work",
    "description",
    "subject",
    "metadata_minimal",
)
TOC_EVIDENCE_TYPES = frozenset({"toc_exact", "toc_public_web_exact", "toc_same_work"})
METADATA_EVIDENCE_TYPES = frozenset({"description", "subject", "metadata_minimal"})
DEFAULT_SAMPLE_SIZES: dict[str, int] = {
    "toc_same_work": 6,
    "toc_exact": 16,
    "toc_public_web_exact": 16,
    "description": 16,
    "subject": 20,
    "metadata_minimal": 20,
}
DEFAULT_PER_BOOK_CAP = 8
DEFAULT_GENERIC_TOC_CAP = 2
GENERIC_TOC_TITLES = frozenset(
    {
        "appendix",
        "bibliography",
        "exercises",
        "further reading",
        "introduction",
        "notes",
        "preface",
        "problems",
        "references",
        "review questions",
        "summary",
    }
)


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class EvidencePredictionMatch(_Strict):
    concept_id: str
    matching_alias: str
    match_method: Literal["normalized_alias_phrase_v1"]


class EvidenceConceptReviewRow(_Strict):
    topic: str
    book_id: str
    book_title: str
    evidence_id: str
    evidence_type: EvidenceType
    evidence_text: str
    toc_path: list[str] | None = None
    source_id: str
    provider: str
    edition_relation: str
    source_evidence_tier: str | None = None
    predicted_concept_ids: list[str]
    prediction_matches: list[EvidencePredictionMatch]
    prediction_status: Literal["matched", "unmatched", "ambiguous_alias"]
    human_gold_concept_ids: list[str]
    review_status: Literal["unreviewed", "reviewed"]
    review_outcome: Literal["pending", "labeled", "no_concept", "not_judgable"]
    review_note: str | None = None

    @field_validator("predicted_concept_ids", "human_gold_concept_ids")
    @classmethod
    def concept_sets_are_sorted_and_unique(cls, value: list[str]) -> list[str]:
        if value != sorted(set(value)):
            raise ValueError("concept ID lists must be sorted and unique")
        return value

    @model_validator(mode="after")
    def review_and_prediction_fields_are_consistent(self) -> "EvidenceConceptReviewRow":
        match_ids = sorted({match.concept_id for match in self.prediction_matches})
        if match_ids != self.predicted_concept_ids:
            raise ValueError("prediction matches disagree with predicted concept IDs")
        if self.prediction_status == "matched" and not self.predicted_concept_ids:
            raise ValueError("matched predictions require a concept")
        if self.prediction_status != "matched" and self.predicted_concept_ids:
            raise ValueError("unmatched or ambiguous predictions cannot contain concepts")
        if self.review_status == "unreviewed":
            if self.review_outcome != "pending" or self.human_gold_concept_ids:
                raise ValueError("unreviewed rows must be pending without human gold")
        elif self.review_outcome == "pending":
            raise ValueError("reviewed rows cannot remain pending")
        if self.review_outcome == "labeled" and not self.human_gold_concept_ids:
            raise ValueError("labeled rows require human gold concepts")
        if self.review_outcome in {"no_concept", "not_judgable"} and (self.human_gold_concept_ids):
            raise ValueError(f"{self.review_outcome} rows cannot contain human gold concepts")
        if self.evidence_type in TOC_EVIDENCE_TYPES and not self.toc_path:
            raise ValueError("TOC review rows require a TOC path")
        if self.evidence_type not in TOC_EVIDENCE_TYPES and self.toc_path is not None:
            raise ValueError("metadata review rows cannot contain a TOC path")
        if self.review_note is not None and not self.review_note.strip():
            raise ValueError("review_note must be null or non-blank")
        return self


class EvidenceConceptGoldReview(_Strict):
    review_version: Literal["evidence-concept-gold-review-v1"]
    sample_version: Literal["evidence-concept-gold-sample-v1"]
    book_evidence_contract_version: Literal["book-evidence-v1"]
    book_evidence_hash: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    feature_config_hash: str
    graph_hash: str
    graph_version: str
    matching_config_hash: str
    matching_config_version: str
    requested_sample_size_by_evidence_type: dict[str, int]
    per_book_cap: int = Field(gt=0)
    generic_toc_cap_by_evidence_type: int = Field(ge=0)
    entries: list[EvidenceConceptReviewRow]

    @model_validator(mode="after")
    def entries_are_unique_and_within_caps(self) -> "EvidenceConceptGoldReview":
        keys = [(row.topic, row.book_id, row.evidence_id) for row in self.entries]
        if len(keys) != len(set(keys)):
            raise ValueError("duplicate evidence concept review row")
        counts = Counter(row.book_id for row in self.entries)
        if counts and max(counts.values()) > self.per_book_cap:
            raise ValueError("evidence review exceeds per-book sampling cap")
        unknown = set(self.requested_sample_size_by_evidence_type) - set(SUPPORTED_EVIDENCE_TYPES)
        if unknown:
            raise ValueError(f"unsupported evidence sample groups: {sorted(unknown)}")
        if any(size <= 0 for size in self.requested_sample_size_by_evidence_type.values()):
            raise ValueError("requested sample sizes must be positive")
        return self


class LoadedEvidenceConceptGoldReview(_Strict):
    review: EvidenceConceptGoldReview
    content_hash: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")


class EvidenceMetric(_Strict):
    scope: str
    entry_count: int = Field(ge=1)
    book_count: int = Field(ge=1)
    exact_set_match_rate: float = Field(ge=0, le=1)
    micro_precision: float = Field(ge=0, le=1)
    micro_recall: float = Field(ge=0, le=1)
    micro_f1: float = Field(ge=0, le=1)
    macro_precision: float = Field(ge=0, le=1)
    macro_recall: float = Field(ge=0, le=1)
    macro_f1: float = Field(ge=0, le=1)
    book_macro_precision: float = Field(ge=0, le=1)
    book_macro_recall: float = Field(ge=0, le=1)
    book_macro_f1: float = Field(ge=0, le=1)
    true_positive_count: int = Field(ge=0)
    false_positive_count: int = Field(ge=0)
    false_negative_count: int = Field(ge=0)


class PolicyMetric(_Strict):
    policy: Literal["toc_only", "metadata_only", "combined_unweighted"]
    book_count: int = Field(ge=1)
    exact_set_match_rate: float = Field(ge=0, le=1)
    micro_precision: float = Field(ge=0, le=1)
    micro_recall: float = Field(ge=0, le=1)
    micro_f1: float = Field(ge=0, le=1)
    book_macro_precision: float = Field(ge=0, le=1)
    book_macro_recall: float = Field(ge=0, le=1)
    book_macro_f1: float = Field(ge=0, le=1)
    unique_book_concept_assignment_count: int = Field(ge=0)
    raw_prediction_occurrence_count: int = Field(ge=0)


class EvidenceConceptEvaluationReport(_Strict):
    evaluation_version: Literal["evidence-concept-gold-evaluation-v1"]
    review_version: str
    sample_version: str
    review_file_hash: str
    total_entry_count: int = Field(ge=0)
    reviewed_entry_count: int = Field(ge=0)
    evaluable_entry_count: int = Field(ge=0)
    not_judgable_entry_count: int = Field(ge=0)
    remaining_entry_count: int = Field(ge=0)
    metric_status: Literal["unavailable", "partial", "complete"]
    metrics: list[EvidenceMetric]
    policy_metrics: list[PolicyMetric]


def _selection_key(row: EvidenceConceptReviewRow) -> tuple[str, str]:
    payload = "\0".join(
        (SAMPLE_VERSION, row.evidence_type, row.topic, row.book_id, row.evidence_id)
    ).encode()
    return hashlib.sha256(payload).hexdigest(), row.evidence_id


def _is_generic_toc(row: EvidenceConceptReviewRow) -> bool:
    if row.evidence_type not in TOC_EVIDENCE_TYPES:
        return False
    normalized = normalize_text(row.evidence_text)
    return normalized in GENERIC_TOC_TITLES or any(
        normalized.startswith(f"{title} ") for title in GENERIC_TOC_TITLES
    )


def _deduplicate_candidates(
    rows: list[EvidenceConceptReviewRow],
) -> list[EvidenceConceptReviewRow]:
    selected: dict[tuple[str, str, str, str], EvidenceConceptReviewRow] = {}
    for row in sorted(rows, key=_selection_key):
        key = (row.topic, row.book_id, row.evidence_type, normalize_text(row.evidence_text))
        selected.setdefault(key, row)
    return list(selected.values())


def _sample_group(
    rows: list[EvidenceConceptReviewRow],
    size: int,
    book_counts: Counter[str],
    per_book_cap: int,
    generic_cap: int,
) -> list[EvidenceConceptReviewRow]:
    by_topic_book: dict[str, dict[str, list[EvidenceConceptReviewRow]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for row in _deduplicate_candidates(rows):
        by_topic_book[row.topic][row.book_id].append(row)
    for books in by_topic_book.values():
        for book_rows in books.values():
            book_rows.sort(key=_selection_key, reverse=True)

    selected: list[EvidenceConceptReviewRow] = []
    generic_count = 0
    topics = sorted(by_topic_book)
    while len(selected) < size:
        progressed = False
        for topic in topics:
            books = by_topic_book[topic]
            selected_for_topic = False
            for book_id in sorted(
                books,
                key=lambda value: (
                    book_counts[value],
                    hashlib.sha256(f"{SAMPLE_VERSION}\0{topic}\0{value}".encode()).hexdigest(),
                    value,
                ),
            ):
                if book_counts[book_id] >= per_book_cap:
                    continue
                candidates = books[book_id]
                while candidates:
                    candidate = candidates.pop()
                    if _is_generic_toc(candidate) and generic_count >= generic_cap:
                        continue
                    selected.append(candidate)
                    book_counts[book_id] += 1
                    generic_count += _is_generic_toc(candidate)
                    progressed = True
                    selected_for_topic = True
                    break
                if selected_for_topic:
                    break
            if len(selected) >= size:
                break
        if not progressed:
            break
    return selected


def build_evidence_concept_gold_review(
    records: list[ImportedBookEvidence],
    graph: LoadedConceptGraph,
    features: LoadedFeatureConfig,
    matching: LoadedConceptMatchingConfig,
    book_evidence_hash: str,
    sample_size_by_evidence_type: dict[str, int] | None = None,
    per_book_cap: int = DEFAULT_PER_BOOK_CAP,
    generic_toc_cap: int = DEFAULT_GENERIC_TOC_CAP,
) -> EvidenceConceptGoldReview:
    """Build a deterministic source-stratified blank human-review template."""

    sample_sizes = dict(sample_size_by_evidence_type or DEFAULT_SAMPLE_SIZES)
    population: dict[str, list[EvidenceConceptReviewRow]] = defaultdict(list)
    for record in sorted(records, key=lambda item: item.book.book_id):
        topics = sorted(set(record.book.topics) & set(graph.graph.nodes))
        for topic in topics:
            for item in sorted(record.evidence, key=lambda evidence: evidence.evidence_id):
                if item.evidence_type not in sample_sizes:
                    continue
                matches, ambiguous = match_concept_text(item.text, topic, features, graph, matching)
                prediction_matches = [
                    EvidencePredictionMatch(
                        concept_id=match.concept_id,
                        matching_alias=match.matching_alias,
                        match_method=match.match_method,
                    )
                    for match in matches
                ]
                population[item.evidence_type].append(
                    EvidenceConceptReviewRow(
                        topic=topic,
                        book_id=record.book.book_id,
                        book_title=record.book.title,
                        evidence_id=item.evidence_id,
                        evidence_type=item.evidence_type,
                        evidence_text=item.text,
                        toc_path=item.toc_path,
                        source_id=item.source_id,
                        provider=item.provider,
                        edition_relation=item.edition_relation,
                        source_evidence_tier=item.source_evidence_tier,
                        predicted_concept_ids=sorted(match.concept_id for match in matches),
                        prediction_matches=prediction_matches,
                        prediction_status=(
                            "ambiguous_alias"
                            if ambiguous
                            else "matched"
                            if matches
                            else "unmatched"
                        ),
                        human_gold_concept_ids=[],
                        review_status="unreviewed",
                        review_outcome="pending",
                    )
                )

    book_counts: Counter[str] = Counter()
    entries: list[EvidenceConceptReviewRow] = []
    for evidence_type, requested in sample_sizes.items():
        entries.extend(
            _sample_group(
                population[evidence_type],
                requested,
                book_counts,
                per_book_cap,
                generic_toc_cap,
            )
        )
    entries.sort(key=lambda row: (row.topic, row.book_id, row.evidence_type, row.evidence_id))
    return EvidenceConceptGoldReview(
        review_version=REVIEW_VERSION,
        sample_version=SAMPLE_VERSION,
        book_evidence_contract_version="book-evidence-v1",
        book_evidence_hash=book_evidence_hash,
        feature_config_hash=features.content_hash,
        graph_hash=graph.content_hash,
        graph_version=graph.graph.graph_version,
        matching_config_hash=matching.content_hash,
        matching_config_version=matching.config.config_version,
        requested_sample_size_by_evidence_type=sample_sizes,
        per_book_cap=per_book_cap,
        generic_toc_cap_by_evidence_type=generic_toc_cap,
        entries=entries,
    )


def load_evidence_concept_gold_review(
    path: Path,
    expected: EvidenceConceptGoldReview,
    graph: LoadedConceptGraph,
) -> LoadedEvidenceConceptGoldReview:
    """Load human edits while pinning all generated identities and predictions."""

    try:
        content = path.read_bytes()
        review = EvidenceConceptGoldReview.model_validate_json(content)
        expected_by_key = {
            (row.topic, row.book_id, row.evidence_id): row for row in expected.entries
        }
        actual_by_key = {(row.topic, row.book_id, row.evidence_id): row for row in review.entries}
        if set(actual_by_key) != set(expected_by_key):
            missing = sorted(set(expected_by_key) - set(actual_by_key))
            unknown = sorted(set(actual_by_key) - set(expected_by_key))
            raise ValueError(f"review sample differs; missing={missing}, unknown={unknown}")
        editable = {
            "human_gold_concept_ids",
            "review_status",
            "review_outcome",
            "review_note",
        }
        for key, actual in actual_by_key.items():
            reference = expected_by_key[key]
            actual_fixed = actual.model_dump(exclude=editable)
            reference_fixed = reference.model_dump(exclude=editable)
            if actual_fixed != reference_fixed:
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
            "book_evidence_contract_version",
            "book_evidence_hash",
            "feature_config_hash",
            "graph_hash",
            "graph_version",
            "matching_config_hash",
            "matching_config_version",
            "requested_sample_size_by_evidence_type",
            "per_book_cap",
            "generic_toc_cap_by_evidence_type",
        )
        if any(getattr(review, field) != getattr(expected, field) for field in header_fields):
            raise ValueError("review provenance differs from the deterministic sample")
    except (OSError, ValidationError, ValueError) as exc:
        raise ConfigError(f"invalid evidence concept gold review: {path}: {exc}") from exc
    return LoadedEvidenceConceptGoldReview(
        review=review,
        content_hash=f"sha256:{hashlib.sha256(content).hexdigest()}",
    )


def _prf(predicted: set[str], gold: set[str]) -> tuple[int, int, int, float, float, float]:
    tp = len(predicted & gold)
    fp = len(predicted - gold)
    fn = len(gold - predicted)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return tp, fp, fn, precision, recall, f1


def calculate_evidence_metrics(rows: list[EvidenceConceptReviewRow], scope: str) -> EvidenceMetric:
    """Calculate row micro/macro metrics and equal-weight book macro metrics."""

    if not rows:
        raise ValueError("cannot calculate evidence metrics without reviewed rows")
    totals = [0, 0, 0]
    row_values: list[tuple[float, float, float]] = []
    exact = 0
    by_book: dict[str, tuple[set[str], set[str]]] = {}
    for row in rows:
        predicted = set(row.predicted_concept_ids)
        gold = set(row.human_gold_concept_ids)
        tp, fp, fn, precision, recall, f1 = _prf(predicted, gold)
        totals[0] += tp
        totals[1] += fp
        totals[2] += fn
        row_values.append((precision, recall, f1))
        exact += predicted == gold
        book_predicted, book_gold = by_book.setdefault(row.book_id, (set(), set()))
        book_predicted.update(predicted)
        book_gold.update(gold)
    tp, fp, fn = totals
    micro_precision = tp / (tp + fp) if tp + fp else 0.0
    micro_recall = tp / (tp + fn) if tp + fn else 0.0
    micro_f1 = (
        2 * micro_precision * micro_recall / (micro_precision + micro_recall)
        if micro_precision + micro_recall
        else 0.0
    )
    book_values = [_prf(predicted, gold)[3:] for predicted, gold in by_book.values()]
    return EvidenceMetric(
        scope=scope,
        entry_count=len(rows),
        book_count=len(by_book),
        exact_set_match_rate=exact / len(rows),
        micro_precision=micro_precision,
        micro_recall=micro_recall,
        micro_f1=micro_f1,
        macro_precision=sum(value[0] for value in row_values) / len(row_values),
        macro_recall=sum(value[1] for value in row_values) / len(row_values),
        macro_f1=sum(value[2] for value in row_values) / len(row_values),
        book_macro_precision=sum(value[0] for value in book_values) / len(book_values),
        book_macro_recall=sum(value[1] for value in book_values) / len(book_values),
        book_macro_f1=sum(value[2] for value in book_values) / len(book_values),
        true_positive_count=tp,
        false_positive_count=fp,
        false_negative_count=fn,
    )


def _policy_metric(
    rows: list[EvidenceConceptReviewRow],
    policy: Literal["toc_only", "metadata_only", "combined_unweighted"],
) -> PolicyMetric:
    allowed = {
        "toc_only": TOC_EVIDENCE_TYPES,
        "metadata_only": METADATA_EVIDENCE_TYPES,
        "combined_unweighted": TOC_EVIDENCE_TYPES | METADATA_EVIDENCE_TYPES,
    }[policy]
    by_book_gold: dict[str, set[str]] = defaultdict(set)
    by_book_predicted: dict[str, set[str]] = defaultdict(set)
    eligible_books: set[str] = set()
    raw_occurrences = 0
    for row in rows:
        by_book_gold[row.book_id].update(row.human_gold_concept_ids)
        if row.evidence_type in allowed:
            eligible_books.add(row.book_id)
            by_book_predicted[row.book_id].update(row.predicted_concept_ids)
            raw_occurrences += len(row.predicted_concept_ids)
    values = [
        _prf(by_book_predicted[book_id], by_book_gold[book_id])
        for book_id in sorted(eligible_books)
    ]
    if not values:
        raise ValueError(f"no reviewed books are eligible for policy {policy}")
    tp = sum(value[0] for value in values)
    fp = sum(value[1] for value in values)
    fn = sum(value[2] for value in values)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return PolicyMetric(
        policy=policy,
        book_count=len(values),
        exact_set_match_rate=sum(value[1] == 0 and value[2] == 0 for value in values) / len(values),
        micro_precision=precision,
        micro_recall=recall,
        micro_f1=f1,
        book_macro_precision=sum(value[3] for value in values) / len(values),
        book_macro_recall=sum(value[4] for value in values) / len(values),
        book_macro_f1=sum(value[5] for value in values) / len(values),
        unique_book_concept_assignment_count=sum(
            len(by_book_predicted[book_id]) for book_id in eligible_books
        ),
        raw_prediction_occurrence_count=raw_occurrences,
    )


def evaluate_evidence_concept_gold(
    loaded: LoadedEvidenceConceptGoldReview,
) -> EvidenceConceptEvaluationReport:
    """Evaluate reviewed and judgeable rows; never treats pending rows as gold."""

    all_rows = loaded.review.entries
    reviewed = [row for row in all_rows if row.review_status == "reviewed"]
    evaluable = [row for row in reviewed if row.review_outcome != "not_judgable"]
    status: Literal["unavailable", "partial", "complete"]
    if not evaluable:
        status = "unavailable"
    elif len(reviewed) < len(all_rows):
        status = "partial"
    else:
        status = "complete"
    metrics: list[EvidenceMetric] = []
    if evaluable:
        scopes: list[tuple[str, list[EvidenceConceptReviewRow]]] = [("all", evaluable)]
        families = (("toc", TOC_EVIDENCE_TYPES), ("metadata", METADATA_EVIDENCE_TYPES))
        scopes.extend(
            (f"topic:{topic}", [row for row in evaluable if row.topic == topic])
            for topic in sorted({row.topic for row in evaluable})
        )
        scopes.extend(
            (
                f"evidence_type:{evidence_type}",
                [row for row in evaluable if row.evidence_type == evidence_type],
            )
            for evidence_type in SUPPORTED_EVIDENCE_TYPES
            if any(row.evidence_type == evidence_type for row in evaluable)
        )
        scopes.extend(
            (
                f"family:{family}",
                [row for row in evaluable if row.evidence_type in types],
            )
            for family, types in families
            if any(row.evidence_type in types for row in evaluable)
        )
        metrics = [calculate_evidence_metrics(rows, scope) for scope, rows in scopes]
    policy_metrics = []
    if evaluable and status == "complete":
        for policy in ("toc_only", "metadata_only", "combined_unweighted"):
            policy_types = {
                "toc_only": TOC_EVIDENCE_TYPES,
                "metadata_only": METADATA_EVIDENCE_TYPES,
                "combined_unweighted": TOC_EVIDENCE_TYPES | METADATA_EVIDENCE_TYPES,
            }[policy]
            if any(row.evidence_type in policy_types for row in evaluable):
                policy_metrics.append(_policy_metric(evaluable, policy))
    return EvidenceConceptEvaluationReport(
        evaluation_version=EVALUATION_VERSION,
        review_version=loaded.review.review_version,
        sample_version=loaded.review.sample_version,
        review_file_hash=loaded.content_hash,
        total_entry_count=len(all_rows),
        reviewed_entry_count=len(reviewed),
        evaluable_entry_count=len(evaluable),
        not_judgable_entry_count=sum(row.review_outcome == "not_judgable" for row in reviewed),
        remaining_entry_count=len(all_rows) - len(reviewed),
        metric_status=status,
        metrics=metrics,
        policy_metrics=policy_metrics,
    )


def update_review_entry(
    review: EvidenceConceptGoldReview,
    graph: LoadedConceptGraph,
    evidence_id: str,
    outcome: Literal["labeled", "no_concept", "not_judgable"],
    gold_concept_ids: list[str],
    note: str | None,
) -> EvidenceConceptGoldReview:
    """Return a validated review with exactly one explicit human decision applied."""

    matches = [index for index, row in enumerate(review.entries) if row.evidence_id == evidence_id]
    if len(matches) != 1:
        raise ValueError(f"expected exactly one review row for evidence ID {evidence_id}")
    index = matches[0]
    row = review.entries[index]
    gold = sorted(set(gold_concept_ids))
    unknown = set(gold) - set(graph.graph.nodes[row.topic])
    if unknown:
        raise ValueError(f"unknown {row.topic} concepts: {sorted(unknown)}")
    updated = row.model_copy(
        update={
            "human_gold_concept_ids": gold,
            "review_status": "reviewed",
            "review_outcome": outcome,
            "review_note": note,
        }
    )
    updated = EvidenceConceptReviewRow.model_validate(updated.model_dump())
    entries = list(review.entries)
    entries[index] = updated
    return EvidenceConceptGoldReview.model_validate(
        {**review.model_dump(), "entries": [entry.model_dump() for entry in entries]}
    )


def review_summary(review: EvidenceConceptGoldReview) -> dict[str, object]:
    """Provide deterministic sampling and review progress diagnostics."""

    books = Counter(row.book_id for row in review.entries)
    return {
        "sample_count": len(review.entries),
        "topic_counts": dict(sorted(Counter(row.topic for row in review.entries).items())),
        "evidence_type_counts": dict(
            sorted(Counter(row.evidence_type for row in review.entries).items())
        ),
        "book_count": len(books),
        "maximum_samples_per_book": max(books.values(), default=0),
        "predicted_entry_count": sum(bool(row.predicted_concept_ids) for row in review.entries),
        "prediction_coverage": (
            sum(bool(row.predicted_concept_ids) for row in review.entries) / len(review.entries)
            if review.entries
            else 0.0
        ),
        "reviewed_entry_count": sum(row.review_status == "reviewed" for row in review.entries),
        "remaining_entry_count": sum(row.review_status == "unreviewed" for row in review.entries),
    }


def review_json_bytes(review: EvidenceConceptGoldReview) -> bytes:
    """Canonical pretty JSON used by determinism tests and CLI writes."""

    return (
        json.dumps(
            review.model_dump(mode="json"),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    ).encode()
