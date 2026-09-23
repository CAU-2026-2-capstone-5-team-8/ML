"""Ablation evaluation for matcher variants against frozen human gold."""

from collections import defaultdict
from dataclasses import dataclass
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from bookmatch_ml.concept_v2.evidence_evaluation import (
    METADATA_EVIDENCE_TYPES,
    SUPPORTED_EVIDENCE_TYPES,
    TOC_EVIDENCE_TYPES,
    EvidenceMetric,
    LoadedEvidenceConceptGoldReview,
    _prf,
    calculate_evidence_metrics,
)
from bookmatch_ml.concept_v2.graph import LoadedConceptGraph
from bookmatch_ml.concept_v2.matcher_experiments import (
    LoadedMatcherV2ExperimentConfig,
    match_concept_text_v2_a,
    match_concept_text_v2_b,
    match_concept_toc_v2_c1,
    match_concept_toc_v2_c2,
)
from bookmatch_ml.concept_v2.profile import LoadedConceptMatchingConfig
from bookmatch_ml.config import LoadedFeatureConfig

EXPERIMENT_VERSION = "matcher-v2-ablation-v1"
MatcherVariant = Literal[
    "v1",
    "v2_a_overlap",
    "v2_b_forms",
    "v2_c1_naive_path_union",
    "v2_c2_context_assisted",
]


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class ErrorAssignment(_Strict):
    kind: Literal["false_positive", "false_negative"]
    evidence_id: str
    concept_id: str


class ExperimentPolicyMetric(_Strict):
    policy: Literal["toc_only", "metadata_only", "combined_unweighted"]
    book_count: int = Field(ge=1)
    true_positive_count: int = Field(ge=0)
    false_positive_count: int = Field(ge=0)
    false_negative_count: int = Field(ge=0)
    exact_set_match_rate: float = Field(ge=0, le=1)
    micro_precision: float = Field(ge=0, le=1)
    micro_recall: float = Field(ge=0, le=1)
    micro_f1: float = Field(ge=0, le=1)
    book_macro_precision: float = Field(ge=0, le=1)
    book_macro_recall: float = Field(ge=0, le=1)
    book_macro_f1: float = Field(ge=0, le=1)
    unique_book_concept_assignment_count: int = Field(ge=0)
    raw_prediction_occurrence_count: int = Field(ge=0)


class MatcherVariantResult(_Strict):
    variant: MatcherVariant
    metrics: list[EvidenceMetric]
    policy_metrics: list[ExperimentPolicyMetric]
    error_assignments: list[ErrorAssignment]
    fixed_v1_errors: list[ErrorAssignment]
    remaining_v1_errors: list[ErrorAssignment]
    newly_introduced_errors: list[ErrorAssignment]


class MatcherV2ExperimentReport(_Strict):
    experiment_version: Literal["matcher-v2-ablation-v1"]
    frozen_review_hash: str
    feature_config_hash: str
    graph_hash: str
    matching_v1_config_hash: str
    experiment_config_hash: str
    evaluable_entry_count: int = Field(ge=1)
    variants: list[MatcherVariantResult]


@dataclass(frozen=True)
class _EvaluationRow:
    topic: str
    book_id: str
    evidence_id: str
    evidence_type: str
    predicted_concept_ids: list[str]
    human_gold_concept_ids: list[str]


def _predict_variant(
    variant: MatcherVariant,
    row,
    features: LoadedFeatureConfig,
    graph: LoadedConceptGraph,
    matching: LoadedConceptMatchingConfig,
    experiment: LoadedMatcherV2ExperimentConfig,
) -> list[str]:
    if variant == "v1":
        return list(row.predicted_concept_ids)
    if variant == "v2_a_overlap":
        matches, _ = match_concept_text_v2_a(
            row.evidence_text,
            row.topic,
            features,
            graph,
            matching,
        )
    elif variant == "v2_b_forms" or row.evidence_type not in TOC_EVIDENCE_TYPES:
        matches, _ = match_concept_text_v2_b(
            row.evidence_text,
            row.topic,
            row.evidence_type,
            features,
            graph,
            matching,
            experiment,
        )
    elif variant == "v2_c1_naive_path_union":
        matches, _ = match_concept_toc_v2_c1(
            row.evidence_text,
            row.toc_path,
            row.topic,
            row.evidence_type,
            features,
            graph,
            matching,
            experiment,
        )
    else:
        matches, _ = match_concept_toc_v2_c2(
            row.evidence_text,
            row.toc_path,
            row.topic,
            row.evidence_type,
            features,
            graph,
            matching,
            experiment,
        )
    return sorted({match.concept_id for match in matches})


def _rows_with_predictions(review, predictions: dict[str, list[str]]) -> list[_EvaluationRow]:
    return [
        _EvaluationRow(
            topic=row.topic,
            book_id=row.book_id,
            evidence_id=row.evidence_id,
            evidence_type=row.evidence_type,
            predicted_concept_ids=predictions[row.evidence_id],
            human_gold_concept_ids=list(row.human_gold_concept_ids),
        )
        for row in review.entries
        if row.review_status == "reviewed" and row.review_outcome != "not_judgable"
    ]


def _metrics(rows: list[_EvaluationRow]) -> list[EvidenceMetric]:
    scopes: list[tuple[str, list[_EvaluationRow]]] = [("all", rows)]
    scopes.extend(
        (f"topic:{topic}", [row for row in rows if row.topic == topic])
        for topic in sorted({row.topic for row in rows})
    )
    scopes.extend(
        (
            f"evidence_type:{evidence_type}",
            [row for row in rows if row.evidence_type == evidence_type],
        )
        for evidence_type in SUPPORTED_EVIDENCE_TYPES
        if any(row.evidence_type == evidence_type for row in rows)
    )
    for family, types in (("toc", TOC_EVIDENCE_TYPES), ("metadata", METADATA_EVIDENCE_TYPES)):
        family_rows = [row for row in rows if row.evidence_type in types]
        if family_rows:
            scopes.append((f"family:{family}", family_rows))
    return [calculate_evidence_metrics(scope_rows, scope) for scope, scope_rows in scopes]


def _policy_metric(
    rows: list[_EvaluationRow],
    policy: Literal["toc_only", "metadata_only", "combined_unweighted"],
) -> ExperimentPolicyMetric:
    allowed = {
        "toc_only": TOC_EVIDENCE_TYPES,
        "metadata_only": METADATA_EVIDENCE_TYPES,
        "combined_unweighted": TOC_EVIDENCE_TYPES | METADATA_EVIDENCE_TYPES,
    }[policy]
    gold_by_book: dict[str, set[str]] = defaultdict(set)
    predicted_by_book: dict[str, set[str]] = defaultdict(set)
    eligible_books: set[str] = set()
    raw_occurrences = 0
    for row in rows:
        gold_by_book[row.book_id].update(row.human_gold_concept_ids)
        if row.evidence_type in allowed:
            eligible_books.add(row.book_id)
            predicted_by_book[row.book_id].update(row.predicted_concept_ids)
            raw_occurrences += len(row.predicted_concept_ids)
    values = [
        _prf(predicted_by_book[book_id], gold_by_book[book_id])
        for book_id in sorted(eligible_books)
    ]
    tp = sum(value[0] for value in values)
    fp = sum(value[1] for value in values)
    fn = sum(value[2] for value in values)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return ExperimentPolicyMetric(
        policy=policy,
        book_count=len(values),
        true_positive_count=tp,
        false_positive_count=fp,
        false_negative_count=fn,
        exact_set_match_rate=sum(value[1] == 0 and value[2] == 0 for value in values) / len(values),
        micro_precision=precision,
        micro_recall=recall,
        micro_f1=f1,
        book_macro_precision=sum(value[3] for value in values) / len(values),
        book_macro_recall=sum(value[4] for value in values) / len(values),
        book_macro_f1=sum(value[5] for value in values) / len(values),
        unique_book_concept_assignment_count=sum(
            len(predicted_by_book[book_id]) for book_id in eligible_books
        ),
        raw_prediction_occurrence_count=raw_occurrences,
    )


def _errors(rows: list[_EvaluationRow]) -> list[ErrorAssignment]:
    errors: list[ErrorAssignment] = []
    for row in rows:
        predicted = set(row.predicted_concept_ids)
        gold = set(row.human_gold_concept_ids)
        errors.extend(
            ErrorAssignment(kind="false_positive", evidence_id=row.evidence_id, concept_id=concept)
            for concept in sorted(predicted - gold)
        )
        errors.extend(
            ErrorAssignment(kind="false_negative", evidence_id=row.evidence_id, concept_id=concept)
            for concept in sorted(gold - predicted)
        )
    return sorted(errors, key=lambda item: (item.kind, item.evidence_id, item.concept_id))


def _error_key(error: ErrorAssignment) -> tuple[str, str, str]:
    return error.kind, error.evidence_id, error.concept_id


def build_matcher_v2_experiment_report(
    loaded: LoadedEvidenceConceptGoldReview,
    features: LoadedFeatureConfig,
    graph: LoadedConceptGraph,
    matching: LoadedConceptMatchingConfig,
    experiment: LoadedMatcherV2ExperimentConfig,
) -> MatcherV2ExperimentReport:
    """Evaluate isolated variants without mutating frozen predictions or gold."""

    variants: tuple[MatcherVariant, ...] = (
        "v1",
        "v2_a_overlap",
        "v2_b_forms",
        "v2_c1_naive_path_union",
        "v2_c2_context_assisted",
    )
    predictions_by_variant = {
        variant: {
            row.evidence_id: _predict_variant(
                variant,
                row,
                features,
                graph,
                matching,
                experiment,
            )
            for row in loaded.review.entries
        }
        for variant in variants
    }
    rows_by_variant = {
        variant: _rows_with_predictions(loaded.review, predictions)
        for variant, predictions in predictions_by_variant.items()
    }
    baseline_errors = _errors(rows_by_variant["v1"])
    baseline_by_key = {_error_key(error): error for error in baseline_errors}
    results: list[MatcherVariantResult] = []
    for variant in variants:
        rows = rows_by_variant[variant]
        errors = _errors(rows)
        errors_by_key = {_error_key(error): error for error in errors}
        fixed_keys = set(baseline_by_key) - set(errors_by_key)
        remaining_keys = set(baseline_by_key) & set(errors_by_key)
        new_keys = set(errors_by_key) - set(baseline_by_key)
        results.append(
            MatcherVariantResult(
                variant=variant,
                metrics=_metrics(rows),
                policy_metrics=[
                    _policy_metric(rows, policy)
                    for policy in ("toc_only", "metadata_only", "combined_unweighted")
                ],
                error_assignments=errors,
                fixed_v1_errors=[baseline_by_key[key] for key in sorted(fixed_keys)],
                remaining_v1_errors=[baseline_by_key[key] for key in sorted(remaining_keys)],
                newly_introduced_errors=[errors_by_key[key] for key in sorted(new_keys)],
            )
        )
    return MatcherV2ExperimentReport(
        experiment_version=EXPERIMENT_VERSION,
        frozen_review_hash=loaded.content_hash,
        feature_config_hash=features.content_hash,
        graph_hash=graph.content_hash,
        matching_v1_config_hash=matching.content_hash,
        experiment_config_hash=experiment.content_hash,
        evaluable_entry_count=len(rows_by_variant["v1"]),
        variants=results,
    )
