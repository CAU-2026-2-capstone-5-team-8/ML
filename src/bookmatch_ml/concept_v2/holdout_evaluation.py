"""Evaluate fixed matcher predictions on completed fresh human-review holdouts."""

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from bookmatch_ml.concept_v2.evidence_evaluation import (
    METADATA_EVIDENCE_TYPES,
    SUPPORTED_EVIDENCE_TYPES,
    TOC_EVIDENCE_TYPES,
    EvidenceMetric,
    calculate_evidence_metrics,
)
from bookmatch_ml.concept_v2.holdout import (
    MATCHER_VARIANTS,
    EvidenceConceptHoldoutManifest,
    EvidenceConceptHoldoutPredictions,
    EvidenceConceptHoldoutReview,
)
from bookmatch_ml.concept_v2.matcher_experiment_evaluation import ErrorAssignment, MatcherVariant
from bookmatch_ml.config import ConfigError

EVALUATION_VERSION = "evidence-concept-holdout-evaluation-v1"


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class HoldoutVariantEvaluation(_Strict):
    variant: MatcherVariant
    metrics: list[EvidenceMetric]
    error_assignments: list[ErrorAssignment]


class EvidenceConceptHoldoutEvaluationReport(_Strict):
    evaluation_version: Literal["evidence-concept-holdout-evaluation-v1"]
    holdout_kind: Literal["general", "challenge"]
    manifest_hash: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    prediction_artifact_hash: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    review_hash: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    total_entry_count: int = Field(ge=1)
    evaluable_entry_count: int = Field(ge=1)
    not_judgable_entry_count: int = Field(ge=0)
    category_membership_is_overlapping: bool
    variants: list[HoldoutVariantEvaluation]


@dataclass(frozen=True)
class _EvaluationRow:
    topic: str
    book_id: str
    evidence_id: str
    evidence_type: str
    challenge_categories: tuple[str, ...]
    predicted_concept_ids: list[str]
    human_gold_concept_ids: list[str]


def load_holdout_predictions(
    path: Path,
    manifest: EvidenceConceptHoldoutManifest,
    manifest_hash: str,
) -> EvidenceConceptHoldoutPredictions:
    """Load a fixed prediction artifact and verify its manifest membership."""

    try:
        predictions = EvidenceConceptHoldoutPredictions.model_validate_json(path.read_bytes())
        if predictions.manifest_hash != manifest_hash:
            raise ValueError("holdout predictions refer to a different manifest")
        expected = [row.evidence_id for row in manifest.entries]
        actual = [row.evidence_id for row in predictions.entries]
        if actual != expected or len(actual) != len(set(actual)):
            raise ValueError("holdout prediction membership or order differs from manifest")
    except (OSError, ValidationError, ValueError) as exc:
        raise ConfigError(f"invalid evidence concept holdout predictions: {path}: {exc}") from exc
    return predictions


def _metrics(
    rows: list[_EvaluationRow], holdout_kind: Literal["general", "challenge"]
) -> list[EvidenceMetric]:
    scopes: list[tuple[str, list[_EvaluationRow]]] = [("all", rows)]
    if holdout_kind == "general":
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
    else:
        scopes.extend(
            (
                f"challenge_category:{category}",
                [row for row in rows if category in row.challenge_categories],
            )
            for category in sorted(
                {category for row in rows for category in row.challenge_categories}
            )
        )
    return [calculate_evidence_metrics(scope_rows, scope) for scope, scope_rows in scopes]


def _errors(rows: list[_EvaluationRow]) -> list[ErrorAssignment]:
    assignments: list[ErrorAssignment] = []
    for row in rows:
        predicted = set(row.predicted_concept_ids)
        gold = set(row.human_gold_concept_ids)
        assignments.extend(
            ErrorAssignment(kind="false_positive", evidence_id=row.evidence_id, concept_id=value)
            for value in sorted(predicted - gold)
        )
        assignments.extend(
            ErrorAssignment(kind="false_negative", evidence_id=row.evidence_id, concept_id=value)
            for value in sorted(gold - predicted)
        )
    return sorted(
        assignments,
        key=lambda item: (item.kind, item.evidence_id, item.concept_id),
    )


def build_holdout_evaluation_report(
    manifest: EvidenceConceptHoldoutManifest,
    predictions: EvidenceConceptHoldoutPredictions,
    review: EvidenceConceptHoldoutReview,
    manifest_hash: str,
    prediction_hash: str,
    review_hash: str,
) -> EvidenceConceptHoldoutEvaluationReport:
    """Evaluate all fixed variants after a holdout review is complete."""

    if any(row.review_status != "reviewed" for row in review.entries):
        raise ValueError("holdout evaluation requires complete human review")
    prediction_by_id = {row.evidence_id: row for row in predictions.entries}
    variants: list[HoldoutVariantEvaluation] = []
    for variant in MATCHER_VARIANTS:
        rows = [
            _EvaluationRow(
                topic=row.topic,
                book_id=row.book_id,
                evidence_id=row.evidence_id,
                evidence_type=row.evidence_type,
                challenge_categories=tuple(row.challenge_categories),
                predicted_concept_ids=next(
                    item.predicted_concept_ids
                    for item in prediction_by_id[row.evidence_id].predictions
                    if item.variant == variant
                ),
                human_gold_concept_ids=list(row.human_gold_concept_ids),
            )
            for row in review.entries
            if row.review_outcome != "not_judgable"
        ]
        if not rows:
            raise ValueError("holdout evaluation has no evaluable human-reviewed rows")
        variants.append(
            HoldoutVariantEvaluation(
                variant=variant,
                metrics=_metrics(rows, manifest.holdout_kind),
                error_assignments=_errors(rows),
            )
        )
    return EvidenceConceptHoldoutEvaluationReport(
        evaluation_version=EVALUATION_VERSION,
        holdout_kind=manifest.holdout_kind,
        manifest_hash=manifest_hash,
        prediction_artifact_hash=prediction_hash,
        review_hash=review_hash,
        total_entry_count=len(review.entries),
        evaluable_entry_count=sum(row.review_outcome != "not_judgable" for row in review.entries),
        not_judgable_entry_count=sum(
            row.review_outcome == "not_judgable" for row in review.entries
        ),
        category_membership_is_overlapping=manifest.holdout_kind == "challenge",
        variants=variants,
    )
