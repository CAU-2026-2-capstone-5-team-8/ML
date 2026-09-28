"""Versioned human review gate for assessment-worthy concept targets."""

import hashlib
import json
from pathlib import Path

import yaml
from pydantic import ValidationError

from bookmatch_ml.assessment.schemas import (
    AssessmentBlueprint,
    AssessmentConceptReviewArtifact,
    AssessmentConceptReviewCandidate,
    AssessmentConceptReviewPacket,
    AssessmentConceptReviewRow,
    AssessmentEvidenceRef,
    LegacyAssessmentDiagnostics,
    LoadedAssessmentConceptReviews,
    TopicConcept,
    TopicConceptPool,
)
from bookmatch_ml.config import (
    LoadedAssessmentConfig,
    _load_unique_key_yaml,
)


class AssessmentConceptReviewError(ValueError):
    """Review decisions or queue inputs are invalid for the current concept pools."""


def load_assessment_concept_reviews(path: Path) -> LoadedAssessmentConceptReviews:
    """Load strict YAML decisions and retain the exact source-controlled byte hash."""

    try:
        content = path.read_bytes()
    except OSError as exc:
        raise AssessmentConceptReviewError(
            f"cannot read assessment concept review: {path}: {exc}"
        ) from exc
    try:
        artifact = AssessmentConceptReviewArtifact.model_validate(_load_unique_key_yaml(content))
    except (yaml.YAMLError, ValidationError, ValueError) as exc:
        raise AssessmentConceptReviewError(
            f"invalid assessment concept review: {path}: {exc}"
        ) from exc
    return LoadedAssessmentConceptReviews(
        artifact=artifact,
        content_hash=f"sha256:{hashlib.sha256(content).hexdigest()}",
    )


def reviewed_assessment_config(
    loaded_config: LoadedAssessmentConfig,
    loaded_reviews: LoadedAssessmentConceptReviews,
) -> LoadedAssessmentConfig:
    """Bind reviewed selection provenance to both config and exact decision bytes."""

    config = loaded_config.config
    if config.selection_mode != "reviewed":
        raise AssessmentConceptReviewError("assessment configuration is not in reviewed mode")
    if config.review_schema_version != loaded_reviews.artifact.review_version:
        raise AssessmentConceptReviewError(
            "assessment review schema version does not match reviewed configuration"
        )
    identity = json.dumps(
        {
            "assessment_config_hash": loaded_config.content_hash,
            "review_artifact_hash": loaded_reviews.content_hash,
            "review_version": loaded_reviews.artifact.review_version,
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return loaded_config.model_copy(
        update={"content_hash": f"sha256:{hashlib.sha256(identity).hexdigest()}"}
    )


def validate_assessment_concept_reviews(
    loaded_reviews: LoadedAssessmentConceptReviews,
    pools: list[TopicConceptPool],
    loaded_config: LoadedAssessmentConfig,
    *,
    supported_topics: set[str] | None = None,
) -> None:
    """Reject unknown topics and stale rows for topics represented by current pools."""

    if loaded_config.config.selection_mode != "reviewed":
        raise AssessmentConceptReviewError("concept reviews require reviewed assessment mode")
    if loaded_config.config.review_schema_version != loaded_reviews.artifact.review_version:
        raise AssessmentConceptReviewError("review version does not match assessment configuration")
    concepts_by_topic = {
        pool.topic_id: {(item.concept_id, item.role) for item in pool.concepts} for pool in pools
    }
    review_topics = {row.topic_id for row in loaded_reviews.artifact.reviews}
    unknown_topics = sorted(review_topics - (supported_topics or set(concepts_by_topic)))
    if unknown_topics:
        raise AssessmentConceptReviewError("unknown review topics: " + ", ".join(unknown_topics))
    stale = sorted(
        (row.topic_id, row.concept_id, row.concept_role)
        for row in loaded_reviews.artifact.reviews
        if row.topic_id in concepts_by_topic
        and (row.concept_id, row.concept_role) not in concepts_by_topic[row.topic_id]
    )
    if stale:
        formatted = ", ".join("/".join(item) for item in stale)
        raise AssessmentConceptReviewError(
            "stale or unknown concept review decisions: " + formatted
        )


def review_index(
    loaded_reviews: LoadedAssessmentConceptReviews,
) -> dict[tuple[str, str, str], AssessmentConceptReviewRow]:
    """Index the already duplicate-validated authoritative decisions."""

    return {
        (row.topic_id, row.concept_id, row.concept_role): row
        for row in loaded_reviews.artifact.reviews
    }


def eligible_concepts(
    pool: TopicConceptPool,
    role: str,
    loaded_reviews: LoadedAssessmentConceptReviews,
) -> list[TopicConcept]:
    """Keep only explicit eligible decisions, preserving existing pool priority order."""

    decisions = review_index(loaded_reviews)
    return [
        item
        for item in pool.concepts
        if item.role == role
        and (decision := decisions.get((item.topic_id, item.concept_id, item.role))) is not None
        and decision.status == "eligible"
    ]


def _compact_references(concept: TopicConcept, limit: int) -> list[AssessmentEvidenceRef]:
    references = sorted(
        (reference for support in concept.book_support for reference in support.evidence),
        key=lambda item: (item.book_id, item.evidence_type, item.evidence_id),
    )
    return references[:limit]


def build_assessment_concept_review_packet(
    pool: TopicConceptPool,
    legacy_blueprint: AssessmentBlueprint,
    loaded_config: LoadedAssessmentConfig,
    loaded_reviews: LoadedAssessmentConceptReviews,
    *,
    reserve: dict[str, int],
) -> AssessmentConceptReviewPacket:
    """Build a small deterministic evidence packet without making eligibility suggestions."""

    if set(reserve) != {"covered", "prerequisite"} or any(value < 0 for value in reserve.values()):
        raise AssessmentConceptReviewError(
            "reserve must provide non-negative counts for both roles"
        )
    decisions = review_index(loaded_reviews)
    legacy_selected = {
        (item.concept_id, item.role) for item in legacy_blueprint.selected_assessment_concepts
    }
    legacy_targets = {
        (item.primary_concept, item.concept_role) for item in legacy_blueprint.question_specs
    }
    ranks = {
        (item.concept_id, item.role): rank
        for role in ("covered", "prerequisite")
        for rank, item in enumerate(
            (candidate for candidate in pool.concepts if candidate.role == role), start=1
        )
    }
    candidates: list[AssessmentConceptReviewCandidate] = []
    for role in ("covered", "prerequisite"):
        role_candidates = [item for item in pool.concepts if item.role == role]
        window = loaded_config.config.self_assessment_limits[role] + reserve[role]
        for concept in role_candidates[:window]:
            decision = decisions.get((concept.topic_id, concept.concept_id, concept.role))
            candidates.append(
                AssessmentConceptReviewCandidate(
                    topic_id=concept.topic_id,
                    concept_id=concept.concept_id,
                    concept_role=concept.role,
                    assessment_priority=concept.assessment_priority,
                    book_coverage_count=concept.book_coverage_count,
                    book_coverage_rate=concept.book_coverage_rate,
                    mean_book_weight=concept.mean_book_weight,
                    evidence_types=concept.evidence_types,
                    prerequisite_methods=concept.prerequisite_methods,
                    supporting_book_ids=[item.book_id for item in concept.book_support],
                    compact_evidence_references=_compact_references(
                        concept, loaded_config.config.max_evidence_refs_per_spec
                    ),
                    current_self_assessment_selected=(
                        concept.concept_id,
                        concept.role,
                    )
                    in legacy_selected,
                    current_question_spec_target=(concept.concept_id, concept.role)
                    in legacy_targets,
                    priority_rank_within_role=ranks[(concept.concept_id, concept.role)],
                    status=decision.status if decision is not None else "unreviewed",
                    reason_code=decision.reason_code if decision is not None else None,
                    review_note=decision.review_note if decision is not None else "",
                )
            )
    counts = {
        question_type: sum(
            item.question_type == question_type for item in legacy_blueprint.question_specs
        )
        for question_type in ("vocabulary", "background_knowledge", "comprehension")
    }
    covered_count = sum(item.concept_role == "covered" for item in candidates)
    prerequisite_count = len(candidates) - covered_count
    return AssessmentConceptReviewPacket(
        packet_version="assessment-concept-review-packet-v1",
        review_version=loaded_reviews.artifact.review_version,
        topic_id=pool.topic_id,
        reserve=reserve,
        candidate_count=len(candidates),
        covered_candidate_count=covered_count,
        prerequisite_candidate_count=prerequisite_count,
        feature_config_version=pool.feature_config_version,
        feature_config_hash=pool.feature_config_hash,
        assessment_config_version=pool.assessment_config_version,
        assessment_config_hash=pool.assessment_config_hash,
        review_artifact_hash=loaded_reviews.content_hash,
        legacy_diagnostics=LegacyAssessmentDiagnostics(
            selected_covered_concepts=[
                item.concept_id
                for item in legacy_blueprint.selected_assessment_concepts
                if item.role == "covered"
            ],
            selected_prerequisite_concepts=[
                item.concept_id
                for item in legacy_blueprint.selected_assessment_concepts
                if item.role == "prerequisite"
            ],
            question_spec_count_by_type=counts,
            shortages=legacy_blueprint.shortages,
        ),
        candidates=candidates,
    )
