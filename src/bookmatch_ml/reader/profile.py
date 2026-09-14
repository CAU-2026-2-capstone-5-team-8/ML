"""Transparent difficulty-weighted reader profile baseline."""

from collections import defaultdict
from pathlib import Path

from pydantic import ValidationError

from bookmatch_ml.config import LoadedReaderConfig
from bookmatch_ml.schemas import (
    Assessment,
    AssessmentResponse,
    ConceptReadiness,
    ReaderDimensionDetail,
    ReaderProfile,
)


class AssessmentError(ValueError):
    """Assessment input cannot produce a valid reader profile."""


def load_assessment(path: Path) -> Assessment:
    """Load one assessment JSON document with a path-aware error."""

    try:
        content = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise AssessmentError(f"cannot read assessment: {path}: {exc}") from exc
    try:
        return Assessment.model_validate_json(content)
    except (ValidationError, ValueError) as exc:
        raise AssessmentError(f"invalid assessment: {path}: {exc}") from exc


def _response_score(response: AssessmentResponse) -> float:
    if response.score is not None:
        return response.score
    return 1.0 if response.correct else 0.0


def _weighted_result(
    responses: list[AssessmentResponse],
    difficulty_weights: dict[str, float],
) -> tuple[float, float, float]:
    available_weight = sum(difficulty_weights[response.difficulty] for response in responses)
    earned_weight = sum(
        _response_score(response) * difficulty_weights[response.difficulty]
        for response in responses
    )
    return earned_weight / available_weight, earned_weight, available_weight


def build_reader_profile(
    assessment: Assessment,
    loaded_config: LoadedReaderConfig,
) -> ReaderProfile:
    """Score readiness dimensions and tagged concepts with configured difficulty weights."""

    config = loaded_config.config
    unsupported = sorted(
        {
            response.difficulty
            for response in assessment.responses
            if response.difficulty not in config.difficulty_weights
        }
    )
    if unsupported:
        raise AssessmentError(f"unsupported question difficulties: {', '.join(unsupported)}")

    responses_by_type: dict[str, list[AssessmentResponse]] = defaultdict(list)
    for response in assessment.responses:
        responses_by_type[response.question_type].append(response)
    insufficient = [
        question_type
        for question_type in config.required_question_types
        if len(responses_by_type[question_type]) < config.minimum_responses_per_type
    ]
    if insufficient:
        raise AssessmentError(
            "insufficient responses for question types: " + ", ".join(sorted(insufficient))
        )

    dimension_details: list[ReaderDimensionDetail] = []
    dimension_scores: dict[str, float] = {}
    for question_type in config.required_question_types:
        responses = responses_by_type[question_type]
        score, earned_weight, available_weight = _weighted_result(
            responses, config.difficulty_weights
        )
        dimension_scores[question_type] = score
        dimension_details.append(
            ReaderDimensionDetail(
                question_type=question_type,
                score=score,
                response_count=len(responses),
                earned_weight=earned_weight,
                available_weight=available_weight,
            )
        )

    responses_by_concept: dict[str, list[AssessmentResponse]] = defaultdict(list)
    for response in assessment.responses:
        concepts = set(response.concept_tags)
        if response.concept_id is not None:
            concepts.add(response.concept_id)
        for concept in concepts:
            responses_by_concept[concept].append(response)
    concept_readiness: list[ConceptReadiness] = []
    for concept, responses in sorted(responses_by_concept.items()):
        score, earned_weight, available_weight = _weighted_result(
            responses, config.difficulty_weights
        )
        concept_readiness.append(
            ConceptReadiness(
                concept_id=concept,
                score=score,
                response_count=len(responses),
                earned_weight=earned_weight,
                available_weight=available_weight,
            )
        )

    return ReaderProfile(
        assessment_id=assessment.assessment_id,
        topic_id=assessment.topic_id,
        vocabulary=dimension_scores["vocabulary"],
        background_knowledge=dimension_scores["background_knowledge"],
        comprehension=dimension_scores["comprehension"],
        dimension_details=dimension_details,
        concept_readiness=concept_readiness,
        response_count=len(assessment.responses),
        profile_version=config.profile_version,
        config_version=config.config_version,
        config_hash=loaded_config.content_hash,
    )
