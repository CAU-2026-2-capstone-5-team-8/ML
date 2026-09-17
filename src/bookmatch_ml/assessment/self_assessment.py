"""Validate self-report responses against an existing topic concept pool."""

from bookmatch_ml.assessment.pool import AssessmentBlueprintError
from bookmatch_ml.assessment.schemas import ConceptSelfAssessment, TopicConceptPool
from bookmatch_ml.config import LoadedAssessmentConfig


def validate_self_assessment(
    assessment: ConceptSelfAssessment,
    pool: TopicConceptPool,
    loaded_config: LoadedAssessmentConfig,
) -> None:
    """Require canonical, on-topic concept names without inferring mastery."""

    if assessment.self_assessment_version != loaded_config.config.self_assessment_version:
        raise AssessmentBlueprintError("self-assessment version is not supported")
    if (
        pool.assessment_config_version != loaded_config.config.config_version
        or pool.assessment_config_hash != loaded_config.content_hash
    ):
        raise AssessmentBlueprintError("concept pool does not match assessment configuration")
    if assessment.topic_id != pool.topic_id:
        raise AssessmentBlueprintError("self-assessment topic does not match concept pool")
    known = {concept.concept_id for concept in pool.concepts}
    unknown = sorted({response.concept_id for response in assessment.responses} - known)
    if unknown:
        raise AssessmentBlueprintError(f"unknown self-assessment concepts: {', '.join(unknown)}")
