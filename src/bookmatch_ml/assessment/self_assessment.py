"""Validate self-report responses against an existing topic concept pool."""

from bookmatch_ml.assessment.pool import AssessmentBlueprintError
from bookmatch_ml.assessment.schemas import ConceptSelfAssessment, TopicConceptPool


def validate_self_assessment(assessment: ConceptSelfAssessment, pool: TopicConceptPool) -> None:
    """Require canonical, on-topic concept names without inferring mastery."""

    if assessment.topic_id != pool.topic_id:
        raise AssessmentBlueprintError("self-assessment topic does not match concept pool")
    known = {concept.concept_id for concept in pool.concepts}
    unknown = sorted({response.concept_id for response in assessment.responses} - known)
    if unknown:
        raise AssessmentBlueprintError(f"unknown self-assessment concepts: {', '.join(unknown)}")
