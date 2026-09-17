"""Question Difficulty v1 assigns intended cognition, not concept difficulty."""

from bookmatch_ml.assessment.pool import AssessmentBlueprintError
from bookmatch_ml.assessment.schemas import CognitiveOperation, TargetDifficulty
from bookmatch_ml.config import LoadedAssessmentConfig


def assign_target_difficulty(
    operation: CognitiveOperation,
    related_concept_count: int,
    loaded_config: LoadedAssessmentConfig,
) -> tuple[TargetDifficulty, bool, bool, str]:
    """Apply operation and structure rules without occurrence or prose scores."""

    config = loaded_config.config
    try:
        level = config.operation_levels[operation]
        required_related = config.minimum_related_concepts[operation]
    except KeyError as exc:
        raise AssessmentBlueprintError(f"unsupported cognitive operation: {operation}") from exc
    if related_concept_count < required_related:
        raise AssessmentBlueprintError(
            f"{operation} requires at least {required_related} related concepts"
        )
    relationship = operation in config.relationship_operations
    multi_step = operation in config.multi_step_operations
    if level == 1 and (related_concept_count or relationship or multi_step):
        raise AssessmentBlueprintError("Level 1 must be direct recognition or recall")
    if level == 2 and multi_step:
        raise AssessmentBlueprintError("Level 2 cannot require multi-step reasoning")
    if level == 3 and not multi_step:
        raise AssessmentBlueprintError("Level 3 must require multi-step reasoning")
    rationale = {
        "recognize": "identify the primary term directly, without a relation step",
        "recall": "recall the inferred prerequisite directly, without a relation step",
        "compare": "directly distinguish configured related concepts in one comparison step",
        "relate": "explain a configured concept relation in one step",
        "apply": "apply the primary concept in a simple situation grounded in analyzed prose",
        "integrate": (
            "integrate configured concepts through multiple reasoning steps; "
            "verify the relation when authoring"
        ),
        "infer": "draw a multi-step inference from the authored question context",
    }[operation]
    return level, relationship, multi_step, f"Level {level}: {rationale}."
