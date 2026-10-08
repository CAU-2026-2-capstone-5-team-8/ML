"""Observed concept × ability evidence, separate from self reports and legacy scores."""

from collections import defaultdict

from bookmatch_ml.config import LoadedReaderConfig
from bookmatch_ml.schemas import Assessment

OPERATIONS = {
    "recognize": "meaning",
    "recall": "meaning",
    "apply": "application",
    "compare": "reasoning",
    "relate": "reasoning",
    "integrate": "reasoning",
    "infer": "reasoning",
}


def build_concept_ability_profile(
    assessment: Assessment, config: LoadedReaderConfig
) -> dict[str, object]:
    """Do not infer operations from question type or count self reports as tested ability."""
    tested: dict[tuple[str, str, str | None], list] = defaultdict(list)
    reported: dict[str, list] = defaultdict(list)
    unclassified = 0
    for response in assessment.responses:
        if not response.concept_id:
            unclassified += 1
        elif response.answer_mode == "SELF_REPORT":
            reported[response.concept_id].append(response)
        elif response.answer_mode == "MULTIPLE_CHOICE" and response.cognitive_operation:
            tested[
                (
                    response.concept_id,
                    OPERATIONS[response.cognitive_operation],
                    response.measurement_context,
                )
            ].append(response)
        else:
            unclassified += 1

    abilities = []
    for (concept, ability, context), responses in sorted(
        tested.items(), key=lambda row: (row[0][0], row[0][1], row[0][2] or "")
    ):
        weights = [config.config.difficulty_weights[r.difficulty] for r in responses]
        scores = [r.score if r.score is not None else float(r.correct) for r in responses]
        abilities.append(
            {
                "conceptId": concept,
                "ability": ability,
                "measurementContext": context,
                "score": sum(s * w for s, w in zip(scores, weights, strict=True)) / sum(weights),
                "responseCount": len(responses),
                "correctCount": sum(s == 1.0 for s in scores),
                "questionIds": [r.question_id for r in responses],
                "operations": sorted({r.cognitive_operation for r in responses}),
            }
        )
    self_reports = [
        {
            "conceptId": concept,
            "positiveCount": sum(
                (r.score if r.score is not None else float(r.correct)) == 1.0 for r in responses
            ),
            "responseCount": len(responses),
            "questionIds": [r.question_id for r in responses],
        }
        for concept, responses in sorted(reported.items())
    ]
    return {
        "version": "concept-abilities-v2",
        "abilityIds": ["meaning", "application", "reasoning"],
        "abilities": [a for a in abilities if a["measurementContext"] == "prior-knowledge"],
        "providedInformationAbilities": [
            a for a in abilities if a["measurementContext"] == "provided-information"
        ],
        "legacyContextAbilities": [a for a in abilities if a["measurementContext"] is None],
        "selfReports": self_reports,
        "unclassifiedResponseCount": unclassified,
        "interpretation": "observed_answers_not_calibrated_mastery",
    }
