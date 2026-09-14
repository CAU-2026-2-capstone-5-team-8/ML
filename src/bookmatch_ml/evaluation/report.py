"""End-to-end assembly of the versioned baseline evaluation report."""

from bookmatch_ml.config import (
    LoadedEvaluationConfig,
    LoadedFeatureConfig,
    LoadedRankingConfig,
)
from bookmatch_ml.evaluation.ablation import (
    build_evidence_ablation_report,
    summarize_evidence_ablation,
)
from bookmatch_ml.evaluation.difficulty import EvaluationDataError, evaluate_book_difficulty
from bookmatch_ml.evaluation.recommendation import compare_recommendation_baselines
from bookmatch_ml.schemas import (
    BookEvidence,
    BookProfile,
    EvaluationFailureCase,
    EvaluationReport,
    LoadedDifficultyJudgments,
    ReaderProfile,
)


def build_evaluation_report(
    judgments: LoadedDifficultyJudgments,
    reader: ReaderProfile,
    books: list[BookProfile],
    ablation_evidence: BookEvidence,
    feature_config: LoadedFeatureConfig,
    ranking_config: LoadedRankingConfig,
    evaluation_config: LoadedEvaluationConfig,
) -> EvaluationReport:
    """Run all baseline evaluations and retain evidence-limited failure cases."""

    if judgments.judgments.topic_id != reader.topic_id:
        raise EvaluationDataError(
            "difficulty judgment topic does not match reader profile topic: "
            f"{judgments.judgments.topic_id} != {reader.topic_id}"
        )
    difficulty = evaluate_book_difficulty(judgments, books, evaluation_config)
    recommendation = compare_recommendation_baselines(
        reader, books, ranking_config, evaluation_config
    )
    ablation = build_evidence_ablation_report(ablation_evidence, feature_config)
    failure_cases = [
        EvaluationFailureCase(
            category="difficulty_unavailable",
            book_id=book_id,
            detail="No prose-based difficulty score was available for this labeled book.",
        )
        for book_id in difficulty.excluded_book_ids
    ]
    failure_cases.extend(
        EvaluationFailureCase(
            category="ranking_evidence_limited",
            book_id=item.book_id,
            detail=(
                f"Readiness ranking used {item.component_weight_coverage:.2f} of configured "
                "component weight; unavailable: " + ", ".join(item.unavailable_components)
            ),
        )
        for item in recommendation.items
        if item.component_weight_coverage < 1.0
    )
    failure_cases.sort(key=lambda item: (item.category, item.book_id))
    config = evaluation_config.config
    return EvaluationReport(
        evaluation_version=config.evaluation_version,
        config_version=config.config_version,
        config_hash=evaluation_config.content_hash,
        difficulty=difficulty,
        recommendation=recommendation,
        evidence_ablation=ablation,
        ablation_comparisons=summarize_evidence_ablation(ablation),
        failure_cases=failure_cases,
    )
