"""Compare evidence-aware presentation policies over unchanged matching scores."""

from math import isclose

from bookmatch_ml.config import LoadedRankingConfig, LoadedRankingPolicyExperimentConfig
from bookmatch_ml.evaluation.difficulty import EvaluationDataError
from bookmatch_ml.ranking.matching import rank_books
from bookmatch_ml.schemas import (
    BookProfile,
    RankedBook,
    RankingPolicyCandidate,
    RankingPolicyEvaluationReport,
    RankingPolicyName,
    RankingPolicyResult,
    RankingPolicyTopKDifference,
    ReaderProfile,
)


def _meets_minimum(coverage: float, minimum: float) -> bool:
    return coverage >= minimum or isclose(coverage, minimum, rel_tol=0, abs_tol=1e-9)


def _topic_score(item: RankedBook) -> float:
    if item.components.topic_fit is None:
        raise EvaluationDataError(f"topic fit is unavailable for policy evaluation: {item.book_id}")
    return item.components.topic_fit


def _result_for_policy(
    policy: RankingPolicyName,
    baseline: list[RankedBook],
    minimum: float,
    top_k: int,
) -> RankingPolicyResult:
    if policy == "renormalized":
        readiness = list(baseline)
        limited: list[RankedBook] = []
    else:
        readiness = [
            item for item in baseline if _meets_minimum(item.component_weight_coverage, minimum)
        ]
        limited = [
            item for item in baseline if not _meets_minimum(item.component_weight_coverage, minimum)
        ]
        limited.sort(key=lambda item: (-_topic_score(item), item.book_id))

    baseline_position = {item.book_id: index for index, item in enumerate(baseline, start=1)}
    items = []
    for display_position, item in enumerate([*readiness, *limited], start=1):
        eligible = display_position <= len(readiness)
        group = (
            "readiness"
            if eligible
            else "ineligible"
            if policy == "minimum_coverage"
            else "topic_only"
        )
        reason = None
        if not eligible:
            unavailable = ", ".join(item.unavailable_components) or "none"
            reason = (
                f"component_weight_coverage {item.component_weight_coverage:.6g} is below "
                f"minimum {minimum:.6g}; unavailable components: {unavailable}"
            )
        items.append(
            RankingPolicyCandidate(
                book_id=item.book_id,
                display_position=display_position,
                position_within_group=(
                    display_position if eligible else display_position - len(readiness)
                ),
                renormalized_position=baseline_position[item.book_id],
                display_position_change=baseline_position[item.book_id] - display_position,
                group=group,
                readiness_evidence=(
                    "full" if isclose(item.component_weight_coverage, 1.0) else "partial"
                ),
                readiness_score=item.score if eligible else None,
                diagnostic_renormalized_score=item.score,
                topic_only_score=_topic_score(item) if not eligible else None,
                component_weight_coverage=item.component_weight_coverage,
                unavailable_components=item.unavailable_components,
                active_weights=item.active_weights,
                components=item.components,
                knowledge_components=item.knowledge_components,
                knowledge_active_weights=item.knowledge_active_weights,
                knowledge_weight_coverage=item.knowledge_weight_coverage,
                diagnostics=item.diagnostics,
                reasons=item.reasons,
                eligibility_reason=reason,
                book_feature_version=item.book_feature_version,
                book_config_version=item.book_config_version,
                book_config_hash=item.book_config_hash,
            )
        )
    return RankingPolicyResult(
        policy=policy,
        readiness_candidate_count=len(readiness),
        evidence_limited_count=len(limited),
        readiness_top_k=[item.book_id for item in readiness[:top_k]],
        readiness_top_k_complete=len(readiness) >= top_k,
        items=items,
    )


def evaluate_ranking_policies(
    reader: ReaderProfile,
    books: list[BookProfile],
    ranking_config: LoadedRankingConfig,
    policy_config: LoadedRankingPolicyExperimentConfig,
    *,
    canonical_book_count: int,
) -> RankingPolicyEvaluationReport:
    """Expose policy tradeoffs without changing the score or the production API."""

    config = policy_config.config
    if not books:
        raise EvaluationDataError("ranking policy evaluation requires book profiles")
    baseline = rank_books(reader, books, ranking_config, limit=len(books)).items
    if not baseline:
        raise EvaluationDataError(f"no ranking candidates for topic {reader.topic_id}")
    if len(baseline) < config.top_k:
        raise EvaluationDataError(
            f"policy top_k {config.top_k} exceeds candidate count {len(baseline)}"
        )
    results = [
        _result_for_policy(policy, baseline, config.minimum_component_weight_coverage, config.top_k)
        for policy in config.policies
    ]
    renormalized = next(result for result in results if result.policy == "renormalized")
    differences = []
    for result in results:
        if result.policy == "renormalized":
            continue
        comparable = result.readiness_top_k_complete
        differences.append(
            RankingPolicyTopKDifference(
                left_policy="renormalized",
                right_policy=result.policy,
                comparable=comparable,
                left_only=(
                    sorted(set(renormalized.readiness_top_k) - set(result.readiness_top_k))
                    if comparable
                    else []
                ),
                right_only=(
                    sorted(set(result.readiness_top_k) - set(renormalized.readiness_top_k))
                    if comparable
                    else []
                ),
                limitation=(
                    None
                    if comparable
                    else "Fewer eligible readiness candidates than configured top_k."
                ),
            )
        )
    return RankingPolicyEvaluationReport(
        evaluation_version=config.evaluation_version,
        topic_id=reader.topic_id,
        canonical_book_count=canonical_book_count,
        candidate_count=len(baseline),
        top_k=config.top_k,
        minimum_component_weight_coverage=config.minimum_component_weight_coverage,
        policy_config_version=config.config_version,
        policy_config_hash=policy_config.content_hash,
        ranking_model_version=ranking_config.config.model_version,
        ranking_config_version=ranking_config.config.config_version,
        ranking_config_hash=ranking_config.content_hash,
        reader_profile_version=reader.profile_version,
        reader_config_version=reader.config_version,
        reader_config_hash=reader.config_hash,
        policies=results,
        top_k_differences=differences,
    )
