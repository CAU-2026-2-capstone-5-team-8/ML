"""Topic-only versus readiness-aware recommendation comparison."""

from bookmatch_ml.config import LoadedEvaluationConfig, LoadedRankingConfig
from bookmatch_ml.evaluation.difficulty import EvaluationDataError
from bookmatch_ml.ranking.matching import rank_books
from bookmatch_ml.schemas import (
    BookProfile,
    ReaderProfile,
    RecommendationComparisonItem,
    RecommendationComparisonReport,
)


def compare_recommendation_baselines(
    reader: ReaderProfile,
    books: list[BookProfile],
    ranking_config: LoadedRankingConfig,
    evaluation_config: LoadedEvaluationConfig,
) -> RecommendationComparisonReport:
    """Compare deterministic topic ordering with the full reader-readiness ranking."""

    if ranking_config.config.filter_topic_candidates:
        candidates = [
            book
            for book in books
            if book.concept_profile.topic_distribution.get(reader.topic_id, 0.0) > 0
        ]
    else:
        candidates = list(books)
    if not candidates:
        raise EvaluationDataError(f"no recommendation candidates for topic {reader.topic_id}")

    topic_only = sorted(
        (
            (book.book_id, book.concept_profile.topic_distribution.get(reader.topic_id, 0.0))
            for book in candidates
        ),
        key=lambda item: (-item[1], item[0]),
    )
    readiness = rank_books(reader, candidates, ranking_config, limit=len(candidates)).items
    topic_rank = {book_id: index for index, (book_id, _) in enumerate(topic_only, start=1)}
    topic_score = dict(topic_only)
    readiness_rank = {item.book_id: index for index, item in enumerate(readiness, start=1)}

    items = [
        RecommendationComparisonItem(
            book_id=item.book_id,
            topic_only_rank=topic_rank[item.book_id],
            topic_only_score=topic_score[item.book_id],
            readiness_rank=readiness_rank[item.book_id],
            readiness_score=item.score,
            readiness_rank_change=topic_rank[item.book_id] - readiness_rank[item.book_id],
            component_weight_coverage=item.component_weight_coverage,
            unavailable_components=item.unavailable_components,
        )
        for item in readiness
    ]
    comparison_k = min(evaluation_config.config.recommendation.top_k, len(candidates))
    topic_top_k = {book_id for book_id, _ in topic_only[:comparison_k]}
    readiness_top_k = {item.book_id for item in readiness[:comparison_k]}
    overlap_count = len(topic_top_k & readiness_top_k)
    return RecommendationComparisonReport(
        topic_id=reader.topic_id,
        candidate_count=len(candidates),
        comparison_k=comparison_k,
        top_k_overlap_count=overlap_count,
        top_k_overlap_rate=overlap_count / comparison_k,
        changed_position_count=sum(item.topic_only_rank != item.readiness_rank for item in items),
        topic_only_version=evaluation_config.config.recommendation.topic_only_version,
        readiness_model_version=ranking_config.config.model_version,
        ranking_config_version=ranking_config.config.config_version,
        ranking_config_hash=ranking_config.content_hash,
        items=items,
    )
