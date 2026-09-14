"""Configurable score decomposition with missing-component renormalization."""

from bookmatch_ml.config import LoadedRankingConfig
from bookmatch_ml.ranking.explanation import build_reasons
from bookmatch_ml.schemas import (
    BookProfile,
    KnowledgeFitComponents,
    MatchingDiagnostics,
    RankedBook,
    RankingComponents,
    RankingResponse,
    ReaderProfile,
)


class RankingError(ValueError):
    """A reader-book fit cannot be calculated from the available components."""


def _fit(readiness: float, demand: float) -> float:
    return max(0.0, 1.0 - abs(readiness - demand))


def _renormalized_weighted_mean(
    values: dict[str, float | None],
    configured_weights: dict[str, float],
) -> tuple[float | None, dict[str, float]]:
    active = {name: value for name, value in values.items() if value is not None}
    active_weight = sum(configured_weights[name] for name in active)
    if not active or active_weight == 0:
        return None, {}
    normalized_weights = {
        name: configured_weights[name] / active_weight
        for name in configured_weights
        if name in active
    }
    score = sum(active[name] * weight for name, weight in normalized_weights.items())
    return score, normalized_weights


def _prerequisite_concept_fit(
    reader: ReaderProfile,
    book: BookProfile,
) -> tuple[float | None, int]:
    reader_concepts = {item.concept_id: item.score for item in reader.concept_readiness}
    matched = [
        prerequisite
        for prerequisite in book.concept_profile.prerequisite_concepts
        if prerequisite.concept in reader_concepts
    ]
    total_weight = sum(item.weight for item in matched)
    if not matched or total_weight == 0:
        return None, 0
    score = sum(reader_concepts[item.concept] * item.weight for item in matched) / total_weight
    return score, len(matched)


def score_book_fit(
    reader: ReaderProfile,
    book: BookProfile,
    loaded_config: LoadedRankingConfig,
) -> RankedBook:
    """Calculate one explainable reader-book fit score."""

    config = loaded_config.config
    topic_distribution = book.concept_profile.topic_distribution
    topic_fit = topic_distribution.get(reader.topic_id, 0.0) if topic_distribution else None
    difficulty = book.difficulty_profile
    vocabulary_fit = (
        _fit(reader.vocabulary, difficulty.lexical_difficulty)
        if difficulty.lexical_difficulty is not None
        else None
    )
    comprehension_fit = (
        _fit(reader.comprehension, difficulty.syntactic_complexity)
        if difficulty.syntactic_complexity is not None
        else None
    )

    concept_density_fit = (
        _fit(reader.background_knowledge, difficulty.concept_density)
        if difficulty.concept_density is not None
        else None
    )
    prerequisite_demand_fit = (
        _fit(reader.background_knowledge, difficulty.prerequisite_demand)
        if difficulty.prerequisite_demand is not None
        else None
    )
    prerequisite_concept_fit, assessed_count = _prerequisite_concept_fit(reader, book)
    knowledge_values = {
        "concept_density_fit": concept_density_fit,
        "prerequisite_demand_fit": prerequisite_demand_fit,
        "prerequisite_concept_fit": prerequisite_concept_fit,
    }
    knowledge_fit, active_knowledge_weights = _renormalized_weighted_mean(
        knowledge_values, config.knowledge_weights
    )
    knowledge_demands = {
        "concept_density_fit": difficulty.concept_density,
        "prerequisite_demand_fit": difficulty.prerequisite_demand,
        "prerequisite_concept_fit": None,
    }
    knowledge_demand, _ = _renormalized_weighted_mean(knowledge_demands, config.knowledge_weights)

    component_values = {
        "topic_fit": topic_fit,
        "vocabulary_fit": vocabulary_fit,
        "knowledge_fit": knowledge_fit,
        "comprehension_fit": comprehension_fit,
    }
    score, active_weights = _renormalized_weighted_mean(component_values, config.component_weights)
    if score is None:
        raise RankingError(f"no ranking components are available for book {book.book_id}")

    components = RankingComponents(**component_values)
    knowledge_components = KnowledgeFitComponents(**knowledge_values)
    diagnostics = MatchingDiagnostics(
        lexical_demand=difficulty.lexical_difficulty,
        knowledge_demand=knowledge_demand,
        syntactic_demand=difficulty.syntactic_complexity,
        assessed_prerequisite_count=assessed_count,
        inferred_prerequisite_count=len(book.concept_profile.prerequisite_concepts),
    )
    unavailable = [name for name in config.component_weights if component_values[name] is None]
    knowledge_weight_coverage = sum(
        weight
        for name, weight in config.knowledge_weights.items()
        if knowledge_values[name] is not None
    )
    component_weight_coverage = sum(
        weight
        for name, weight in config.component_weights.items()
        if component_values[name] is not None
    )
    return RankedBook(
        book_id=book.book_id,
        score=score,
        components=components,
        knowledge_components=knowledge_components,
        knowledge_active_weights=active_knowledge_weights,
        knowledge_weight_coverage=knowledge_weight_coverage,
        active_weights=active_weights,
        component_weight_coverage=component_weight_coverage,
        unavailable_components=unavailable,
        diagnostics=diagnostics,
        reasons=build_reasons(reader, book, components, diagnostics, config.explanation),
        model_version=config.model_version,
        config_version=config.config_version,
        config_hash=loaded_config.content_hash,
        reader_profile_version=reader.profile_version,
        reader_config_version=reader.config_version,
        reader_config_hash=reader.config_hash,
        book_feature_version=book.feature_version,
        book_config_version=book.config_version,
        book_config_hash=book.config_hash,
    )


def rank_books(
    reader: ReaderProfile,
    books: list[BookProfile],
    loaded_config: LoadedRankingConfig,
    limit: int,
) -> RankingResponse:
    """Return deterministic topic-filtered top-K recommendations."""

    if limit < 1:
        raise RankingError("ranking limit must be at least 1")
    config = loaded_config.config
    candidates = books
    if config.filter_topic_candidates:
        candidates = [
            book
            for book in books
            if book.concept_profile.topic_distribution.get(reader.topic_id, 0.0) > 0
        ]
    items = [score_book_fit(reader, book, loaded_config) for book in candidates]
    items.sort(key=lambda item: (-item.score, item.book_id))
    return RankingResponse(
        topic_id=reader.topic_id,
        items=items[:limit],
        model_version=config.model_version,
        config_version=config.config_version,
        config_hash=loaded_config.content_hash,
        reader_profile_version=reader.profile_version,
        reader_config_version=reader.config_version,
        reader_config_hash=reader.config_hash,
    )
