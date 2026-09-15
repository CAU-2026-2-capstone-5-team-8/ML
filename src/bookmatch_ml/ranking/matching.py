"""Configurable score decomposition with missing-component renormalization."""

from bookmatch_ml.config import LoadedRankingConfig
from bookmatch_ml.ranking.explanation import build_reasons
from bookmatch_ml.schemas import (
    BookProfile,
    KnowledgeFitComponents,
    MatchingBookProfile,
    MatchingConcept,
    MatchingConceptReadiness,
    MatchingDiagnostics,
    MatchingReaderProfile,
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
    reader: MatchingReaderProfile,
    book: MatchingBookProfile,
) -> tuple[float | None, int]:
    reader_concepts = {item.concept_id: item.score for item in reader.concept_readiness}
    matched = [
        prerequisite
        for prerequisite in book.prerequisite_concepts
        if prerequisite.concept in reader_concepts
    ]
    total_weight = sum(item.weight for item in matched)
    if not matched or total_weight == 0:
        return None, 0
    score = sum(reader_concepts[item.concept] * item.weight for item in matched) / total_weight
    return score, len(matched)


def to_matching_reader_profile(reader: ReaderProfile) -> MatchingReaderProfile:
    """Project the full assessment result onto the stable matching input."""

    return MatchingReaderProfile(
        topic_id=reader.topic_id,
        vocabulary=reader.vocabulary,
        background_knowledge=reader.background_knowledge,
        comprehension=reader.comprehension,
        concept_readiness=[
            MatchingConceptReadiness(concept_id=item.concept_id, score=item.score)
            for item in reader.concept_readiness
        ],
        profile_version=reader.profile_version,
        config_version=reader.config_version,
        config_hash=reader.config_hash,
    )


def to_matching_book_profile(book: BookProfile) -> MatchingBookProfile:
    """Project the batch book profile onto the stable matching input."""

    difficulty = book.difficulty_profile
    return MatchingBookProfile(
        book_id=book.book_id,
        topic_distribution=book.concept_profile.topic_distribution,
        covered_concepts=[
            MatchingConcept(concept=item.concept, weight=item.weight)
            for item in book.concept_profile.covered_concepts
        ],
        prerequisite_concepts=[
            MatchingConcept(concept=item.concept, weight=item.weight)
            for item in book.concept_profile.prerequisite_concepts
        ],
        lexical_difficulty=difficulty.lexical_difficulty,
        syntactic_complexity=difficulty.syntactic_complexity,
        concept_density=difficulty.concept_density,
        prerequisite_demand=difficulty.prerequisite_demand,
        feature_version=book.feature_version,
        config_version=book.config_version,
        config_hash=book.config_hash,
    )


def score_matching_book_fit(
    reader: MatchingReaderProfile,
    book: MatchingBookProfile,
    loaded_config: LoadedRankingConfig,
) -> RankedBook:
    """Calculate one fit score from explicit integration-safe inputs."""

    config = loaded_config.config
    topic_distribution = book.topic_distribution
    topic_fit = topic_distribution.get(reader.topic_id, 0.0) if topic_distribution else None
    vocabulary_fit = (
        _fit(reader.vocabulary, book.lexical_difficulty)
        if book.lexical_difficulty is not None
        else None
    )
    comprehension_fit = (
        _fit(reader.comprehension, book.syntactic_complexity)
        if book.syntactic_complexity is not None
        else None
    )

    concept_density_fit = (
        _fit(reader.background_knowledge, book.concept_density)
        if book.concept_density is not None
        else None
    )
    prerequisite_demand_fit = (
        _fit(reader.background_knowledge, book.prerequisite_demand)
        if book.prerequisite_demand is not None
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
        "concept_density_fit": book.concept_density,
        "prerequisite_demand_fit": book.prerequisite_demand,
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
        lexical_demand=book.lexical_difficulty,
        knowledge_demand=knowledge_demand,
        syntactic_demand=book.syntactic_complexity,
        assessed_prerequisite_count=assessed_count,
        inferred_prerequisite_count=len(book.prerequisite_concepts),
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


def rank_matching_books(
    reader: MatchingReaderProfile,
    books: list[MatchingBookProfile],
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
            book for book in books if book.topic_distribution.get(reader.topic_id, 0.0) > 0
        ]
    items = [score_matching_book_fit(reader, book, loaded_config) for book in candidates]
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


def score_book_fit(
    reader: ReaderProfile,
    book: BookProfile,
    loaded_config: LoadedRankingConfig,
) -> RankedBook:
    """Calculate one fit score from full internal profiles."""

    return score_matching_book_fit(
        to_matching_reader_profile(reader),
        to_matching_book_profile(book),
        loaded_config,
    )


def rank_books(
    reader: ReaderProfile,
    books: list[BookProfile],
    loaded_config: LoadedRankingConfig,
    limit: int,
) -> RankingResponse:
    """Return deterministic topic-filtered top-K recommendations."""

    return rank_matching_books(
        to_matching_reader_profile(reader),
        [to_matching_book_profile(book) for book in books],
        loaded_config,
        limit,
    )
