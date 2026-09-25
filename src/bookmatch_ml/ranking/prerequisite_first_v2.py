"""Production-safe prerequisite-first ranking with no scalar composite score."""

from typing import Literal

from pydantic import Field, field_validator

from bookmatch_ml.config import LoadedRankingV2Config
from bookmatch_ml.ranking.concept_readiness_experiments import (
    AcceptedGraphProjection,
    AcceptedPrerequisiteEdge,
    prerequisite_ancestors,
)
from bookmatch_ml.schemas import MatchingBookProfile, MatchingReaderProfile, StrictModel

RANKING_V2_MODEL = "rank-prerequisite-first-v2"
RankingV2Availability = Literal["personalizable", "concept_only", "evidence_unavailable"]


class RankingV2Error(ValueError):
    """Production ranking-v2 inputs violate the versioned contract."""


class RankingV2TargetUnavailableError(RankingV2Error):
    """A requested target book cannot receive a personalized v2 rank."""


class PrerequisiteFirstBookProfile(StrictModel):
    """Book-level v2 input with reviewed transitive prerequisites already materialized."""

    book_id: str = Field(min_length=1)
    topic_id: str = Field(min_length=1)
    covered_concepts: list[str]
    inferred_prerequisites: list[str]
    feature_version: str = Field(min_length=1)
    config_version: str = Field(min_length=1)
    config_hash: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")

    @field_validator("covered_concepts", "inferred_prerequisites")
    @classmethod
    def concepts_must_be_unique_and_sorted(cls, value: list[str]) -> list[str]:
        if value != sorted(set(value)) or any(not concept.strip() for concept in value):
            raise ValueError("rank-v2 concepts must be non-blank, unique, and sorted")
        return value


class RankingV2Item(StrictModel):
    """One personalizable result with both canonical axes and no composite score."""

    book_id: str
    rank: int = Field(ge=1)
    availability_status: Literal["personalizable"]
    prerequisite_readiness: float = Field(ge=0, le=1)
    prerequisite_assessed_count: int = Field(ge=1)
    prerequisite_total_count: int = Field(ge=1)
    prerequisite_coverage: float = Field(gt=0, le=1)
    direct_learning_opportunity: float | None = Field(default=None, ge=0, le=1)
    direct_assessed_count: int = Field(ge=0)
    direct_total_count: int = Field(ge=1)
    direct_coverage: float = Field(ge=0, le=1)
    covered_concepts: list[str]
    inferred_prerequisites: list[str]
    reasons: list[str] = Field(min_length=2)
    model_version: Literal["rank-prerequisite-first-v2"]
    book_feature_version: str
    book_config_version: str
    book_config_hash: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")


class RankingV2Diagnostics(StrictModel):
    """Pool counts and explicit Top-K shortage without fallback filling."""

    requested_limit: int = Field(ge=1)
    returned_count: int = Field(ge=0)
    topic_candidate_count: int = Field(ge=0)
    personalizable_count: int = Field(ge=0)
    concept_only_count: int = Field(ge=0)
    evidence_unavailable_count: int = Field(ge=0)
    fallback_count: int = Field(ge=0)
    personalized_candidate_shortage: int = Field(ge=0)


class RankingV2Response(StrictModel):
    """Versioned response whose ordering is represented by rank and two axes."""

    topic_id: str
    items: list[RankingV2Item]
    diagnostics: RankingV2Diagnostics
    model_version: Literal["rank-prerequisite-first-v2"]
    config_version: Literal["ranking-v2-config-v1"]
    config_hash: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    concept_graph_version: str
    concept_graph_hash: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    graph_review_version: str
    graph_review_hash: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    reader_profile_version: str
    reader_config_version: str
    reader_config_hash: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")


class _CalculatedBook(StrictModel):
    """Private calculated state retained before filtering and ordering."""

    profile: PrerequisiteFirstBookProfile
    availability_status: RankingV2Availability
    prerequisite_readiness: float | None = Field(default=None, ge=0, le=1)
    prerequisite_assessed_count: int = Field(ge=0)
    prerequisite_total_count: int = Field(ge=0)
    prerequisite_coverage: float | None = Field(default=None, ge=0, le=1)
    direct_learning_opportunity: float | None = Field(default=None, ge=0, le=1)
    direct_assessed_count: int = Field(ge=0)
    direct_total_count: int = Field(ge=0)
    direct_coverage: float | None = Field(default=None, ge=0, le=1)


def build_ranking_v2_projection(
    loaded_config: LoadedRankingV2Config,
) -> AcceptedGraphProjection:
    """Build the server-owned accepted graph projection from the frozen v2 policy."""

    config = loaded_config.config
    return AcceptedGraphProjection(
        graph_version=config.concept_graph_version,
        graph_hash=config.concept_graph_hash,
        review_config_version=config.graph_review_version,
        review_config_hash=config.graph_review_hash,
        nodes=config.nodes,
        edges=[
            AcceptedPrerequisiteEdge(
                topic=edge.topic,
                prerequisite=edge.prerequisite,
                dependent=edge.dependent,
            )
            for edge in config.accepted_edges
        ],
    )


def build_prerequisite_first_book_profiles(
    books: list[MatchingBookProfile],
    projection: AcceptedGraphProjection,
) -> list[PrerequisiteFirstBookProfile]:
    """Materialize accepted transitive prerequisites without changing the source adapter."""

    book_ids = [book.book_id for book in books]
    if len(book_ids) != len(set(book_ids)):
        raise RankingV2Error("duplicate rank-v2 candidate book_id")
    profiles: list[PrerequisiteFirstBookProfile] = []
    for book in sorted(books, key=lambda item: item.book_id):
        topics = sorted(topic for topic, score in book.topic_distribution.items() if score > 0)
        if len(topics) != 1:
            raise RankingV2Error(
                f"rank-v2 candidate must belong to exactly one positive topic: {book.book_id}"
            )
        topic = topics[0]
        if topic not in projection.nodes:
            raise RankingV2Error(f"rank-v2 candidate topic is absent from graph: {topic}")
        covered = {item.concept for item in book.covered_concepts}
        unknown = covered - set(projection.nodes[topic])
        if unknown:
            raise RankingV2Error(
                f"rank-v2 candidate has unknown covered concepts: {book.book_id}: {sorted(unknown)}"
            )
        prerequisites: set[str] = set()
        for concept in covered:
            prerequisites.update(
                prerequisite_ancestors(projection, topic, concept, transitive=True)
            )
        profiles.append(
            PrerequisiteFirstBookProfile(
                book_id=book.book_id,
                topic_id=topic,
                covered_concepts=sorted(covered),
                inferred_prerequisites=sorted(prerequisites),
                feature_version=book.feature_version,
                config_version=book.config_version,
                config_hash=book.config_hash,
            )
        )
    return profiles


def _mean(values: list[float]) -> float | None:
    """Return a mean only when at least one assessed value exists."""

    return sum(values) / len(values) if values else None


def _calculate(
    reader: MatchingReaderProfile,
    book: PrerequisiteFirstBookProfile,
) -> _CalculatedBook:
    """Calculate separate prerequisite and opportunity axes with coverage."""

    mastery = {item.concept_id: item.score for item in reader.concept_readiness}
    assessed_prerequisites = [
        mastery[concept] for concept in book.inferred_prerequisites if concept in mastery
    ]
    assessed_covered = [mastery[concept] for concept in book.covered_concepts if concept in mastery]
    prerequisite_readiness = _mean(assessed_prerequisites)
    direct_mastery = _mean(assessed_covered)
    if not book.covered_concepts:
        status: RankingV2Availability = "evidence_unavailable"
    elif book.inferred_prerequisites and assessed_prerequisites:
        status = "personalizable"
    else:
        status = "concept_only"
    return _CalculatedBook(
        profile=book,
        availability_status=status,
        prerequisite_readiness=prerequisite_readiness,
        prerequisite_assessed_count=len(assessed_prerequisites),
        prerequisite_total_count=len(book.inferred_prerequisites),
        prerequisite_coverage=(
            len(assessed_prerequisites) / len(book.inferred_prerequisites)
            if book.inferred_prerequisites
            else None
        ),
        direct_learning_opportunity=(1 - direct_mastery if direct_mastery is not None else None),
        direct_assessed_count=len(assessed_covered),
        direct_total_count=len(book.covered_concepts),
        direct_coverage=(
            len(assessed_covered) / len(book.covered_concepts) if book.covered_concepts else None
        ),
    )


def _ordering_key(book: _CalculatedBook) -> tuple:
    """Apply the exact prerequisite-first policy without tolerance or arithmetic fusion."""

    if book.prerequisite_readiness is None:
        raise RankingV2Error(f"personalizable book lacks readiness: {book.profile.book_id}")
    opportunity = book.direct_learning_opportunity
    return (
        -book.prerequisite_readiness,
        opportunity is None,
        -opportunity if opportunity is not None else 0.0,
        book.profile.book_id,
    )


def _reasons(book: _CalculatedBook) -> list[str]:
    """Explain only assessed evidence, coverage, and remaining uncertainty."""

    reasons = [
        f"선행 개념 {book.prerequisite_total_count}개 중 "
        f"{book.prerequisite_assessed_count}개가 평가되었고 평균 준비도는 "
        f"{book.prerequisite_readiness:.2f}입니다."
    ]
    if book.direct_learning_opportunity is None:
        reasons.append("다루는 개념의 readiness가 평가되지 않아 학습 기회는 알 수 없습니다.")
    else:
        reasons.append(
            f"책의 개념 {book.direct_total_count}개 중 {book.direct_assessed_count}개가 "
            f"평가되었고 학습 기회는 {book.direct_learning_opportunity:.2f}입니다."
        )
    if book.prerequisite_coverage is not None and book.prerequisite_coverage < 1:
        reasons.append("평가되지 않은 선행 개념이 있어 준비도 판단 범위가 제한됩니다.")
    if book.direct_coverage is not None and book.direct_coverage < 1:
        reasons.append("평가되지 않은 책의 개념이 있어 학습 기회 판단 범위가 제한됩니다.")
    return reasons


def _to_item(book: _CalculatedBook, rank: int) -> RankingV2Item:
    """Convert one eligible calculated book into the public internal result."""

    if book.prerequisite_readiness is None or book.prerequisite_coverage is None:
        raise RankingV2Error(
            f"personalizable book lacks prerequisite values: {book.profile.book_id}"
        )
    if book.direct_coverage is None:
        raise RankingV2Error(f"personalizable book lacks direct coverage: {book.profile.book_id}")
    return RankingV2Item(
        book_id=book.profile.book_id,
        rank=rank,
        availability_status="personalizable",
        prerequisite_readiness=book.prerequisite_readiness,
        prerequisite_assessed_count=book.prerequisite_assessed_count,
        prerequisite_total_count=book.prerequisite_total_count,
        prerequisite_coverage=book.prerequisite_coverage,
        direct_learning_opportunity=book.direct_learning_opportunity,
        direct_assessed_count=book.direct_assessed_count,
        direct_total_count=book.direct_total_count,
        direct_coverage=book.direct_coverage,
        covered_concepts=book.profile.covered_concepts,
        inferred_prerequisites=book.profile.inferred_prerequisites,
        reasons=_reasons(book),
        model_version=RANKING_V2_MODEL,
        book_feature_version=book.profile.feature_version,
        book_config_version=book.profile.config_version,
        book_config_hash=book.profile.config_hash,
    )


def rank_prerequisite_first_v2(
    reader: MatchingReaderProfile,
    books: list[PrerequisiteFirstBookProfile],
    loaded_config: LoadedRankingV2Config,
    *,
    limit: int,
    book_id: str | None = None,
) -> RankingV2Response:
    """Return at most Top-K personalizable books and never fill from fallback pools."""

    if limit < 1:
        raise RankingV2Error("ranking-v2 limit must be at least 1")
    ids = [book.book_id for book in books]
    if len(ids) != len(set(ids)):
        raise RankingV2Error("duplicate rank-v2 candidate book_id")
    if book_id is not None and book_id not in set(ids):
        raise RankingV2Error("rank-v2 target book is not a candidate")
    topic_books = [book for book in books if book.topic_id == reader.topic_id]
    if book_id is not None and not any(book.book_id == book_id for book in topic_books):
        raise RankingV2Error("rank-v2 target book topic differs from reader topic")

    calculated = [_calculate(reader, book) for book in topic_books]
    personalizable = sorted(
        [book for book in calculated if book.availability_status == "personalizable"],
        key=_ordering_key,
    )
    counts = {
        status: sum(book.availability_status == status for book in calculated)
        for status in ("personalizable", "concept_only", "evidence_unavailable")
    }
    if book_id is None:
        selected = [(book, rank) for rank, book in enumerate(personalizable[:limit], start=1)]
    else:
        target = next(book for book in calculated if book.profile.book_id == book_id)
        if target.availability_status != "personalizable":
            raise RankingV2TargetUnavailableError(
                f"rank-v2 target book is not personalizable: {target.availability_status}"
            )
        selected = [
            (target, index)
            for index, book in enumerate(personalizable, start=1)
            if book.profile.book_id == book_id
        ]
    items = [_to_item(book, rank) for book, rank in selected]
    requested_count = 1 if book_id is not None else limit
    shortage = max(0, requested_count - len(items))
    config = loaded_config.config
    return RankingV2Response(
        topic_id=reader.topic_id,
        items=items,
        diagnostics=RankingV2Diagnostics(
            requested_limit=requested_count,
            returned_count=len(items),
            topic_candidate_count=len(topic_books),
            personalizable_count=counts["personalizable"],
            concept_only_count=counts["concept_only"],
            evidence_unavailable_count=counts["evidence_unavailable"],
            fallback_count=counts["concept_only"] + counts["evidence_unavailable"],
            personalized_candidate_shortage=shortage,
        ),
        model_version=config.model_version,
        config_version=config.config_version,
        config_hash=loaded_config.content_hash,
        concept_graph_version=config.concept_graph_version,
        concept_graph_hash=config.concept_graph_hash,
        graph_review_version=config.graph_review_version,
        graph_review_hash=config.graph_review_hash,
        reader_profile_version=reader.profile_version,
        reader_config_version=reader.config_version,
        reader_config_hash=reader.config_hash,
    )
