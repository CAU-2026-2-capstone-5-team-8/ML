"""Two independent concept axes, missing-mastery coverage, and batch comparison."""

from collections.abc import Iterable
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from bookmatch_ml.concept_v2.profile import (
    BookConceptProfileV2,
    LoadedConceptMatchingConfig,
)
from bookmatch_ml.config import LoadedRankingConfig
from bookmatch_ml.ranking.matching import score_book_fit
from bookmatch_ml.schemas import BookProfile, ReaderProfile


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class ConceptAxis(_Strict):
    score: float | None = Field(ge=0, le=1)
    assessment_coverage: float | None = Field(ge=0, le=1)
    total_weight: float = Field(ge=0)
    assessed_weight: float = Field(ge=0)
    semantics: Literal["calculated", "unassessed", "not_applicable"]


class ConceptMatchingItem(_Strict):
    book_id: str
    title: str
    topic_id: str
    status: Literal["eligible", "challenge_candidate", "insufficient_evidence"]
    readiness: ConceptAxis
    opportunity: ConceptAxis
    mastered_prerequisites: list[str]
    weak_prerequisites: list[str]
    unknown_prerequisites: list[str]
    already_known_concepts: list[str]
    learning_candidate_concepts: list[str]
    unknown_mastery_concepts: list[str]
    evidence_limitations: list[str]
    v1_score: float | None = None
    v1_component_weight_coverage: float | None = None
    text_complexity_available: bool | None = None
    lexical_difficulty: float | None = None
    syntactic_complexity: float | None = None
    profile: BookConceptProfileV2


class ConceptMatchingReport(_Strict):
    topic_id: str
    candidate_count: int
    items: list[ConceptMatchingItem]
    model_version: str
    matching_config_version: str
    matching_config_hash: str
    graph_version: str
    graph_hash: str
    feature_config_hash: str
    reader_profile_version: str
    reader_config_hash: str
    input_hashes: dict[str, str]


def _axis(
    weights: Iterable[tuple[str, float]], mastery: dict[str, float], opportunity: bool
) -> ConceptAxis:
    terms = list(weights)
    total = sum(weight for _, weight in terms)
    if total == 0:
        return ConceptAxis(
            score=None,
            assessment_coverage=None,
            total_weight=0,
            assessed_weight=0,
            semantics="not_applicable",
        )
    assessed = [(concept, weight) for concept, weight in terms if concept in mastery]
    assessed_weight = sum(weight for _, weight in assessed)
    if assessed_weight == 0:
        return ConceptAxis(
            score=None,
            assessment_coverage=0,
            total_weight=total,
            assessed_weight=0,
            semantics="unassessed",
        )
    score = (
        sum(
            weight * ((1 - mastery[concept]) if opportunity else mastery[concept])
            for concept, weight in assessed
        )
        / assessed_weight
    )
    return ConceptAxis(
        score=score,
        assessment_coverage=assessed_weight / total,
        total_weight=total,
        assessed_weight=assessed_weight,
        semantics="calculated",
    )


def match_book_concepts(
    reader: ReaderProfile,
    profile: BookConceptProfileV2,
    matching: LoadedConceptMatchingConfig,
    v1_book: BookProfile | None = None,
    v1_ranking: LoadedRankingConfig | None = None,
) -> ConceptMatchingItem:
    if reader.topic_id != profile.topic_id:
        raise ValueError("reader and book concept profile topics differ")
    if (v1_book is None) != (v1_ranking is None):
        raise ValueError("v1 book and ranking config must be supplied together")
    if v1_book is not None and v1_book.book_id != profile.book_id:
        raise ValueError("v1 book_id differs from v2 book_id")
    mastery = {item.concept_id: item.score for item in reader.concept_readiness}
    if len(mastery) != len(reader.concept_readiness):
        raise ValueError("duplicate reader concept_readiness")
    prerequisites = profile.prerequisite_requirements
    covered = profile.covered_concepts
    readiness = _axis(((item.concept_id, item.weight) for item in prerequisites), mastery, False)
    opportunity = _axis(
        ((item.concept_id, item.coverage_weight) for item in covered), mastery, True
    )
    policy = matching.config.policy
    limitations: list[str] = []
    if readiness.semantics == "not_applicable":
        limitations.append(
            "No external or late prerequisite candidate was projected from this TOC."
        )
    elif readiness.assessment_coverage < policy.minimum_readiness_assessment_coverage:
        limitations.append(
            "Prerequisite mastery assessment coverage is below the experimental threshold."
        )
    if opportunity.semantics == "not_applicable":
        limitations.append("No configured concept was mapped from this TOC.")
    elif opportunity.assessment_coverage < policy.minimum_opportunity_assessment_coverage:
        limitations.append(
            "Book-concept mastery assessment coverage is below the experimental threshold."
        )
    if profile.diagnostics.unmatched_toc_entries or profile.diagnostics.ambiguous_toc_entries:
        limitations.append(
            "Some TOC entries were unmatched or ambiguous under the configured lexicon."
        )
    if (
        (
            readiness.semantics != "not_applicable"
            and readiness.assessment_coverage < policy.minimum_readiness_assessment_coverage
        )
        or opportunity.assessment_coverage is None
        or opportunity.assessment_coverage < policy.minimum_opportunity_assessment_coverage
    ):
        status = "insufficient_evidence"
    elif readiness.score is not None and readiness.score < policy.minimum_readiness_score:
        status = "challenge_candidate"
    else:
        status = "eligible"

    v1 = score_book_fit(reader, v1_book, v1_ranking) if v1_book is not None else None
    difficulty = v1_book.difficulty_profile if v1_book is not None else None
    return ConceptMatchingItem(
        book_id=profile.book_id,
        title=profile.title,
        topic_id=profile.topic_id,
        status=status,
        readiness=readiness,
        opportunity=opportunity,
        mastered_prerequisites=sorted(
            item.concept_id
            for item in prerequisites
            if mastery.get(item.concept_id, -1) >= policy.known_mastery_threshold
        ),
        weak_prerequisites=sorted(
            item.concept_id
            for item in prerequisites
            if item.concept_id in mastery
            and mastery[item.concept_id] < policy.learning_mastery_threshold
        ),
        unknown_prerequisites=sorted(
            item.concept_id for item in prerequisites if item.concept_id not in mastery
        ),
        already_known_concepts=sorted(
            item.concept_id
            for item in covered
            if mastery.get(item.concept_id, -1) >= policy.known_mastery_threshold
        ),
        learning_candidate_concepts=sorted(
            item.concept_id
            for item in covered
            if item.concept_id in mastery
            and mastery[item.concept_id] < policy.learning_mastery_threshold
        ),
        unknown_mastery_concepts=sorted(
            item.concept_id for item in covered if item.concept_id not in mastery
        ),
        evidence_limitations=limitations,
        v1_score=v1.score if v1 else None,
        v1_component_weight_coverage=v1.component_weight_coverage if v1 else None,
        text_complexity_available=(difficulty.analyzed_document_count > 0) if difficulty else None,
        lexical_difficulty=difficulty.lexical_difficulty if difficulty else None,
        syntactic_complexity=difficulty.syntactic_complexity if difficulty else None,
        profile=profile,
    )


def order_matching_items(items: list[ConceptMatchingItem]) -> list[ConceptMatchingItem]:
    """Experimental display order: evidence/readiness gate, then opportunity; retain all books."""

    priority = {"eligible": 0, "challenge_candidate": 1, "insufficient_evidence": 2}
    return sorted(
        items,
        key=lambda item: (
            priority[item.status],
            -(item.opportunity.score if item.opportunity.score is not None else -1),
            -(item.readiness.score if item.readiness.score is not None else -1),
            item.book_id,
        ),
    )
