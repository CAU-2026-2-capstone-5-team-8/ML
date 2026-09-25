"""Deterministic multi-reader concept-ranking diagnostics outside production ranking."""

import hashlib
import json
from itertools import combinations
from pathlib import Path
from typing import Literal

import yaml
from pydantic import Field, ValidationError, field_validator, model_validator

from bookmatch_ml.concept_v2.book_evidence_mapping import (
    BookEvidenceConceptMappingReport,
)
from bookmatch_ml.concept_v2.graph import LoadedConceptGraph
from bookmatch_ml.concept_v2.profile import LoadedConceptMatchingConfig
from bookmatch_ml.concept_v2.validation import LoadedConceptGraphReviews
from bookmatch_ml.config import ConfigError, _load_unique_key_yaml
from bookmatch_ml.ranking.concept_readiness_experiments import (
    AcceptedGraphProjection,
    ConceptReadinessBookDiagnostic,
    DiagnosticSnapshot,
    GroupStatistics,
    build_accepted_graph_projection,
    evaluate_concept_readiness_ranking,
)
from bookmatch_ml.schemas import (
    ConceptReadiness,
    ReaderDimensionDetail,
    ReaderProfile,
    StrictModel,
)

EXPERIMENT_VERSION = "multi-reader-concept-ranking-experiment-v1"
SCENARIO_ORDER = ("beginner", "intermediate", "advanced", "uneven")
VARIANT_ORDER = (
    "topic_only",
    "direct_opportunity",
    "direct_prerequisite",
    "transitive_prerequisite",
    "d2_unweighted",
    "e_prerequisite_first",
)


class MultiReaderConceptExperimentError(ValueError):
    """Experiment inputs cannot produce a trustworthy deterministic report."""


class ScenarioDefinition(StrictModel):
    scenario_id: Literal["beginner", "intermediate", "advanced", "uneven"]
    description: str = Field(min_length=1)
    readiness: dict[str, float]

    @field_validator("readiness")
    @classmethod
    def readiness_must_be_nonempty_and_normalized(cls, value: dict[str, float]) -> dict[str, float]:
        if not value or any(not concept.strip() for concept in value):
            raise ValueError("scenario readiness requires non-blank canonical concept IDs")
        if any(score < 0 or score > 1 for score in value.values()):
            raise ValueError("scenario readiness scores must be in [0, 1]")
        return value


class FrozenHumanPair(StrictModel):
    pair_id: str = Field(min_length=1)
    topic_id: str = Field(min_length=1)
    left_book_id: str = Field(min_length=1)
    right_book_id: str = Field(min_length=1)
    human_preference: Literal["A", "B", "tie", "not_judgable"]


class MultiReaderConceptExperimentConfig(StrictModel):
    config_version: Literal["multi-reader-concept-ranking-config-v1"]
    experiment_version: Literal["multi-reader-concept-ranking-experiment-v1"]
    primary_lexicographic_tolerance: float = Field(ge=0, le=1)
    lexicographic_tolerance_sensitivity: list[float] = Field(min_length=1)
    reviewed_report_hash: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    expected_legacy_agreement: dict[Literal["d1", "d2", "direct_only", "prerequisite_only"], int]
    scenarios: dict[str, list[ScenarioDefinition]]
    frozen_human_pairs: list[FrozenHumanPair]

    @model_validator(mode="after")
    def validate_experiment_matrix(self) -> "MultiReaderConceptExperimentConfig":
        if set(self.scenarios) != {"linear-algebra", "operating-systems"}:
            raise ValueError("scenarios must cover exactly the two experiment topics")
        for topic, scenarios in self.scenarios.items():
            ids = [scenario.scenario_id for scenario in scenarios]
            if tuple(ids) != SCENARIO_ORDER:
                raise ValueError(f"{topic} scenarios must use the documented order")
        tolerances = self.lexicographic_tolerance_sensitivity
        if tolerances != sorted(set(tolerances)):
            raise ValueError("lexicographic tolerances must be unique and sorted")
        if self.primary_lexicographic_tolerance not in tolerances:
            raise ValueError("primary tolerance must appear in sensitivity values")
        expected = {"d1", "d2", "direct_only", "prerequisite_only"}
        if set(self.expected_legacy_agreement) != expected:
            raise ValueError("expected legacy agreement keys are incomplete")
        pair_ids = [pair.pair_id for pair in self.frozen_human_pairs]
        if len(pair_ids) != 10 or len(pair_ids) != len(set(pair_ids)):
            raise ValueError("frozen human review must contain ten unique pairs")
        return self


class LoadedMultiReaderConceptExperimentConfig(StrictModel):
    config: MultiReaderConceptExperimentConfig
    content_hash: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")


class OrderingGroup(StrictModel):
    group_index: int = Field(ge=1)
    available: bool
    group_key: str
    book_ids: list[str] = Field(min_length=1)


class VariantOrdering(StrictModel):
    variant_id: Literal[
        "topic_only",
        "direct_opportunity",
        "direct_prerequisite",
        "transitive_prerequisite",
        "d2_unweighted",
        "e_prerequisite_first",
    ]
    tolerance: float | None = Field(default=None, ge=0, le=1)
    statistics: GroupStatistics
    groups: list[OrderingGroup]


class ScenarioProfile(StrictModel):
    scenario_id: str
    description: str
    reader_profile_hash: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    reader_profile: ReaderProfile


class ScenarioEvaluation(StrictModel):
    topic_id: str
    scenario: ScenarioProfile
    direct_coverage_distribution: dict[str, int]
    direct_prerequisite_coverage_distribution: dict[str, int]
    transitive_prerequisite_coverage_distribution: dict[str, int]
    diagnostics: list[ConceptReadinessBookDiagnostic]
    variants: list[VariantOrdering]
    lexicographic_sensitivity: list[VariantOrdering]


class ProfileBookState(StrictModel):
    scenario_id: str
    e_group_index: int
    variant_group_indices: dict[str, int]
    direct_learning_opportunity: float | None = Field(default=None, ge=0, le=1)
    direct_coverage: float | None = Field(default=None, ge=0, le=1)
    direct_prerequisite_readiness: float | None = Field(default=None, ge=0, le=1)
    direct_prerequisite_coverage: float | None = Field(default=None, ge=0, le=1)
    prerequisite_readiness: float | None = Field(default=None, ge=0, le=1)
    prerequisite_coverage: float | None = Field(default=None, ge=0, le=1)
    d2_unweighted: float | None = Field(default=None, ge=0, le=1)


class CrossProfileCase(StrictModel):
    topic_id: str
    case_type: Literal[
        "beginner_to_intermediate_improvement",
        "advanced_opportunity_reduction",
        "uneven_prerequisite_weakness",
        "stable_position",
    ]
    observed: bool
    book_id: str | None = None
    title: str | None = None
    covered_concepts: list[str]
    direct_prerequisites: list[str]
    transitive_prerequisites: list[str]
    states: list[ProfileBookState]
    explanation: str


class EvidenceGapSummary(StrictModel):
    topic_id: str
    no_concept_evidence_count: int = Field(ge=0)
    book_ids: list[str]


class VariantComparison(StrictModel):
    topic_id: str
    scenario_id: str
    comparable_pair_count: int = Field(ge=0)
    prerequisite_vs_d2_agreement_count: int = Field(ge=0)
    prerequisite_vs_d2_agreement_rate: float | None = Field(default=None, ge=0, le=1)
    prerequisite_vs_e_agreement_count: int = Field(ge=0)
    prerequisite_vs_e_agreement_rate: float | None = Field(default=None, ge=0, le=1)
    d2_vs_e_agreement_count: int = Field(ge=0)
    d2_vs_e_agreement_rate: float | None = Field(default=None, ge=0, le=1)
    prerequisite_tie_pair_count: int = Field(ge=0)
    e_direct_tiebreak_pair_count: int = Field(ge=0)
    d2_opportunity_override_count: int = Field(ge=0)
    representative_differences: list[str]


class HumanPairEvaluation(StrictModel):
    pair_id: str
    topic_id: str
    left_book_id: str
    right_book_id: str
    human_preference: Literal["A", "B", "tie", "not_judgable"]
    d1_prediction: Literal["A", "B", "tie", "not_comparable"]
    d2_prediction: Literal["A", "B", "tie", "not_comparable"]
    direct_only_prediction: Literal["A", "B", "tie", "not_comparable"]
    prerequisite_only_prediction: Literal["A", "B", "tie", "not_comparable"]
    e_prediction: Literal["A", "B", "tie", "not_comparable"]


class AgreementSummary(StrictModel):
    variant_id: str
    topic_id: str | None = None
    comparable_pair_count: int = Field(ge=0)
    agreement_count: int = Field(ge=0)
    agreement_rate: float | None = Field(default=None, ge=0, le=1)


class BehavioralObservation(StrictModel):
    topic_id: str
    scenario_id: str
    question: str
    representative_book_id: str | None = None
    observation: str


class MultiReaderConceptRankingReport(StrictModel):
    experiment_version: Literal["multi-reader-concept-ranking-experiment-v1"]
    config_version: str
    config_hash: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    reviewed_report_hash: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    mapping_report_hash: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    input_hashes: dict[str, str]
    production_ranking_modified: Literal[False]
    occurrence_count_used_as_weight: Literal[False]
    prose_difficulty_used: Literal[False]
    unknown_readiness_treated_as_zero: Literal[False]
    accepted_graph: AcceptedGraphProjection
    primary_lexicographic_tolerance: float = Field(ge=0, le=1)
    scenarios: list[ScenarioEvaluation]
    cross_profile_cases: list[CrossProfileCase]
    behavioral_observations: list[BehavioralObservation]
    evidence_gaps: list[EvidenceGapSummary]
    variant_comparisons: list[VariantComparison]
    human_pair_evaluations: list[HumanPairEvaluation]
    human_pair_agreement: list[AgreementSummary]


def load_multi_reader_concept_experiment_config(
    path: Path,
) -> LoadedMultiReaderConceptExperimentConfig:
    """Load the strict multi-reader experiment config and retain its byte hash."""

    try:
        content = path.read_bytes()
    except OSError as exc:
        raise ConfigError(f"cannot read multi-reader experiment config: {path}: {exc}") from exc
    try:
        payload = _load_unique_key_yaml(content)
        config = MultiReaderConceptExperimentConfig.model_validate(payload)
    except (yaml.YAMLError, ValidationError, ValueError) as exc:
        raise ConfigError(f"invalid multi-reader experiment config: {path}: {exc}") from exc
    return LoadedMultiReaderConceptExperimentConfig(
        config=config,
        content_hash=f"sha256:{hashlib.sha256(content).hexdigest()}",
    )


def _sha256_payload(payload: object) -> str:
    content = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return f"sha256:{hashlib.sha256(content.encode()).hexdigest()}"


def _build_scenario_profile(
    topic: str,
    scenario: ScenarioDefinition,
    *,
    config_hash: str,
    config_version: str,
) -> ScenarioProfile:
    readiness = dict(sorted(scenario.readiness.items()))
    mean = sum(readiness.values()) / len(readiness)
    details = [
        ReaderDimensionDetail(
            question_type=question_type,
            score=mean,
            response_count=1,
            earned_weight=mean,
            available_weight=1.0,
        )
        for question_type in ("vocabulary", "background_knowledge", "comprehension")
    ]
    profile = ReaderProfile(
        assessment_id=f"deterministic-scenario:{topic}:{scenario.scenario_id}",
        topic_id=topic,
        vocabulary=mean,
        background_knowledge=mean,
        comprehension=mean,
        dimension_details=details,
        concept_readiness=[
            ConceptReadiness(
                concept_id=concept,
                score=score,
                response_count=1,
                earned_weight=score,
                available_weight=1.0,
            )
            for concept, score in readiness.items()
        ],
        response_count=3,
        profile_version="multi-reader-concept-scenario-v1",
        config_version=config_version,
        config_hash=config_hash,
    )
    return ScenarioProfile(
        scenario_id=scenario.scenario_id,
        description=scenario.description,
        reader_profile_hash=_sha256_payload(profile.model_dump(mode="json")),
        reader_profile=profile,
    )


def build_scenario_profiles(
    loaded_config: LoadedMultiReaderConceptExperimentConfig,
    projection: AcceptedGraphProjection,
) -> list[ScenarioProfile]:
    """Build all configured profiles after exact canonical concept validation."""

    profiles: list[ScenarioProfile] = []
    for topic in sorted(loaded_config.config.scenarios):
        canonical = set(projection.nodes.get(topic, []))
        for scenario in loaded_config.config.scenarios[topic]:
            if set(scenario.readiness) != canonical:
                missing = sorted(canonical - set(scenario.readiness))
                extra = sorted(set(scenario.readiness) - canonical)
                raise MultiReaderConceptExperimentError(
                    f"scenario canonical concepts mismatch: {topic}/{scenario.scenario_id}: "
                    f"missing={missing}, extra={extra}"
                )
            profiles.append(
                _build_scenario_profile(
                    topic,
                    scenario,
                    config_hash=loaded_config.content_hash,
                    config_version=loaded_config.config.config_version,
                )
            )
    return profiles


def _coverage_distribution(
    books: list[ConceptReadinessBookDiagnostic], axis: str
) -> dict[str, int]:
    counts: dict[str, int] = {}
    for book in books:
        selected = getattr(book, axis)
        coverage = selected.assessment_coverage
        key = "not_applicable" if coverage is None else f"{coverage:.6f}"
        counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items()))


def _statistics(groups: list[OrderingGroup], candidate_count: int) -> GroupStatistics:
    available_groups = [group for group in groups if group.available]
    available_count = sum(len(group.book_ids) for group in available_groups)
    sizes = [len(group.book_ids) for group in groups]
    singletons = sum(size == 1 for size in sizes)
    return GroupStatistics(
        candidate_count=candidate_count,
        available_value_count=available_count,
        unavailable_value_count=candidate_count - available_count,
        distinct_available_value_count=len(available_groups),
        distinct_group_count=len(groups),
        largest_tie_group_size=max(sizes, default=0),
        singleton_group_count=singletons,
        uniquely_distinguished_book_count=singletons,
        baseline_largest_tie_reduction=max(0, candidate_count - max(sizes, default=0)),
    )


def _topic_only_ordering(books: list[ConceptReadinessBookDiagnostic]) -> VariantOrdering:
    group = OrderingGroup(
        group_index=1,
        available=True,
        group_key="topic_fit:1.000000000000",
        book_ids=sorted(book.book_id for book in books),
    )
    return VariantOrdering(
        variant_id="topic_only",
        statistics=_statistics([group], len(books)),
        groups=[group],
    )


def _scalar_ordering(
    books: list[ConceptReadinessBookDiagnostic],
    *,
    variant_id: str,
    value_getter,
) -> VariantOrdering:
    values: dict[str, list[str]] = {}
    unavailable: list[str] = []
    for book in books:
        value = value_getter(book)
        if value is None:
            unavailable.append(book.book_id)
        else:
            values.setdefault(f"{value:.12f}", []).append(book.book_id)
    groups: list[OrderingGroup] = []
    for key in sorted(values, key=float, reverse=True):
        groups.append(
            OrderingGroup(
                group_index=len(groups) + 1,
                available=True,
                group_key=key,
                book_ids=sorted(values[key]),
            )
        )
    if unavailable:
        groups.append(
            OrderingGroup(
                group_index=len(groups) + 1,
                available=False,
                group_key="unavailable",
                book_ids=sorted(unavailable),
            )
        )
    return VariantOrdering(
        variant_id=variant_id,
        statistics=_statistics(groups, len(books)),
        groups=groups,
    )


def _lexicographic_ordering(
    books: list[ConceptReadinessBookDiagnostic], tolerance: float
) -> VariantOrdering:
    available = [
        book for book in books if book.transitive_prerequisite_readiness.mastery_mean is not None
    ]
    unavailable = [
        book.book_id
        for book in books
        if book.transitive_prerequisite_readiness.mastery_mean is None
    ]
    available.sort(
        key=lambda book: (
            -book.transitive_prerequisite_readiness.mastery_mean,
            book.book_id,
        )
    )
    bands: list[list[ConceptReadinessBookDiagnostic]] = []
    anchors: list[float] = []
    for book in available:
        readiness = book.transitive_prerequisite_readiness.mastery_mean
        if not bands or anchors[-1] - readiness > tolerance + 1e-12:
            bands.append([book])
            anchors.append(readiness)
        else:
            bands[-1].append(book)

    groups: list[OrderingGroup] = []
    for band_index, band in enumerate(bands, start=1):
        by_opportunity: dict[str, list[str]] = {}
        missing_opportunity: list[str] = []
        readiness_values = [book.transitive_prerequisite_readiness.mastery_mean for book in band]
        for book in band:
            opportunity = book.direct.learning_opportunity
            if opportunity is None:
                missing_opportunity.append(book.book_id)
            else:
                by_opportunity.setdefault(f"{opportunity:.12f}", []).append(book.book_id)
        band_label = (
            f"band:{band_index}:readiness:{min(readiness_values):.12f}.."
            f"{max(readiness_values):.12f}"
        )
        for opportunity in sorted(by_opportunity, key=float, reverse=True):
            groups.append(
                OrderingGroup(
                    group_index=len(groups) + 1,
                    available=True,
                    group_key=f"{band_label}|direct:{opportunity}",
                    book_ids=sorted(by_opportunity[opportunity]),
                )
            )
        if missing_opportunity:
            groups.append(
                OrderingGroup(
                    group_index=len(groups) + 1,
                    available=True,
                    group_key=f"{band_label}|direct:unavailable",
                    book_ids=sorted(missing_opportunity),
                )
            )
    if unavailable:
        groups.append(
            OrderingGroup(
                group_index=len(groups) + 1,
                available=False,
                group_key="unavailable",
                book_ids=sorted(unavailable),
            )
        )
    return VariantOrdering(
        variant_id="e_prerequisite_first",
        tolerance=tolerance,
        statistics=_statistics(groups, len(books)),
        groups=groups,
    )


def _variant_orderings(
    books: list[ConceptReadinessBookDiagnostic], tolerance: float
) -> list[VariantOrdering]:
    return [
        _topic_only_ordering(books),
        _scalar_ordering(
            books,
            variant_id="direct_opportunity",
            value_getter=lambda book: book.direct.learning_opportunity,
        ),
        _scalar_ordering(
            books,
            variant_id="direct_prerequisite",
            value_getter=lambda book: book.direct_prerequisite_readiness.mastery_mean,
        ),
        _scalar_ordering(
            books,
            variant_id="transitive_prerequisite",
            value_getter=lambda book: book.transitive_prerequisite_readiness.mastery_mean,
        ),
        _scalar_ordering(
            books,
            variant_id="d2_unweighted",
            value_getter=lambda book: book.unweighted_combination,
        ),
        _lexicographic_ordering(books, tolerance),
    ]


def build_variant_orderings(
    books: list[ConceptReadinessBookDiagnostic], tolerance: float
) -> list[VariantOrdering]:
    """Build all six deterministic diagnostic orderings for one reader scenario."""

    return _variant_orderings(books, tolerance)


def _prediction(left: float | None, right: float | None) -> str:
    if left is None or right is None:
        return "not_comparable"
    if left > right:
        return "A"
    if right > left:
        return "B"
    return "tie"


def _e_prediction(left: DiagnosticSnapshot, right: DiagnosticSnapshot, tolerance: float) -> str:
    if left.prerequisite_readiness is None or right.prerequisite_readiness is None:
        return "not_comparable"
    difference = left.prerequisite_readiness - right.prerequisite_readiness
    if abs(difference) > tolerance + 1e-12:
        return "A" if difference > 0 else "B"
    return _prediction(left.direct_learning_opportunity, right.direct_learning_opportunity)


def _d1_prediction(left: DiagnosticSnapshot, right: DiagnosticSnapshot) -> str:
    priority = {"eligible": 0, "challenge_candidate": 1, "insufficient_evidence": 2}
    left_priority = priority[left.two_stage_status]
    right_priority = priority[right.two_stage_status]
    if left_priority != right_priority:
        return "A" if left_priority < right_priority else "B"
    direct = _prediction(left.direct_learning_opportunity, right.direct_learning_opportunity)
    if direct != "tie":
        return direct
    return _prediction(left.prerequisite_readiness, right.prerequisite_readiness)


def _direction(left: float, right: float) -> int:
    if left > right:
        return 1
    if right > left:
        return -1
    return 0


def _e_direction(
    left: ConceptReadinessBookDiagnostic,
    right: ConceptReadinessBookDiagnostic,
    tolerance: float,
) -> int:
    left_readiness = left.transitive_prerequisite_readiness.mastery_mean
    right_readiness = right.transitive_prerequisite_readiness.mastery_mean
    difference = left_readiness - right_readiness
    if abs(difference) > tolerance + 1e-12:
        return 1 if difference > 0 else -1
    return _direction(left.direct.learning_opportunity, right.direct.learning_opportunity)


def _variant_comparison(
    topic: str,
    scenario_id: str,
    books: list[ConceptReadinessBookDiagnostic],
    tolerance: float,
) -> VariantComparison:
    comparable = [
        book
        for book in books
        if book.transitive_prerequisite_readiness.mastery_mean is not None
        and book.direct.learning_opportunity is not None
        and book.unweighted_combination is not None
    ]
    counts = {"prerequisite_d2": 0, "prerequisite_e": 0, "d2_e": 0}
    prerequisite_ties = 0
    e_tiebreaks = 0
    d2_overrides = 0
    examples: list[str] = []
    pairs = list(combinations(comparable, 2))
    for left, right in pairs:
        prerequisite = _direction(
            left.transitive_prerequisite_readiness.mastery_mean,
            right.transitive_prerequisite_readiness.mastery_mean,
        )
        d2 = _direction(left.unweighted_combination, right.unweighted_combination)
        e = _e_direction(left, right, tolerance)
        counts["prerequisite_d2"] += prerequisite == d2
        counts["prerequisite_e"] += prerequisite == e
        counts["d2_e"] += d2 == e
        if prerequisite == 0:
            prerequisite_ties += 1
            e_tiebreaks += e != 0
        if prerequisite and d2 == -prerequisite:
            d2_overrides += 1
        if len(examples) < 5 and len({prerequisite, d2, e}) > 1:
            examples.append(
                f"{left.book_id} vs {right.book_id}: prerequisite={prerequisite}, d2={d2}, e={e}"
            )
    total = len(pairs)

    def rate(value: int) -> float | None:
        return value / total if total else None

    return VariantComparison(
        topic_id=topic,
        scenario_id=scenario_id,
        comparable_pair_count=total,
        prerequisite_vs_d2_agreement_count=counts["prerequisite_d2"],
        prerequisite_vs_d2_agreement_rate=rate(counts["prerequisite_d2"]),
        prerequisite_vs_e_agreement_count=counts["prerequisite_e"],
        prerequisite_vs_e_agreement_rate=rate(counts["prerequisite_e"]),
        d2_vs_e_agreement_count=counts["d2_e"],
        d2_vs_e_agreement_rate=rate(counts["d2_e"]),
        prerequisite_tie_pair_count=prerequisite_ties,
        e_direct_tiebreak_pair_count=e_tiebreaks,
        d2_opportunity_override_count=d2_overrides,
        representative_differences=examples,
    )


def _rank_map(ordering: VariantOrdering) -> dict[str, int]:
    return {book_id: group.group_index for group in ordering.groups for book_id in group.book_ids}


def _book_state(
    evaluation: ScenarioEvaluation,
    book: ConceptReadinessBookDiagnostic,
) -> ProfileBookState:
    variant_group_indices = {
        variant.variant_id: _rank_map(variant)[book.book_id] for variant in evaluation.variants
    }
    return ProfileBookState(
        scenario_id=evaluation.scenario.scenario_id,
        e_group_index=variant_group_indices["e_prerequisite_first"],
        variant_group_indices=variant_group_indices,
        direct_learning_opportunity=book.direct.learning_opportunity,
        direct_coverage=book.direct.assessment_coverage,
        direct_prerequisite_readiness=book.direct_prerequisite_readiness.mastery_mean,
        direct_prerequisite_coverage=book.direct_prerequisite_readiness.assessment_coverage,
        prerequisite_readiness=book.transitive_prerequisite_readiness.mastery_mean,
        prerequisite_coverage=book.transitive_prerequisite_readiness.assessment_coverage,
        d2_unweighted=book.unweighted_combination,
    )


def _cross_profile_cases(
    topic: str, evaluations: list[ScenarioEvaluation]
) -> list[CrossProfileCase]:
    by_scenario = {evaluation.scenario.scenario_id: evaluation for evaluation in evaluations}
    diagnostics = {
        scenario_id: {book.book_id: book for book in evaluation.diagnostics}
        for scenario_id, evaluation in by_scenario.items()
    }
    first = diagnostics[SCENARIO_ORDER[0]]
    candidate_ids = sorted(
        book_id
        for book_id, book in first.items()
        if book.transitive_prerequisite_readiness.mastery_mean is not None
        and book.direct.learning_opportunity is not None
    )

    def states(book_id: str) -> list[ProfileBookState]:
        return [
            _book_state(by_scenario[scenario_id], diagnostics[scenario_id][book_id])
            for scenario_id in SCENARIO_ORDER
        ]

    def build(case_type: str, book_id: str | None, explanation: str) -> CrossProfileCase:
        if book_id is None:
            return CrossProfileCase(
                topic_id=topic,
                case_type=case_type,
                observed=False,
                covered_concepts=[],
                direct_prerequisites=[],
                transitive_prerequisites=[],
                states=[],
                explanation=explanation,
            )
        book = first[book_id]
        return CrossProfileCase(
            topic_id=topic,
            case_type=case_type,
            observed=True,
            book_id=book_id,
            title=book.title,
            covered_concepts=book.covered_concepts,
            direct_prerequisites=book.direct_prerequisites,
            transitive_prerequisites=book.transitive_prerequisites,
            states=states(book_id),
            explanation=explanation,
        )

    state_map = {book_id: states(book_id) for book_id in candidate_ids}
    improvement = max(
        candidate_ids,
        key=lambda book_id: (
            state_map[book_id][0].e_group_index - state_map[book_id][1].e_group_index,
            book_id,
        ),
        default=None,
    )
    if improvement is not None:
        delta = state_map[improvement][0].e_group_index - state_map[improvement][1].e_group_index
        if delta <= 0:
            improvement = None
    opportunity = max(
        candidate_ids,
        key=lambda book_id: (
            state_map[book_id][1].direct_learning_opportunity
            - state_map[book_id][2].direct_learning_opportunity,
            book_id,
        ),
        default=None,
    )
    uneven = max(
        candidate_ids,
        key=lambda book_id: (
            state_map[book_id][3].e_group_index - state_map[book_id][1].e_group_index,
            book_id,
        ),
        default=None,
    )
    if uneven is not None:
        delta = state_map[uneven][3].e_group_index - state_map[uneven][1].e_group_index
        if delta <= 0:
            uneven = None
    stable_candidates = [
        book_id
        for book_id in candidate_ids
        if len({state.e_group_index for state in state_map[book_id]}) == 1
    ]
    stable = stable_candidates[0] if stable_candidates else None
    return [
        build(
            "beginner_to_intermediate_improvement",
            improvement,
            "Largest observed E ordering-group improvement from beginner to intermediate."
            if improvement
            else (
                "No evidence-backed book moved to an earlier E group from beginner to intermediate."
            ),
        ),
        build(
            "advanced_opportunity_reduction",
            opportunity,
            "Largest direct-learning-opportunity reduction from intermediate to advanced.",
        ),
        build(
            "uneven_prerequisite_weakness",
            uneven,
            "Largest observed E ordering-group decline from intermediate to uneven."
            if uneven
            else "No evidence-backed book moved to a later E group under the uneven profile.",
        ),
        build(
            "stable_position",
            stable,
            "Evidence-backed book remained in the same E ordering group across all profiles."
            if stable
            else "No evidence-backed book stayed in one E ordering group across all profiles.",
        ),
    ]


def _behavioral_observations(
    topic: str, evaluations: list[ScenarioEvaluation], cases: list[CrossProfileCase]
) -> list[BehavioralObservation]:
    by_scenario = {evaluation.scenario.scenario_id: evaluation for evaluation in evaluations}

    def top_book(scenario_id: str) -> ConceptReadinessBookDiagnostic | None:
        evaluation = by_scenario[scenario_id]
        ordering = next(
            variant
            for variant in evaluation.variants
            if variant.variant_id == "e_prerequisite_first"
        )
        first_available = next((group for group in ordering.groups if group.available), None)
        if first_available is None:
            return None
        book_id = first_available.book_ids[0]
        return next(book for book in evaluation.diagnostics if book.book_id == book_id)

    observations: list[BehavioralObservation] = []
    beginner = top_book("beginner")
    advanced = top_book("advanced")
    intermediate = top_book("intermediate")
    uneven_case = next(case for case in cases if case.case_type == "uneven_prerequisite_weakness")
    for scenario_id, question, book in (
        (
            "beginner",
            "Does prerequisite-first ordering prevent opportunity alone from dominating?",
            beginner,
        ),
        (
            "advanced",
            "Does an advanced reader show reduced opportunity on already-known coverage?",
            advanced,
        ),
        (
            "intermediate",
            "Is a book with readiness and remaining opportunity identifiable?",
            intermediate,
        ),
    ):
        if book is None:
            observation = "No comparable evidence-backed book was available."
            book_id = None
        else:
            observation = (
                f"{book.title}: prerequisite="
                f"{book.transitive_prerequisite_readiness.mastery_mean:.3f}, "
                f"opportunity={book.direct.learning_opportunity:.3f}, "
                f"direct_coverage={book.direct.assessment_coverage:.3f}, "
                f"prerequisite_coverage="
                f"{book.transitive_prerequisite_readiness.assessment_coverage:.3f}."
            )
            book_id = book.book_id
        observations.append(
            BehavioralObservation(
                topic_id=topic,
                scenario_id=scenario_id,
                question=question,
                representative_book_id=book_id,
                observation=observation,
            )
        )
    observations.append(
        BehavioralObservation(
            topic_id=topic,
            scenario_id="uneven",
            question="Does an asymmetric prerequisite weakness change relative ordering?",
            representative_book_id=uneven_case.book_id,
            observation=uneven_case.explanation,
        )
    )
    return observations


def _human_pair_evaluation(
    fixed_report,
    config: MultiReaderConceptExperimentConfig,
) -> tuple[list[HumanPairEvaluation], list[AgreementSummary]]:
    generated = {pair.pair_id: pair for pair in fixed_report.human_pair_review_packet}
    configured = {pair.pair_id: pair for pair in config.frozen_human_pairs}
    if set(generated) != set(configured):
        raise MultiReaderConceptExperimentError("frozen human pair IDs changed")
    results: list[HumanPairEvaluation] = []
    for pair_id in sorted(configured):
        pair = generated[pair_id]
        frozen = configured[pair_id]
        if (
            pair.topic_id != frozen.topic_id
            or pair.left.book_id != frozen.left_book_id
            or pair.right.book_id != frozen.right_book_id
        ):
            raise MultiReaderConceptExperimentError(f"frozen pair membership changed: {pair_id}")
        results.append(
            HumanPairEvaluation(
                pair_id=pair_id,
                topic_id=pair.topic_id,
                left_book_id=pair.left.book_id,
                right_book_id=pair.right.book_id,
                human_preference=frozen.human_preference,
                d1_prediction=_d1_prediction(pair.left, pair.right),
                d2_prediction=_prediction(
                    pair.left.unweighted_combination, pair.right.unweighted_combination
                ),
                direct_only_prediction=_prediction(
                    pair.left.direct_learning_opportunity,
                    pair.right.direct_learning_opportunity,
                ),
                prerequisite_only_prediction=_prediction(
                    pair.left.prerequisite_readiness,
                    pair.right.prerequisite_readiness,
                ),
                e_prediction=_e_prediction(
                    pair.left,
                    pair.right,
                    config.primary_lexicographic_tolerance,
                ),
            )
        )
    summaries: list[AgreementSummary] = []
    field_by_variant = {
        "d1": "d1_prediction",
        "d2": "d2_prediction",
        "direct_only": "direct_only_prediction",
        "prerequisite_only": "prerequisite_only_prediction",
        "e_prerequisite_first": "e_prediction",
    }
    for variant, field in field_by_variant.items():
        for topic in ("linear-algebra", "operating-systems", None):
            selected = [result for result in results if topic is None or result.topic_id == topic]
            comparable = [
                result
                for result in selected
                if result.human_preference != "not_judgable"
                and getattr(result, field) != "not_comparable"
            ]
            agreement = sum(
                getattr(result, field) == result.human_preference for result in comparable
            )
            summaries.append(
                AgreementSummary(
                    variant_id=variant,
                    topic_id=topic,
                    comparable_pair_count=len(comparable),
                    agreement_count=agreement,
                    agreement_rate=agreement / len(comparable) if comparable else None,
                )
            )
        combined = next(
            summary
            for summary in summaries
            if summary.variant_id == variant and summary.topic_id is None
        )
        if variant in config.expected_legacy_agreement:
            expected = config.expected_legacy_agreement[variant]
            if combined.agreement_count != expected:
                raise MultiReaderConceptExperimentError(
                    f"legacy {variant} pair agreement changed: "
                    f"{combined.agreement_count} != {expected}"
                )
    return results, summaries


def evaluate_frozen_human_pairs(
    fixed_report,
    config: MultiReaderConceptExperimentConfig,
) -> tuple[list[HumanPairEvaluation], list[AgreementSummary]]:
    """Compare frozen pair membership and labels without changing any diagnostic."""

    return _human_pair_evaluation(fixed_report, config)


def evaluate_multi_reader_concept_ranking(
    mapping: BookEvidenceConceptMappingReport,
    fixed_readers: list[ReaderProfile],
    graph: LoadedConceptGraph,
    reviews: LoadedConceptGraphReviews,
    matching: LoadedConceptMatchingConfig,
    loaded_config: LoadedMultiReaderConceptExperimentConfig,
    *,
    mapping_report_hash: str,
    fixed_reader_hashes: dict[str, str],
    input_hashes: dict[str, str],
) -> MultiReaderConceptRankingReport:
    """Evaluate frozen concept diagnostics across eight deterministic readers."""

    config = loaded_config.config
    if config.experiment_version != EXPERIMENT_VERSION:
        raise MultiReaderConceptExperimentError("experiment version mismatch")
    projection = build_accepted_graph_projection(graph, reviews)
    if len(projection.edges) != 18:
        raise MultiReaderConceptExperimentError("accepted graph must retain exactly 18 edges")
    scenario_profiles = build_scenario_profiles(loaded_config, projection)

    scenario_evaluations: list[ScenarioEvaluation] = []
    comparisons: list[VariantComparison] = []
    for topic in sorted(config.scenarios):
        for profile in [
            item for item in scenario_profiles if item.reader_profile.topic_id == topic
        ]:
            base_report = evaluate_concept_readiness_ranking(
                mapping,
                [profile.reader_profile],
                graph,
                reviews,
                matching,
                mapping_report_hash=mapping_report_hash,
                reader_profile_hashes={topic: profile.reader_profile_hash},
                input_hashes=input_hashes,
            )
            books = base_report.books
            variants = build_variant_orderings(
                books,
                config.primary_lexicographic_tolerance,
            )
            evaluation = ScenarioEvaluation(
                topic_id=topic,
                scenario=profile,
                direct_coverage_distribution=_coverage_distribution(books, "direct"),
                direct_prerequisite_coverage_distribution=_coverage_distribution(
                    books, "direct_prerequisite_readiness"
                ),
                transitive_prerequisite_coverage_distribution=_coverage_distribution(
                    books, "transitive_prerequisite_readiness"
                ),
                diagnostics=books,
                variants=variants,
                lexicographic_sensitivity=[
                    _lexicographic_ordering(books, tolerance)
                    for tolerance in config.lexicographic_tolerance_sensitivity
                ],
            )
            scenario_evaluations.append(evaluation)
            comparisons.append(
                _variant_comparison(
                    topic,
                    profile.scenario_id,
                    books,
                    config.primary_lexicographic_tolerance,
                )
            )

    cross_profile_cases: list[CrossProfileCase] = []
    observations: list[BehavioralObservation] = []
    evidence_gaps: list[EvidenceGapSummary] = []
    for topic in sorted(config.scenarios):
        topic_evaluations = [
            evaluation for evaluation in scenario_evaluations if evaluation.topic_id == topic
        ]
        cases = _cross_profile_cases(topic, topic_evaluations)
        cross_profile_cases.extend(cases)
        observations.extend(_behavioral_observations(topic, topic_evaluations, cases))
        first = topic_evaluations[0]
        without_evidence = sorted(
            book.book_id for book in first.diagnostics if not book.covered_concepts
        )
        evidence_gaps.append(
            EvidenceGapSummary(
                topic_id=topic,
                no_concept_evidence_count=len(without_evidence),
                book_ids=without_evidence,
            )
        )

    fixed_report = evaluate_concept_readiness_ranking(
        mapping,
        fixed_readers,
        graph,
        reviews,
        matching,
        mapping_report_hash=mapping_report_hash,
        reader_profile_hashes=fixed_reader_hashes,
        input_hashes=input_hashes,
    )
    human_pairs, human_agreement = evaluate_frozen_human_pairs(fixed_report, config)
    return MultiReaderConceptRankingReport(
        experiment_version=EXPERIMENT_VERSION,
        config_version=config.config_version,
        config_hash=loaded_config.content_hash,
        reviewed_report_hash=config.reviewed_report_hash,
        mapping_report_hash=mapping_report_hash,
        input_hashes=dict(sorted(input_hashes.items())),
        production_ranking_modified=False,
        occurrence_count_used_as_weight=False,
        prose_difficulty_used=False,
        unknown_readiness_treated_as_zero=False,
        accepted_graph=projection,
        primary_lexicographic_tolerance=config.primary_lexicographic_tolerance,
        scenarios=scenario_evaluations,
        cross_profile_cases=cross_profile_cases,
        behavioral_observations=observations,
        evidence_gaps=evidence_gaps,
        variant_comparisons=comparisons,
        human_pair_evaluations=human_pairs,
        human_pair_agreement=human_agreement,
    )
