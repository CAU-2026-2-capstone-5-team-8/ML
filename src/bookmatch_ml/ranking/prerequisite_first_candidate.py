"""Exact prerequisite-first ranking-v2 candidate isolated from production rank-v1."""

from collections import Counter
from typing import Literal

from pydantic import Field

from bookmatch_ml.concept_v2.book_evidence_mapping import (
    BookEvidenceConceptMappingReport,
)
from bookmatch_ml.concept_v2.graph import LoadedConceptGraph
from bookmatch_ml.concept_v2.profile import LoadedConceptMatchingConfig
from bookmatch_ml.concept_v2.validation import LoadedConceptGraphReviews
from bookmatch_ml.ranking.concept_readiness_experiments import (
    AcceptedGraphProjection,
    ConceptReadinessBookDiagnostic,
    DiagnosticSnapshot,
    build_accepted_graph_projection,
    evaluate_concept_readiness_ranking,
)
from bookmatch_ml.ranking.multi_reader_concept_experiments import (
    LoadedMultiReaderConceptExperimentConfig,
    MultiReaderConceptExperimentConfig,
    ScenarioProfile,
    build_scenario_profiles,
)
from bookmatch_ml.schemas import ReaderProfile, StrictModel

CANDIDATE_VERSION = "concept-prerequisite-ranking-v2-candidate-v1"


class PrerequisiteFirstCandidateError(ValueError):
    """Candidate inputs cannot produce a trustworthy deterministic ranking."""


class CandidateBook(StrictModel):
    """One book in the candidate ranking or an explicit fallback pool."""

    book_id: str = Field(min_length=1)
    title: str = Field(min_length=1)
    topic_id: str = Field(min_length=1)
    prerequisite_readiness: float | None = Field(default=None, ge=0, le=1)
    prerequisite_assessed_count: int = Field(ge=0)
    prerequisite_total_count: int = Field(ge=0)
    prerequisite_coverage: float | None = Field(default=None, ge=0, le=1)
    direct_learning_opportunity: float | None = Field(default=None, ge=0, le=1)
    direct_assessed_count: int = Field(ge=0)
    direct_total_count: int = Field(ge=0)
    direct_coverage: float | None = Field(default=None, ge=0, le=1)
    covered_concepts: list[str]
    inferred_prerequisites: list[str]
    availability_status: Literal["personalizable", "concept_only", "evidence_unavailable"]
    rank: int | None = Field(default=None, ge=1)
    ordering_group: int | None = Field(default=None, ge=1)
    explanation_reasons: list[str] = Field(min_length=1)
    ranking_candidate_version: Literal["concept-prerequisite-ranking-v2-candidate-v1"]


class CandidateRankingStatistics(StrictModel):
    """Availability and resolution statistics for one topic and reader."""

    total_candidate_count: int = Field(ge=0)
    personalizable_count: int = Field(ge=0)
    concept_only_count: int = Field(ge=0)
    evidence_unavailable_count: int = Field(ge=0)
    fallback_count: int = Field(ge=0)
    prerequisite_readiness_distinct_value_count: int = Field(ge=0)
    final_ordering_group_count: int = Field(ge=0)
    largest_tie_group_size: int = Field(ge=0)
    singleton_group_count: int = Field(ge=0)


class CandidateScenarioRanking(StrictModel):
    """Full deterministic candidate output for one configured reader scenario."""

    topic_id: str
    scenario_id: str
    scenario_description: str
    reader_profile_hash: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    statistics: CandidateRankingStatistics
    books: list[CandidateBook]


class CandidateHumanPairEvaluation(StrictModel):
    """Frozen human pair and the untuned exact candidate prediction."""

    pair_id: str
    topic_id: str
    left_book_id: str
    right_book_id: str
    human_preference: Literal["A", "B", "tie", "not_judgable"]
    candidate_prediction: Literal["A", "B", "tie", "not_comparable"]


class CandidateHumanAgreement(StrictModel):
    """Agreement summary for one topic or the combined packet."""

    topic_id: str | None = None
    comparable_pair_count: int = Field(ge=0)
    agreement_count: int = Field(ge=0)
    agreement_rate: float | None = Field(default=None, ge=0, le=1)


class PrerequisiteFirstCandidateReport(StrictModel):
    """Offline validation report for the production candidate."""

    ranking_candidate_version: Literal["concept-prerequisite-ranking-v2-candidate-v1"]
    production_ranking_modified: Literal[False]
    ordering_policy: Literal[
        "prerequisite_readiness_desc_then_direct_opportunity_desc_then_book_id"
    ]
    tolerance_used: Literal[False]
    coverage_used_as_score: Literal[False]
    occurrence_count_used_as_weight: Literal[False]
    source_quality_used_as_weight: Literal[False]
    unknown_readiness_treated_as_zero: Literal[False]
    config_hash: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    mapping_report_hash: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    input_hashes: dict[str, str]
    accepted_graph: AcceptedGraphProjection
    scenarios: list[CandidateScenarioRanking]
    human_pair_evaluations: list[CandidateHumanPairEvaluation]
    human_pair_agreement: list[CandidateHumanAgreement]


def _availability(book: ConceptReadinessBookDiagnostic) -> str:
    """Classify evidence without inventing a readiness value."""

    if not book.covered_concepts:
        return "evidence_unavailable"
    if (
        book.transitive_prerequisite_readiness.total_concept_count > 0
        and book.transitive_prerequisite_readiness.assessed_concept_count > 0
    ):
        return "personalizable"
    return "concept_only"


def _explanations(book: ConceptReadinessBookDiagnostic, status: str) -> list[str]:
    """Build bounded Korean explanations grounded only in reported diagnostics."""

    prerequisite = book.transitive_prerequisite_readiness
    direct = book.direct
    if status == "evidence_unavailable":
        return ["source-aware concept evidence가 없어 개인화 순위를 계산하지 않습니다."]
    if status == "concept_only":
        if prerequisite.total_concept_count == 0:
            return ["다루는 개념은 확인됐지만 accepted graph에서 선행 개념을 추론할 수 없습니다."]
        return ["평가된 선행 개념이 없어 개인화 순위를 계산하지 않습니다."]

    reasons = [
        f"선행 개념 {prerequisite.total_concept_count}개 중 "
        f"{prerequisite.assessed_concept_count}개가 평가됐고 평균 준비도는 "
        f"{prerequisite.mastery_mean:.2f}입니다."
    ]
    if direct.learning_opportunity is not None:
        reasons.append(
            f"책의 개념 {direct.total_concept_count}개 중 {direct.assessed_concept_count}개가 "
            f"평가됐고 학습 기회는 {direct.learning_opportunity:.2f}입니다."
        )
    else:
        reasons.append("다루는 개념의 readiness가 평가되지 않아 학습 기회는 알 수 없습니다.")
    if prerequisite.assessment_coverage is not None and prerequisite.assessment_coverage < 1:
        reasons.append("평가되지 않은 선행 개념이 있어 준비도 판단에는 제한이 있습니다.")
    return reasons


def _personalizable_sort_key(book: ConceptReadinessBookDiagnostic) -> tuple:
    """Apply exact lexicographic policy with missing secondary values last."""

    readiness = book.transitive_prerequisite_readiness.mastery_mean
    if readiness is None:
        raise PrerequisiteFirstCandidateError(
            f"personalizable book lacks prerequisite readiness: {book.book_id}"
        )
    opportunity = book.direct.learning_opportunity
    return (
        -readiness,
        opportunity is None,
        -opportunity if opportunity is not None else 0.0,
        book.book_id,
    )


def _diagnostic_group_key(book: ConceptReadinessBookDiagnostic) -> tuple[float, float | None]:
    """Return the exact primary and secondary values defining an ordering group."""

    readiness = book.transitive_prerequisite_readiness.mastery_mean
    if readiness is None:
        raise PrerequisiteFirstCandidateError(
            f"personalizable book lacks prerequisite readiness: {book.book_id}"
        )
    return readiness, book.direct.learning_opportunity


def build_candidate_ranking(
    topic_id: str,
    scenario_id: str,
    scenario_description: str,
    reader_profile_hash: str,
    diagnostics: list[ConceptReadinessBookDiagnostic],
) -> CandidateScenarioRanking:
    """Build exact prerequisite-first ranks and explicit fallback pools."""

    topic_books = [book for book in diagnostics if book.topic_id == topic_id]
    if not topic_books:
        raise PrerequisiteFirstCandidateError(f"no candidate books for topic: {topic_id}")
    book_ids = [book.book_id for book in topic_books]
    if len(book_ids) != len(set(book_ids)):
        raise PrerequisiteFirstCandidateError(f"duplicate candidate book_id: {topic_id}")

    by_status: dict[str, list[ConceptReadinessBookDiagnostic]] = {
        "personalizable": [],
        "concept_only": [],
        "evidence_unavailable": [],
    }
    for book in topic_books:
        by_status[_availability(book)].append(book)
    personalizable = sorted(by_status["personalizable"], key=_personalizable_sort_key)

    group_by_book: dict[str, int] = {}
    group_sizes: Counter[int] = Counter()
    previous_key: tuple[float, float | None] | None = None
    group_index = 0
    for book in personalizable:
        key = _diagnostic_group_key(book)
        if previous_key is None or key != previous_key:
            group_index += 1
            previous_key = key
        group_by_book[book.book_id] = group_index
        group_sizes[group_index] += 1

    output: list[CandidateBook] = []
    rank_by_book = {book.book_id: index for index, book in enumerate(personalizable, start=1)}
    ordered = [
        *personalizable,
        *sorted(by_status["concept_only"], key=lambda book: book.book_id),
        *sorted(by_status["evidence_unavailable"], key=lambda book: book.book_id),
    ]
    for book in ordered:
        status = _availability(book)
        prerequisite = book.transitive_prerequisite_readiness
        direct = book.direct
        output.append(
            CandidateBook(
                book_id=book.book_id,
                title=book.title,
                topic_id=book.topic_id,
                prerequisite_readiness=prerequisite.mastery_mean,
                prerequisite_assessed_count=prerequisite.assessed_concept_count,
                prerequisite_total_count=prerequisite.total_concept_count,
                prerequisite_coverage=prerequisite.assessment_coverage,
                direct_learning_opportunity=direct.learning_opportunity,
                direct_assessed_count=direct.assessed_concept_count,
                direct_total_count=direct.total_concept_count,
                direct_coverage=direct.assessment_coverage,
                covered_concepts=book.covered_concepts,
                inferred_prerequisites=book.transitive_prerequisites,
                availability_status=status,
                rank=rank_by_book.get(book.book_id),
                ordering_group=group_by_book.get(book.book_id),
                explanation_reasons=_explanations(book, status),
                ranking_candidate_version=CANDIDATE_VERSION,
            )
        )

    prerequisite_values = {
        book.transitive_prerequisite_readiness.mastery_mean for book in personalizable
    }
    return CandidateScenarioRanking(
        topic_id=topic_id,
        scenario_id=scenario_id,
        scenario_description=scenario_description,
        reader_profile_hash=reader_profile_hash,
        statistics=CandidateRankingStatistics(
            total_candidate_count=len(topic_books),
            personalizable_count=len(personalizable),
            concept_only_count=len(by_status["concept_only"]),
            evidence_unavailable_count=len(by_status["evidence_unavailable"]),
            fallback_count=(
                len(by_status["concept_only"]) + len(by_status["evidence_unavailable"])
            ),
            prerequisite_readiness_distinct_value_count=len(prerequisite_values),
            final_ordering_group_count=len(group_sizes),
            largest_tie_group_size=max(group_sizes.values(), default=0),
            singleton_group_count=sum(size == 1 for size in group_sizes.values()),
        ),
        books=output,
    )


def evaluate_candidate_scenario(
    mapping: BookEvidenceConceptMappingReport,
    profile: ScenarioProfile,
    graph: LoadedConceptGraph,
    reviews: LoadedConceptGraphReviews,
    matching: LoadedConceptMatchingConfig,
    *,
    mapping_report_hash: str,
    input_hashes: dict[str, str],
) -> CandidateScenarioRanking:
    """Evaluate one configured profile and build its candidate ordering."""

    topic = profile.reader_profile.topic_id
    base = evaluate_concept_readiness_ranking(
        mapping,
        [profile.reader_profile],
        graph,
        reviews,
        matching,
        mapping_report_hash=mapping_report_hash,
        reader_profile_hashes={topic: profile.reader_profile_hash},
        input_hashes=input_hashes,
    )
    return build_candidate_ranking(
        topic,
        profile.scenario_id,
        profile.description,
        profile.reader_profile_hash,
        base.books,
    )


def _candidate_prediction(
    left: DiagnosticSnapshot, right: DiagnosticSnapshot
) -> Literal["A", "B", "tie", "not_comparable"]:
    """Apply the exact two-axis policy to one frozen human pair."""

    if left.prerequisite_readiness is None or right.prerequisite_readiness is None:
        return "not_comparable"
    if left.prerequisite_readiness != right.prerequisite_readiness:
        return "A" if left.prerequisite_readiness > right.prerequisite_readiness else "B"
    if left.direct_learning_opportunity is None:
        return "tie" if right.direct_learning_opportunity is None else "B"
    if right.direct_learning_opportunity is None:
        return "A"
    if left.direct_learning_opportunity != right.direct_learning_opportunity:
        return "A" if left.direct_learning_opportunity > right.direct_learning_opportunity else "B"
    return "tie"


def evaluate_candidate_human_pairs(
    fixed_report,
    config: MultiReaderConceptExperimentConfig,
) -> tuple[list[CandidateHumanPairEvaluation], list[CandidateHumanAgreement]]:
    """Validate frozen membership and compare the untuned candidate to human labels."""

    generated = {pair.pair_id: pair for pair in fixed_report.human_pair_review_packet}
    frozen = {pair.pair_id: pair for pair in config.frozen_human_pairs}
    if set(generated) != set(frozen):
        raise PrerequisiteFirstCandidateError("frozen human pair IDs changed")
    evaluations: list[CandidateHumanPairEvaluation] = []
    for pair_id in sorted(frozen):
        pair = generated[pair_id]
        expected = frozen[pair_id]
        if (
            pair.topic_id != expected.topic_id
            or pair.left.book_id != expected.left_book_id
            or pair.right.book_id != expected.right_book_id
        ):
            raise PrerequisiteFirstCandidateError(f"frozen pair membership changed: {pair_id}")
        evaluations.append(
            CandidateHumanPairEvaluation(
                pair_id=pair_id,
                topic_id=pair.topic_id,
                left_book_id=pair.left.book_id,
                right_book_id=pair.right.book_id,
                human_preference=expected.human_preference,
                candidate_prediction=_candidate_prediction(pair.left, pair.right),
            )
        )

    summaries: list[CandidateHumanAgreement] = []
    for topic in ("linear-algebra", "operating-systems", None):
        selected = [item for item in evaluations if topic is None or item.topic_id == topic]
        comparable = [
            item
            for item in selected
            if item.human_preference != "not_judgable"
            and item.candidate_prediction != "not_comparable"
        ]
        agreement = sum(item.candidate_prediction == item.human_preference for item in comparable)
        summaries.append(
            CandidateHumanAgreement(
                topic_id=topic,
                comparable_pair_count=len(comparable),
                agreement_count=agreement,
                agreement_rate=agreement / len(comparable) if comparable else None,
            )
        )
    return evaluations, summaries


def evaluate_prerequisite_first_candidate(
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
) -> PrerequisiteFirstCandidateReport:
    """Run Scale-50 multi-reader and frozen-pair candidate validation."""

    projection = build_accepted_graph_projection(graph, reviews)
    if len(projection.edges) != 18:
        raise PrerequisiteFirstCandidateError("accepted graph must retain exactly 18 edges")
    profiles = build_scenario_profiles(loaded_config, projection)
    scenarios = [
        evaluate_candidate_scenario(
            mapping,
            profile,
            graph,
            reviews,
            matching,
            mapping_report_hash=mapping_report_hash,
            input_hashes=input_hashes,
        )
        for profile in profiles
    ]

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
    pair_evaluations, pair_agreement = evaluate_candidate_human_pairs(
        fixed_report, loaded_config.config
    )
    return PrerequisiteFirstCandidateReport(
        ranking_candidate_version=CANDIDATE_VERSION,
        production_ranking_modified=False,
        ordering_policy=("prerequisite_readiness_desc_then_direct_opportunity_desc_then_book_id"),
        tolerance_used=False,
        coverage_used_as_score=False,
        occurrence_count_used_as_weight=False,
        source_quality_used_as_weight=False,
        unknown_readiness_treated_as_zero=False,
        config_hash=loaded_config.content_hash,
        mapping_report_hash=mapping_report_hash,
        input_hashes=dict(sorted(input_hashes.items())),
        accepted_graph=projection,
        scenarios=scenarios,
        human_pair_evaluations=pair_evaluations,
        human_pair_agreement=pair_agreement,
    )
