"""Offline concept-readiness ranking diagnostics isolated from production rank-v1."""

from collections import Counter, defaultdict, deque
from typing import Literal

from pydantic import Field, model_validator

from bookmatch_ml.concept_v2.book_evidence_mapping import (
    BookEvidenceConceptMappingReport,
)
from bookmatch_ml.concept_v2.graph import LoadedConceptGraph
from bookmatch_ml.concept_v2.profile import LoadedConceptMatchingConfig
from bookmatch_ml.concept_v2.validation import LoadedConceptGraphReviews
from bookmatch_ml.schemas import ReaderProfile, StrictModel

EXPERIMENT_VERSION = "concept-readiness-ranking-experiment-v1"


class ConceptReadinessExperimentError(ValueError):
    """Experiment inputs cannot produce a trustworthy deterministic comparison."""


class AcceptedPrerequisiteEdge(StrictModel):
    topic: str = Field(min_length=1)
    prerequisite: str = Field(min_length=1)
    dependent: str = Field(min_length=1)


class AcceptedGraphProjection(StrictModel):
    """Human-accepted prerequisite edges without mutating the source graph."""

    graph_version: str
    graph_hash: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    review_config_version: str
    review_config_hash: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    nodes: dict[str, list[str]]
    edges: list[AcceptedPrerequisiteEdge]

    @model_validator(mode="after")
    def validate_projection(self) -> "AcceptedGraphProjection":
        membership = {
            concept: topic for topic, concepts in self.nodes.items() for concept in concepts
        }
        keys = [(edge.topic, edge.prerequisite, edge.dependent) for edge in self.edges]
        if keys != sorted(keys) or len(keys) != len(set(keys)):
            raise ValueError("accepted prerequisite edges must be unique and sorted")
        adjacency: dict[str, set[str]] = defaultdict(set)
        indegree = {concept: 0 for concept in membership}
        for edge in self.edges:
            if (
                membership.get(edge.prerequisite) != edge.topic
                or membership.get(edge.dependent) != edge.topic
            ):
                raise ValueError(
                    "accepted prerequisite edge mixes topics or references unknown node"
                )
            if edge.dependent not in adjacency[edge.prerequisite]:
                adjacency[edge.prerequisite].add(edge.dependent)
                indegree[edge.dependent] += 1
        queue = deque(sorted(node for node, degree in indegree.items() if degree == 0))
        visited = 0
        while queue:
            node = queue.popleft()
            visited += 1
            for dependent in sorted(adjacency[node]):
                indegree[dependent] -= 1
                if indegree[dependent] == 0:
                    queue.append(dependent)
        if visited != len(indegree):
            raise ValueError("accepted prerequisite projection contains a cycle")
        return self


class ReadinessAxis(StrictModel):
    total_concept_count: int = Field(ge=0)
    assessed_concept_count: int = Field(ge=0)
    assessment_coverage: float | None = Field(default=None, ge=0, le=1)
    mastery_mean: float | None = Field(default=None, ge=0, le=1)
    semantics: Literal["calculated", "unassessed", "not_applicable"]


class DirectConceptAxis(ReadinessAxis):
    learning_opportunity: float | None = Field(default=None, ge=0, le=1)


class ConceptReadinessBookDiagnostic(StrictModel):
    book_id: str
    title: str
    topic_id: str
    covered_concepts: list[str]
    assessed_covered_mastery: dict[str, float]
    direct: DirectConceptAxis
    direct_prerequisites: list[str]
    transitive_prerequisites: list[str]
    assessed_direct_prerequisite_mastery: dict[str, float]
    assessed_transitive_prerequisite_mastery: dict[str, float]
    direct_prerequisite_readiness: ReadinessAxis
    transitive_prerequisite_readiness: ReadinessAxis
    two_stage_status: Literal["eligible", "challenge_candidate", "insufficient_evidence"]
    unweighted_combination: float | None = Field(default=None, ge=0, le=1)


class GroupStatistics(StrictModel):
    candidate_count: int = Field(ge=0)
    available_value_count: int = Field(ge=0)
    unavailable_value_count: int = Field(ge=0)
    distinct_available_value_count: int = Field(ge=0)
    distinct_group_count: int = Field(ge=0)
    largest_tie_group_size: int = Field(ge=0)
    singleton_group_count: int = Field(ge=0)
    uniquely_distinguished_book_count: int = Field(ge=0)
    baseline_largest_tie_reduction: int = Field(ge=0)


class DirectVariantSummary(StrictModel):
    books_with_concepts: int = Field(ge=0)
    books_with_assessed_concepts: int = Field(ge=0)
    coverage_distribution: dict[str, int]
    opportunity_groups: GroupStatistics


class PrerequisiteVariantSummary(StrictModel):
    books_with_prerequisites: int = Field(ge=0)
    books_with_assessed_prerequisites: int = Field(ge=0)
    coverage_distribution: dict[str, int]
    readiness_groups: GroupStatistics


class CombinedVariantSummary(StrictModel):
    groups: GroupStatistics
    status_counts: dict[str, int] = Field(default_factory=dict)


class TopicExperimentSummary(StrictModel):
    topic_id: str
    candidate_count: int = Field(ge=1)
    topic_only: GroupStatistics
    direct: DirectVariantSummary
    prerequisite_direct: PrerequisiteVariantSummary
    prerequisite_transitive: PrerequisiteVariantSummary
    books_with_transitive_only_additions: int = Field(ge=0)
    two_stage: CombinedVariantSummary
    unweighted_combination: CombinedVariantSummary


class DiagnosticSnapshot(StrictModel):
    book_id: str
    title: str
    covered_concepts: list[str]
    inferred_prerequisites: list[str]
    assessed_covered_mastery: dict[str, float]
    assessed_prerequisite_mastery: dict[str, float]
    direct_coverage: float | None = Field(default=None, ge=0, le=1)
    direct_learning_opportunity: float | None = Field(default=None, ge=0, le=1)
    prerequisite_coverage: float | None = Field(default=None, ge=0, le=1)
    prerequisite_readiness: float | None = Field(default=None, ge=0, le=1)
    two_stage_status: str
    unweighted_combination: float | None = Field(default=None, ge=0, le=1)


class RepresentativeCase(StrictModel):
    topic_id: str
    case_type: Literal[
        "highest_prerequisite_readiness",
        "lowest_prerequisite_readiness",
        "highest_direct_learning_opportunity",
        "lowest_direct_learning_opportunity",
        "direct_prerequisite_divergence",
    ]
    explanation: str
    book: DiagnosticSnapshot


class HumanBookPair(StrictModel):
    pair_id: str
    topic_id: str
    selection_reason: str
    left: DiagnosticSnapshot
    right: DiagnosticSnapshot
    human_preference: None = None
    review_note: None = None


class ConceptReadinessRankingExperimentReport(StrictModel):
    experiment_version: Literal["concept-readiness-ranking-experiment-v1"]
    production_ranking_modified: Literal[False]
    prose_difficulty_used: Literal[False]
    occurrence_count_used_as_weight: Literal[False]
    mapping_report_version: str
    mapping_report_hash: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    matching_config_version: str
    matching_config_hash: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    minimum_readiness_assessment_coverage: float = Field(ge=0, le=1)
    minimum_opportunity_assessment_coverage: float = Field(ge=0, le=1)
    minimum_readiness_score: float = Field(ge=0, le=1)
    accepted_graph: AcceptedGraphProjection
    reader_profile_hashes: dict[str, str]
    input_hashes: dict[str, str]
    topics: list[TopicExperimentSummary]
    books: list[ConceptReadinessBookDiagnostic]
    representative_cases: list[RepresentativeCase]
    human_pair_review_packet: list[HumanBookPair]


def build_accepted_graph_projection(
    graph: LoadedConceptGraph,
    reviews: LoadedConceptGraphReviews,
) -> AcceptedGraphProjection:
    """Project only accepted reviewed edges and reject an incomplete human review."""

    if any(row.review_status == "unreviewed" for row in reviews.config.reviews):
        raise ConceptReadinessExperimentError("concept graph human review is incomplete")
    decisions = {
        (row.topic, row.prerequisite, row.dependent): row.review_status
        for row in reviews.config.reviews
    }
    accepted = [
        AcceptedPrerequisiteEdge(
            topic=edge.topic,
            prerequisite=edge.prerequisite,
            dependent=edge.dependent,
        )
        for edge in graph.graph.edges
        if decisions[(edge.topic, edge.prerequisite, edge.dependent)] == "accepted"
    ]
    return AcceptedGraphProjection(
        graph_version=graph.graph.graph_version,
        graph_hash=graph.content_hash,
        review_config_version=reviews.config.config_version,
        review_config_hash=reviews.content_hash,
        nodes={topic: sorted(nodes) for topic, nodes in sorted(graph.graph.nodes.items())},
        edges=sorted(accepted, key=lambda edge: (edge.topic, edge.prerequisite, edge.dependent)),
    )


def prerequisite_ancestors(
    projection: AcceptedGraphProjection,
    topic: str,
    concept: str,
    *,
    transitive: bool,
) -> set[str]:
    """Return set-valued accepted prerequisites without counting duplicate graph paths."""

    incoming: dict[str, set[str]] = defaultdict(set)
    for edge in projection.edges:
        if edge.topic == topic:
            incoming[edge.dependent].add(edge.prerequisite)
    result = set(incoming[concept])
    if not transitive:
        return result
    queue = list(sorted(result))
    while queue:
        node = queue.pop()
        for prerequisite in incoming[node]:
            if prerequisite not in result:
                result.add(prerequisite)
                queue.append(prerequisite)
    return result


def _readiness_axis(concepts: set[str], mastery: dict[str, float]) -> ReadinessAxis:
    if not concepts:
        return ReadinessAxis(
            total_concept_count=0,
            assessed_concept_count=0,
            assessment_coverage=None,
            mastery_mean=None,
            semantics="not_applicable",
        )
    assessed = {concept: mastery[concept] for concept in concepts if concept in mastery}
    if not assessed:
        return ReadinessAxis(
            total_concept_count=len(concepts),
            assessed_concept_count=0,
            assessment_coverage=0,
            mastery_mean=None,
            semantics="unassessed",
        )
    return ReadinessAxis(
        total_concept_count=len(concepts),
        assessed_concept_count=len(assessed),
        assessment_coverage=len(assessed) / len(concepts),
        mastery_mean=sum(assessed.values()) / len(assessed),
        semantics="calculated",
    )


def _direct_axis(concepts: set[str], mastery: dict[str, float]) -> DirectConceptAxis:
    readiness = _readiness_axis(concepts, mastery)
    return DirectConceptAxis(
        **readiness.model_dump(),
        learning_opportunity=(
            1 - readiness.mastery_mean if readiness.mastery_mean is not None else None
        ),
    )


def _group_statistics(
    keys: list[str | None],
    *,
    baseline_size: int,
) -> GroupStatistics:
    counts = Counter("__unavailable__" if key is None else key for key in keys)
    available = sum(key is not None for key in keys)
    distinct_available = len({key for key in keys if key is not None})
    largest = max(counts.values(), default=0)
    singletons = sum(count == 1 for count in counts.values())
    return GroupStatistics(
        candidate_count=len(keys),
        available_value_count=available,
        unavailable_value_count=len(keys) - available,
        distinct_available_value_count=distinct_available,
        distinct_group_count=len(counts),
        largest_tie_group_size=largest,
        singleton_group_count=singletons,
        uniquely_distinguished_book_count=singletons,
        baseline_largest_tie_reduction=max(0, baseline_size - largest),
    )


def _value_key(value: float | None) -> str | None:
    return None if value is None else f"{value:.12f}"


def _coverage_distribution(axes: list[ReadinessAxis]) -> dict[str, int]:
    counts = Counter(
        "not_applicable" if axis.assessment_coverage is None else f"{axis.assessment_coverage:.6f}"
        for axis in axes
    )
    return dict(sorted(counts.items()))


def _two_stage_status(
    direct: DirectConceptAxis,
    prerequisite: ReadinessAxis,
    matching: LoadedConceptMatchingConfig,
) -> Literal["eligible", "challenge_candidate", "insufficient_evidence"]:
    policy = matching.config.policy
    prerequisite_insufficient = (
        prerequisite.semantics != "not_applicable"
        and prerequisite.assessment_coverage is not None
        and prerequisite.assessment_coverage < policy.minimum_readiness_assessment_coverage
    )
    direct_insufficient = (
        direct.assessment_coverage is None
        or direct.assessment_coverage < policy.minimum_opportunity_assessment_coverage
    )
    if prerequisite_insufficient or direct_insufficient:
        return "insufficient_evidence"
    if (
        prerequisite.mastery_mean is not None
        and prerequisite.mastery_mean < policy.minimum_readiness_score
    ):
        return "challenge_candidate"
    return "eligible"


def _snapshot(book: ConceptReadinessBookDiagnostic) -> DiagnosticSnapshot:
    return DiagnosticSnapshot(
        book_id=book.book_id,
        title=book.title,
        covered_concepts=book.covered_concepts,
        inferred_prerequisites=book.transitive_prerequisites,
        assessed_covered_mastery=book.assessed_covered_mastery,
        assessed_prerequisite_mastery=book.assessed_transitive_prerequisite_mastery,
        direct_coverage=book.direct.assessment_coverage,
        direct_learning_opportunity=book.direct.learning_opportunity,
        prerequisite_coverage=book.transitive_prerequisite_readiness.assessment_coverage,
        prerequisite_readiness=book.transitive_prerequisite_readiness.mastery_mean,
        two_stage_status=book.two_stage_status,
        unweighted_combination=book.unweighted_combination,
    )


def _two_stage_key(book: ConceptReadinessBookDiagnostic) -> tuple[int, float, float, str]:
    priority = {"eligible": 0, "challenge_candidate": 1, "insufficient_evidence": 2}
    opportunity = book.direct.learning_opportunity
    readiness = book.transitive_prerequisite_readiness.mastery_mean
    return (
        priority[book.two_stage_status],
        -(opportunity if opportunity is not None else -1),
        -(readiness if readiness is not None else -1),
        book.book_id,
    )


def _two_stage_group_key(book: ConceptReadinessBookDiagnostic) -> str:
    return "|".join(
        (
            book.two_stage_status,
            _value_key(book.direct.learning_opportunity) or "unavailable",
            _value_key(book.transitive_prerequisite_readiness.mastery_mean) or "unavailable",
        )
    )


def _representative_cases(
    topic: str, books: list[ConceptReadinessBookDiagnostic]
) -> list[RepresentativeCase]:
    cases: list[RepresentativeCase] = []
    prerequisite = [
        book for book in books if book.transitive_prerequisite_readiness.mastery_mean is not None
    ]
    opportunity = [book for book in books if book.direct.learning_opportunity is not None]

    def add(case_type: str, explanation: str, book: ConceptReadinessBookDiagnostic) -> None:
        cases.append(
            RepresentativeCase(
                topic_id=topic,
                case_type=case_type,
                explanation=explanation,
                book=_snapshot(book),
            )
        )

    if prerequisite:
        highest = min(
            prerequisite,
            key=lambda book: (
                -book.transitive_prerequisite_readiness.mastery_mean,
                book.book_id,
            ),
        )
        lowest = min(
            prerequisite,
            key=lambda book: (
                book.transitive_prerequisite_readiness.mastery_mean,
                book.book_id,
            ),
        )
        add(
            "highest_prerequisite_readiness",
            "Highest measured mastery mean over assessed transitive prerequisites.",
            highest,
        )
        add(
            "lowest_prerequisite_readiness",
            "Lowest measured mastery mean over assessed transitive prerequisites.",
            lowest,
        )
    if opportunity:
        highest = min(
            opportunity, key=lambda book: (-book.direct.learning_opportunity, book.book_id)
        )
        lowest = min(opportunity, key=lambda book: (book.direct.learning_opportunity, book.book_id))
        add(
            "highest_direct_learning_opportunity",
            "Highest mean one-minus-mastery over assessed covered concepts.",
            highest,
        )
        add(
            "lowest_direct_learning_opportunity",
            "Lowest mean one-minus-mastery over assessed covered concepts.",
            lowest,
        )
    both = [
        book
        for book in books
        if book.direct.learning_opportunity is not None
        and book.transitive_prerequisite_readiness.mastery_mean is not None
    ]
    if both:
        opportunity_order = {
            book.book_id: index
            for index, book in enumerate(
                sorted(both, key=lambda item: (-item.direct.learning_opportunity, item.book_id))
            )
        }
        prerequisite_order = {
            book.book_id: index
            for index, book in enumerate(
                sorted(
                    both,
                    key=lambda item: (
                        -item.transitive_prerequisite_readiness.mastery_mean,
                        item.book_id,
                    ),
                )
            )
        }
        divergent = min(
            both,
            key=lambda book: (
                -abs(opportunity_order[book.book_id] - prerequisite_order[book.book_id]),
                book.book_id,
            ),
        )
        add(
            "direct_prerequisite_divergence",
            "Largest rank-position gap between direct opportunity and prerequisite readiness.",
            divergent,
        )
    return cases


def _pair_packet(topic: str, books: list[ConceptReadinessBookDiagnostic]) -> list[HumanBookPair]:
    pairs: list[HumanBookPair] = []
    seen: set[frozenset[str]] = set()

    def add(
        reason: str,
        left: ConceptReadinessBookDiagnostic,
        right: ConceptReadinessBookDiagnostic,
    ) -> None:
        key = frozenset((left.book_id, right.book_id))
        if left.book_id == right.book_id or key in seen or len(pairs) == 5:
            return
        seen.add(key)
        pairs.append(
            HumanBookPair(
                pair_id=f"{topic}:pair-{len(pairs) + 1}",
                topic_id=topic,
                selection_reason=reason,
                left=_snapshot(left),
                right=_snapshot(right),
            )
        )

    prerequisite = [
        book for book in books if book.transitive_prerequisite_readiness.mastery_mean is not None
    ]
    if prerequisite:
        ordered = sorted(
            prerequisite,
            key=lambda book: (
                -book.transitive_prerequisite_readiness.mastery_mean,
                book.book_id,
            ),
        )
        add("Largest prerequisite-readiness contrast.", ordered[0], ordered[-1])
    opportunity = [book for book in books if book.direct.learning_opportunity is not None]
    if opportunity:
        ordered = sorted(
            opportunity, key=lambda book: (-book.direct.learning_opportunity, book.book_id)
        )
        add("Largest direct-learning-opportunity contrast.", ordered[0], ordered[-1])

    d2_books = [book for book in books if book.unweighted_combination is not None]
    d1_rank = {
        book.book_id: index for index, book in enumerate(sorted(d2_books, key=_two_stage_key))
    }
    d2_rank = {
        book.book_id: index
        for index, book in enumerate(
            sorted(d2_books, key=lambda item: (-item.unweighted_combination, item.book_id))
        )
    }
    inversions: list[tuple[int, str, str]] = []
    for index, left in enumerate(d2_books):
        for right in d2_books[index + 1 :]:
            d1_direction = d1_rank[left.book_id] - d1_rank[right.book_id]
            d2_direction = d2_rank[left.book_id] - d2_rank[right.book_id]
            if d1_direction * d2_direction < 0:
                magnitude = abs(d1_direction - d2_direction)
                first, second = sorted((left.book_id, right.book_id))
                inversions.append((magnitude, first, second))
    by_id = {book.book_id: book for book in books}
    for _, left_id, right_id in sorted(inversions, key=lambda item: (-item[0], item[1], item[2])):
        add("D1 two-stage and D2 unweighted ordering disagree.", by_id[left_id], by_id[right_id])
        if len(pairs) == 5:
            break
    return pairs


def evaluate_concept_readiness_ranking(
    mapping: BookEvidenceConceptMappingReport,
    readers: list[ReaderProfile],
    graph: LoadedConceptGraph,
    reviews: LoadedConceptGraphReviews,
    matching: LoadedConceptMatchingConfig,
    *,
    mapping_report_hash: str,
    reader_profile_hashes: dict[str, str],
    input_hashes: dict[str, str],
) -> ConceptReadinessRankingExperimentReport:
    """Evaluate topic-only and concept-readiness variants without changing rank-v1."""

    projection = build_accepted_graph_projection(graph, reviews)
    reader_by_topic = {reader.topic_id: reader for reader in readers}
    if len(reader_by_topic) != len(readers):
        raise ConceptReadinessExperimentError("duplicate reader topic")
    if set(reader_by_topic) - set(projection.nodes):
        raise ConceptReadinessExperimentError("reader topic is missing from accepted graph")
    book_ids = [book.book_id for book in mapping.books]
    if len(book_ids) != len(set(book_ids)):
        raise ConceptReadinessExperimentError("duplicate concept mapping book_id")

    diagnostics: list[ConceptReadinessBookDiagnostic] = []
    for topic, reader in sorted(reader_by_topic.items()):
        mastery = {item.concept_id: item.score for item in reader.concept_readiness}
        if len(mastery) != len(reader.concept_readiness):
            raise ConceptReadinessExperimentError(f"duplicate reader readiness concept: {topic}")
        for book in sorted(mapping.books, key=lambda item: item.book_id):
            if topic not in book.topics:
                continue
            covered = {item.concept_id for item in book.concepts if item.topic_id == topic}
            direct_prerequisites: set[str] = set()
            transitive_prerequisites: set[str] = set()
            for concept in covered:
                direct_prerequisites.update(
                    prerequisite_ancestors(projection, topic, concept, transitive=False)
                )
                transitive_prerequisites.update(
                    prerequisite_ancestors(projection, topic, concept, transitive=True)
                )
            direct_axis = _direct_axis(covered, mastery)
            direct_prerequisite_axis = _readiness_axis(direct_prerequisites, mastery)
            transitive_prerequisite_axis = _readiness_axis(transitive_prerequisites, mastery)
            combination = (
                (direct_axis.learning_opportunity + transitive_prerequisite_axis.mastery_mean) / 2
                if direct_axis.learning_opportunity is not None
                and transitive_prerequisite_axis.mastery_mean is not None
                else None
            )
            diagnostics.append(
                ConceptReadinessBookDiagnostic(
                    book_id=book.book_id,
                    title=book.title,
                    topic_id=topic,
                    covered_concepts=sorted(covered),
                    assessed_covered_mastery={
                        concept: mastery[concept] for concept in sorted(covered & mastery.keys())
                    },
                    direct=direct_axis,
                    direct_prerequisites=sorted(direct_prerequisites),
                    transitive_prerequisites=sorted(transitive_prerequisites),
                    assessed_direct_prerequisite_mastery={
                        concept: mastery[concept]
                        for concept in sorted(direct_prerequisites & mastery.keys())
                    },
                    assessed_transitive_prerequisite_mastery={
                        concept: mastery[concept]
                        for concept in sorted(transitive_prerequisites & mastery.keys())
                    },
                    direct_prerequisite_readiness=direct_prerequisite_axis,
                    transitive_prerequisite_readiness=transitive_prerequisite_axis,
                    two_stage_status=_two_stage_status(
                        direct_axis, transitive_prerequisite_axis, matching
                    ),
                    unweighted_combination=combination,
                )
            )

    topic_summaries: list[TopicExperimentSummary] = []
    representatives: list[RepresentativeCase] = []
    pairs: list[HumanBookPair] = []
    for topic in sorted(reader_by_topic):
        books = [book for book in diagnostics if book.topic_id == topic]
        baseline = len(books)
        direct_axes = [book.direct for book in books]
        direct_prerequisite_axes = [book.direct_prerequisite_readiness for book in books]
        transitive_prerequisite_axes = [book.transitive_prerequisite_readiness for book in books]
        two_stage_keys = [_two_stage_group_key(book) for book in books]
        combinations = [_value_key(book.unweighted_combination) for book in books]
        topic_summaries.append(
            TopicExperimentSummary(
                topic_id=topic,
                candidate_count=baseline,
                topic_only=_group_statistics(["1.000000000000"] * baseline, baseline_size=baseline),
                direct=DirectVariantSummary(
                    books_with_concepts=sum(axis.total_concept_count > 0 for axis in direct_axes),
                    books_with_assessed_concepts=sum(
                        axis.assessed_concept_count > 0 for axis in direct_axes
                    ),
                    coverage_distribution=_coverage_distribution(direct_axes),
                    opportunity_groups=_group_statistics(
                        [_value_key(axis.learning_opportunity) for axis in direct_axes],
                        baseline_size=baseline,
                    ),
                ),
                prerequisite_direct=PrerequisiteVariantSummary(
                    books_with_prerequisites=sum(
                        axis.total_concept_count > 0 for axis in direct_prerequisite_axes
                    ),
                    books_with_assessed_prerequisites=sum(
                        axis.assessed_concept_count > 0 for axis in direct_prerequisite_axes
                    ),
                    coverage_distribution=_coverage_distribution(direct_prerequisite_axes),
                    readiness_groups=_group_statistics(
                        [_value_key(axis.mastery_mean) for axis in direct_prerequisite_axes],
                        baseline_size=baseline,
                    ),
                ),
                prerequisite_transitive=PrerequisiteVariantSummary(
                    books_with_prerequisites=sum(
                        axis.total_concept_count > 0 for axis in transitive_prerequisite_axes
                    ),
                    books_with_assessed_prerequisites=sum(
                        axis.assessed_concept_count > 0 for axis in transitive_prerequisite_axes
                    ),
                    coverage_distribution=_coverage_distribution(transitive_prerequisite_axes),
                    readiness_groups=_group_statistics(
                        [_value_key(axis.mastery_mean) for axis in transitive_prerequisite_axes],
                        baseline_size=baseline,
                    ),
                ),
                books_with_transitive_only_additions=sum(
                    set(book.transitive_prerequisites) != set(book.direct_prerequisites)
                    for book in books
                ),
                two_stage=CombinedVariantSummary(
                    groups=_group_statistics(two_stage_keys, baseline_size=baseline),
                    status_counts=dict(
                        sorted(Counter(book.two_stage_status for book in books).items())
                    ),
                ),
                unweighted_combination=CombinedVariantSummary(
                    groups=_group_statistics(combinations, baseline_size=baseline),
                ),
            )
        )
        representatives.extend(_representative_cases(topic, books))
        pairs.extend(_pair_packet(topic, books))

    return ConceptReadinessRankingExperimentReport(
        experiment_version=EXPERIMENT_VERSION,
        production_ranking_modified=False,
        prose_difficulty_used=False,
        occurrence_count_used_as_weight=False,
        mapping_report_version=mapping.report_version,
        mapping_report_hash=mapping_report_hash,
        matching_config_version=matching.config.config_version,
        matching_config_hash=matching.content_hash,
        minimum_readiness_assessment_coverage=(
            matching.config.policy.minimum_readiness_assessment_coverage
        ),
        minimum_opportunity_assessment_coverage=(
            matching.config.policy.minimum_opportunity_assessment_coverage
        ),
        minimum_readiness_score=matching.config.policy.minimum_readiness_score,
        accepted_graph=projection,
        reader_profile_hashes=dict(sorted(reader_profile_hashes.items())),
        input_hashes=dict(sorted(input_hashes.items())),
        topics=topic_summaries,
        books=sorted(diagnostics, key=lambda book: (book.topic_id, book.book_id)),
        representative_cases=representatives,
        human_pair_review_packet=pairs,
    )
