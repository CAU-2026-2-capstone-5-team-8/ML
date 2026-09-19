"""Human-readable structural audit of unchanged concept-matching v2 inputs."""

from collections import Counter, defaultdict
from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from bookmatch_ml.concept_v2.graph import LoadedConceptGraph
from bookmatch_ml.concept_v2.matching import match_book_concepts
from bookmatch_ml.concept_v2.profile import (
    BookConceptProfileV2,
    LoadedConceptMatchingConfig,
    PrerequisiteCandidate,
    build_book_concept_profile_v2,
)
from bookmatch_ml.concept_v2.toc import reconstruct_toc
from bookmatch_ml.config import LoadedFeatureConfig
from bookmatch_ml.schemas import BookEvidence, ReaderProfile


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class FirstTocOccurrence(_Strict):
    toc_entry_id: str
    toc_title: str
    toc_path: list[str]
    toc_level: int = Field(ge=1)
    traversal_position: int = Field(ge=0)
    matching_alias: str
    occurrence_count: int = Field(ge=1)


class EdgeBookObservation(_Strict):
    book_id: str
    book_title: str
    state: Literal[
        "observed_before_in_toc",
        "observed_at_or_after_in_toc",
        "prerequisite_not_mapped",
        "dependent_not_mapped",
    ]
    prerequisite_first: FirstTocOccurrence | None
    dependent_first: FirstTocOccurrence | None


class GraphEdgeReviewRow(_Strict):
    topic: str
    prerequisite: str
    dependent: str
    relation_type: str
    source_type: str
    edge_version: str
    related_book_count: int = Field(ge=0)
    dependent_book_count: int = Field(ge=0)
    dependent_book_ids: list[str]
    jointly_mapped_book_count: int = Field(ge=0)
    books: list[EdgeBookObservation]


class MatchedTocReviewRow(_Strict):
    book_id: str
    book_title: str
    topic: str
    toc_entry_id: str
    source_id: str
    toc_title: str
    toc_path: list[str]
    concept_id: str
    matching_alias: str
    match_method: str
    toc_level: int
    traversal_position: int
    concept_occurrence_count_in_book: int = Field(ge=1)


class UnmatchedTocReviewRow(_Strict):
    book_id: str
    book_title: str
    topic: str
    toc_entry_id: str
    toc_title: str
    toc_path: list[str]
    reason: Literal["unmatched", "ambiguous_alias"]
    traversal_position: int


class ProjectedCandidateReviewRow(_Strict):
    concept_id: str
    reason: Literal["taught_before_use_candidate", "external_or_late_candidate"]
    prerequisite_first: FirstTocOccurrence | None
    target_first_occurrences: list[FirstTocOccurrence]
    graph_paths: list[list[str]]
    relation_source_types: list[str]


class BookMappingReview(_Strict):
    book_id: str
    book_title: str
    topic: str
    total_toc_entries: int = Field(ge=0)
    matched_toc_entries: int = Field(ge=0)
    unmatched_toc_entries: int = Field(ge=0)
    ambiguous_toc_entries: int = Field(ge=0)
    distinct_mapped_concepts: int = Field(ge=0)
    graph_node_count: int = Field(ge=1)
    mapping_rate: float | None = Field(ge=0, le=1)
    distinct_concept_coverage: float = Field(ge=0, le=1)
    representative_unmatched_titles: list[str]
    matched_entries: list[MatchedTocReviewRow]
    unmatched_entries: list[UnmatchedTocReviewRow]
    projected_candidates: list[ProjectedCandidateReviewRow]
    readiness_assessment_coverage: float | None = Field(default=None, ge=0, le=1)
    opportunity_assessment_coverage: float | None = Field(default=None, ge=0, le=1)
    learning_candidate_count: int | None = Field(default=None, ge=0)
    known_concept_count: int | None = Field(default=None, ge=0)
    unknown_mastery_count: int | None = Field(default=None, ge=0)
    measured_learning_weight: float | None = Field(default=None, ge=0)
    total_book_concept_weight: float = Field(ge=0)


class TopicStructuralMetrics(_Strict):
    topic: str
    book_count: int = Field(ge=0)
    total_toc_entries: int = Field(ge=0)
    matched_toc_entries: int = Field(ge=0)
    unmatched_toc_entries: int = Field(ge=0)
    ambiguous_toc_entries: int = Field(ge=0)
    mapping_rate: float | None = Field(ge=0, le=1)
    distinct_mapped_concepts_across_books: int = Field(ge=0)
    graph_node_count: int = Field(ge=1)
    graph_edge_count: int = Field(ge=0)
    graph_edges_jointly_mapped_count: int = Field(ge=0)
    graph_edges_not_jointly_mapped_count: int = Field(ge=0)
    first_occurrence_order_distribution: dict[str, int]
    mean_readiness_assessment_coverage: float | None = Field(default=None, ge=0, le=1)
    mean_opportunity_assessment_coverage: float | None = Field(default=None, ge=0, le=1)


class ConceptValidationReport(_Strict):
    artifact_version: Literal["concept-validation-v1"]
    graph_version: str
    graph_hash: str
    feature_config_hash: str
    matching_config_hash: str
    input_hashes: dict[str, str]
    topics: list[TopicStructuralMetrics]
    graph_edges: list[GraphEdgeReviewRow]
    books: list[BookMappingReview]


ReviewStatus = Literal["unreviewed", "accepted", "rejected", "uncertain"]


class EdgeReviewDecision(_Strict):
    topic: str
    prerequisite: str
    dependent: str
    source_type: Literal["proposed_seed", "human_reviewed_seed"]
    edge_version: str
    status: ReviewStatus
    reviewer_id: str | None = None
    reviewed_at: datetime | None = None
    rationale: str | None = None

    @model_validator(mode="after")
    def decision_has_review_metadata(self) -> "EdgeReviewDecision":
        details = (self.reviewer_id, self.reviewed_at, self.rationale)
        if self.status == "unreviewed" and any(value is not None for value in details):
            raise ValueError("unreviewed edges must not carry review metadata")
        if self.status != "unreviewed" and (
            not self.reviewer_id or not self.rationale or self.reviewed_at is None
        ):
            raise ValueError("reviewed edges require reviewer_id, reviewed_at, and rationale")
        if self.reviewed_at is not None and self.reviewed_at.utcoffset() is None:
            raise ValueError("reviewed_at must include a timezone")
        return self


class ConceptGraphReviewFile(_Strict):
    review_schema_version: Literal["concept-graph-review-v1"]
    graph_version: str
    graph_hash: str
    edges: list[EdgeReviewDecision]

    @model_validator(mode="after")
    def edges_are_unique(self) -> "ConceptGraphReviewFile":
        keys = [(edge.topic, edge.prerequisite, edge.dependent) for edge in self.edges]
        if len(keys) != len(set(keys)):
            raise ValueError("duplicate edge review")
        return self


def build_review_template(graph: LoadedConceptGraph) -> ConceptGraphReviewFile:
    """Emit an unreviewed template; review decisions never enter matching code."""

    return ConceptGraphReviewFile(
        review_schema_version="concept-graph-review-v1",
        graph_version=graph.graph.graph_version,
        graph_hash=graph.content_hash,
        edges=[
            EdgeReviewDecision(
                topic=edge.topic,
                prerequisite=edge.prerequisite,
                dependent=edge.dependent,
                source_type=edge.source_type,
                edge_version=edge.version,
                status="unreviewed",
            )
            for edge in sorted(
                graph.graph.edges,
                key=lambda item: (item.topic, item.prerequisite, item.dependent),
            )
        ],
    )


def _first_occurrence(profile: BookConceptProfileV2, concept: str) -> FirstTocOccurrence | None:
    matches = [item for item in profile.toc_mappings if item.concept_id == concept]
    if not matches:
        return None
    first = min(matches, key=lambda item: (item.traversal_position, item.toc_entry_id))
    return FirstTocOccurrence(
        toc_entry_id=first.toc_entry_id,
        toc_title=first.toc_title,
        toc_path=first.toc_path,
        toc_level=first.toc_level,
        traversal_position=first.traversal_position,
        matching_alias=first.matching_alias,
        occurrence_count=len(matches),
    )


def _representative_titles(rows: list[UnmatchedTocReviewRow]) -> list[str]:
    """Show one title per root branch first, then fill in TOC order to twelve."""

    sample: list[UnmatchedTocReviewRow] = []
    roots: set[str] = set()
    titles: set[str] = set()
    for row in rows:
        root = row.toc_path[0]
        if root not in roots and row.toc_title not in titles:
            sample.append(row)
            roots.add(root)
            titles.add(row.toc_title)
        if len(sample) == 12:
            break
    for row in rows:
        if len(sample) == 12:
            break
        if row.toc_title not in titles:
            sample.append(row)
            titles.add(row.toc_title)
    return [row.toc_title for row in sample]


def _candidate_review(
    candidate: PrerequisiteCandidate, profile: BookConceptProfileV2
) -> ProjectedCandidateReviewRow:
    targets = [
        occurrence
        for concept in candidate.related_target_concepts
        if (occurrence := _first_occurrence(profile, concept)) is not None
    ]
    return ProjectedCandidateReviewRow(
        concept_id=candidate.concept_id,
        reason=candidate.reason,
        prerequisite_first=_first_occurrence(profile, candidate.concept_id),
        target_first_occurrences=sorted(
            targets, key=lambda item: (item.traversal_position, item.toc_entry_id)
        ),
        graph_paths=candidate.graph_paths,
        relation_source_types=candidate.relation_source_types,
    )


def _book_review(
    evidence: BookEvidence,
    profile: BookConceptProfileV2,
    graph: LoadedConceptGraph,
    matching: LoadedConceptMatchingConfig,
    reader: ReaderProfile | None,
) -> BookMappingReview:
    positions = {
        visit.entry.toc_entry_id: visit.traversal_position
        for visit in reconstruct_toc(evidence.toc).traversal
    }
    occurrences = Counter(item.concept_id for item in profile.toc_mappings)
    matched = [
        MatchedTocReviewRow(
            book_id=profile.book_id,
            book_title=profile.title,
            topic=profile.topic_id,
            toc_entry_id=item.toc_entry_id,
            source_id=item.source_id,
            toc_title=item.toc_title,
            toc_path=item.toc_path,
            concept_id=item.concept_id,
            matching_alias=item.matching_alias,
            match_method=item.match_method,
            toc_level=item.toc_level,
            traversal_position=item.traversal_position,
            concept_occurrence_count_in_book=occurrences[item.concept_id],
        )
        for item in profile.toc_mappings
    ]
    unmatched = [
        UnmatchedTocReviewRow(
            book_id=profile.book_id,
            book_title=profile.title,
            topic=profile.topic_id,
            toc_entry_id=item.toc_entry_id,
            toc_title=item.title,
            toc_path=item.toc_path,
            reason=item.reason,
            traversal_position=positions[item.toc_entry_id],
        )
        for item in profile.unmapped_entries
    ]
    counts = profile.diagnostics
    if counts.total_toc_entries != len(positions) or (
        counts.matched_toc_entries + len(unmatched) != counts.total_toc_entries
    ):
        raise ValueError(f"v2 TOC mapping counts disagree for {profile.book_id}")
    matched_ids = {item.toc_entry_id for item in matched}
    if len(matched_ids) != counts.matched_toc_entries or matched_ids & {
        item.toc_entry_id for item in unmatched
    }:
        raise ValueError(f"v2 TOC mapping partitions disagree for {profile.book_id}")
    axes = match_book_concepts(reader, profile, matching) if reader is not None else None
    mastery = (
        {item.concept_id: item.score for item in reader.concept_readiness}
        if reader is not None
        else {}
    )
    node_count = len(graph.graph.nodes[profile.topic_id])
    return BookMappingReview(
        book_id=profile.book_id,
        book_title=profile.title,
        topic=profile.topic_id,
        total_toc_entries=counts.total_toc_entries,
        matched_toc_entries=counts.matched_toc_entries,
        unmatched_toc_entries=counts.unmatched_toc_entries,
        ambiguous_toc_entries=counts.ambiguous_toc_entries,
        distinct_mapped_concepts=counts.concept_count,
        graph_node_count=node_count,
        mapping_rate=(counts.matched_toc_entries / counts.total_toc_entries)
        if counts.total_toc_entries
        else None,
        distinct_concept_coverage=counts.concept_count / node_count,
        representative_unmatched_titles=_representative_titles(unmatched),
        matched_entries=matched,
        unmatched_entries=unmatched,
        projected_candidates=[
            _candidate_review(candidate, profile)
            for candidate in (
                profile.taught_before_use_candidates + profile.prerequisite_requirements
            )
        ],
        readiness_assessment_coverage=axes.readiness.assessment_coverage if axes else None,
        opportunity_assessment_coverage=axes.opportunity.assessment_coverage if axes else None,
        learning_candidate_count=len(axes.learning_candidate_concepts) if axes else None,
        known_concept_count=len(axes.already_known_concepts) if axes else None,
        unknown_mastery_count=len(axes.unknown_mastery_concepts) if axes else None,
        measured_learning_weight=sum(
            item.coverage_weight * (1 - mastery[item.concept_id])
            for item in profile.covered_concepts
            if item.concept_id in mastery
        )
        if axes
        else None,
        total_book_concept_weight=sum(item.coverage_weight for item in profile.covered_concepts),
    )


def build_concept_validation_report(
    evidence: list[BookEvidence],
    graph: LoadedConceptGraph,
    features: LoadedFeatureConfig,
    matching: LoadedConceptMatchingConfig,
    readers: dict[str, ReaderProfile],
    toc_file_hash: str,
    input_hashes: dict[str, str],
) -> ConceptValidationReport:
    """Expose every v2 mapping and direct graph edge without judging its truth."""

    unknown_reader_topics = set(readers) - set(graph.graph.nodes)
    if unknown_reader_topics:
        raise ValueError(f"unsupported reader topics: {sorted(unknown_reader_topics)}")
    profiles_by_topic: dict[str, list[BookConceptProfileV2]] = defaultdict(list)
    book_reviews: list[BookMappingReview] = []
    for book in sorted(evidence, key=lambda item: item.book_id):
        for topic in sorted(set(book.metadata.topics) & set(graph.graph.nodes)):
            profile = build_book_concept_profile_v2(
                book, topic, features, graph, matching, toc_file_hash
            )
            profiles_by_topic[topic].append(profile)
            book_reviews.append(_book_review(book, profile, graph, matching, readers.get(topic)))
    book_reviews.sort(key=lambda item: (item.topic, item.book_id))

    edge_rows: list[GraphEdgeReviewRow] = []
    for edge in sorted(
        graph.graph.edges,
        key=lambda item: (item.topic, item.prerequisite, item.dependent),
    ):
        observations: list[EdgeBookObservation] = []
        for profile in sorted(profiles_by_topic[edge.topic], key=lambda item: item.book_id):
            before = _first_occurrence(profile, edge.prerequisite)
            after = _first_occurrence(profile, edge.dependent)
            if after is None:
                state = "dependent_not_mapped"
            elif before is None:
                state = "prerequisite_not_mapped"
            elif before.traversal_position < after.traversal_position:
                state = "observed_before_in_toc"
            else:
                state = "observed_at_or_after_in_toc"
            observations.append(
                EdgeBookObservation(
                    book_id=profile.book_id,
                    book_title=profile.title,
                    state=state,
                    prerequisite_first=before,
                    dependent_first=after,
                )
            )
        dependent_book_ids = [
            item.book_id for item in observations if item.dependent_first is not None
        ]
        edge_rows.append(
            GraphEdgeReviewRow(
                topic=edge.topic,
                prerequisite=edge.prerequisite,
                dependent=edge.dependent,
                relation_type=edge.relation_type,
                source_type=edge.source_type,
                edge_version=edge.version,
                related_book_count=sum(
                    item.prerequisite_first is not None or item.dependent_first is not None
                    for item in observations
                ),
                dependent_book_count=len(dependent_book_ids),
                dependent_book_ids=dependent_book_ids,
                jointly_mapped_book_count=sum(
                    item.prerequisite_first is not None and item.dependent_first is not None
                    for item in observations
                ),
                books=observations,
            )
        )

    topic_metrics: list[TopicStructuralMetrics] = []
    states = (
        "observed_before_in_toc",
        "observed_at_or_after_in_toc",
        "prerequisite_not_mapped",
        "dependent_not_mapped",
    )
    for topic, nodes in sorted(graph.graph.nodes.items()):
        books = [item for item in book_reviews if item.topic == topic]
        edges = [item for item in edge_rows if item.topic == topic]
        total = sum(item.total_toc_entries for item in books)
        matched = sum(item.matched_toc_entries for item in books)
        mapped_concepts = {row.concept_id for book in books for row in book.matched_entries}
        observed = sum(edge.jointly_mapped_book_count > 0 for edge in edges)
        distribution = Counter(observation.state for edge in edges for observation in edge.books)
        readiness_coverages = [
            item.readiness_assessment_coverage
            for item in books
            if item.readiness_assessment_coverage is not None
        ]
        opportunity_coverages = [
            item.opportunity_assessment_coverage
            for item in books
            if item.opportunity_assessment_coverage is not None
        ]
        topic_metrics.append(
            TopicStructuralMetrics(
                topic=topic,
                book_count=len(books),
                total_toc_entries=total,
                matched_toc_entries=matched,
                unmatched_toc_entries=sum(item.unmatched_toc_entries for item in books),
                ambiguous_toc_entries=sum(item.ambiguous_toc_entries for item in books),
                mapping_rate=(matched / total) if total else None,
                distinct_mapped_concepts_across_books=len(mapped_concepts),
                graph_node_count=len(nodes),
                graph_edge_count=len(edges),
                graph_edges_jointly_mapped_count=observed,
                graph_edges_not_jointly_mapped_count=len(edges) - observed,
                first_occurrence_order_distribution={
                    state: distribution[state] for state in states
                },
                mean_readiness_assessment_coverage=(
                    sum(readiness_coverages) / len(readiness_coverages)
                    if readiness_coverages
                    else None
                ),
                mean_opportunity_assessment_coverage=(
                    sum(opportunity_coverages) / len(opportunity_coverages)
                    if opportunity_coverages
                    else None
                ),
            )
        )
    return ConceptValidationReport(
        artifact_version="concept-validation-v1",
        graph_version=graph.graph.graph_version,
        graph_hash=graph.content_hash,
        feature_config_hash=features.content_hash,
        matching_config_hash=matching.content_hash,
        input_hashes=input_hashes,
        topics=topic_metrics,
        graph_edges=edge_rows,
        books=book_reviews,
    )
