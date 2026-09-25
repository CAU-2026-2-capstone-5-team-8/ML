"""Tests for the isolated exact prerequisite-first ranking candidate."""

from pathlib import Path
from types import SimpleNamespace

import pytest
from typer.testing import CliRunner

from bookmatch_ml.cli import app
from bookmatch_ml.concept_v2.book_evidence_mapping import (
    BookConceptPresence,
    BookEvidenceConceptMapping,
    BookEvidenceConceptMappingReport,
)
from bookmatch_ml.concept_v2.graph import load_concept_graph
from bookmatch_ml.concept_v2.profile import load_concept_matching_config
from bookmatch_ml.concept_v2.validation import load_concept_graph_reviews
from bookmatch_ml.config import load_feature_config
from bookmatch_ml.ranking.concept_readiness_experiments import (
    ConceptReadinessBookDiagnostic,
    DiagnosticSnapshot,
    DirectConceptAxis,
    HumanBookPair,
    ReadinessAxis,
    build_accepted_graph_projection,
    evaluate_concept_readiness_ranking,
    prerequisite_ancestors,
)
from bookmatch_ml.ranking.multi_reader_concept_experiments import (
    build_scenario_profiles,
    load_multi_reader_concept_experiment_config,
)
from bookmatch_ml.ranking.prerequisite_first_candidate import (
    CANDIDATE_VERSION,
    _candidate_prediction,
    build_candidate_ranking,
    evaluate_candidate_human_pairs,
)

ROOT = Path(__file__).resolve().parents[1]
FEATURES = load_feature_config(ROOT / "configs/features.yaml")
GRAPH = load_concept_graph(ROOT / "configs/concept_graph.yaml", FEATURES)
REVIEWS = load_concept_graph_reviews(ROOT / "configs/concept_graph_reviews.yaml", GRAPH)
MATCHING = load_concept_matching_config(ROOT / "configs/concept_matching_v2.yaml")
EXPERIMENT = load_multi_reader_concept_experiment_config(
    ROOT / "configs/multi_reader_concept_ranking_v1.yaml"
)
PROJECTION = build_accepted_graph_projection(GRAPH, REVIEWS)
HASH = "sha256:" + "a" * 64


def _axis(
    total: int,
    assessed: int,
    mastery: float | None,
) -> ReadinessAxis:
    """Build a readiness axis with internally consistent coverage semantics."""

    if total == 0:
        return ReadinessAxis(
            total_concept_count=0,
            assessed_concept_count=0,
            assessment_coverage=None,
            mastery_mean=None,
            semantics="not_applicable",
        )
    return ReadinessAxis(
        total_concept_count=total,
        assessed_concept_count=assessed,
        assessment_coverage=assessed / total,
        mastery_mean=mastery,
        semantics="calculated" if mastery is not None else "unassessed",
    )


def _diagnostic(
    book_id: str,
    *,
    readiness: float | None,
    opportunity: float | None,
    covered_count: int = 1,
    prerequisite_count: int = 1,
    prerequisite_assessed: int | None = None,
) -> ConceptReadinessBookDiagnostic:
    """Build one compact diagnostic fixture for candidate-only behavior."""

    assessed = prerequisite_count if prerequisite_assessed is None else prerequisite_assessed
    direct_assessed = covered_count if opportunity is not None else 0
    direct_readiness = None if opportunity is None else 1 - opportunity
    direct_axis = DirectConceptAxis(
        **_axis(covered_count, direct_assessed, direct_readiness).model_dump(),
        learning_opportunity=opportunity,
    )
    prerequisite_axis = _axis(prerequisite_count, assessed, readiness)
    covered = [f"covered-{index}" for index in range(covered_count)]
    prerequisites = [f"prerequisite-{index}" for index in range(prerequisite_count)]
    return ConceptReadinessBookDiagnostic(
        book_id=book_id,
        title=f"Title {book_id}",
        topic_id="operating-systems",
        covered_concepts=covered,
        assessed_covered_mastery={
            concept: direct_readiness for concept in covered[:direct_assessed]
        },
        direct=direct_axis,
        direct_prerequisites=prerequisites,
        transitive_prerequisites=prerequisites,
        assessed_direct_prerequisite_mastery={
            concept: readiness for concept in prerequisites[:assessed]
        },
        assessed_transitive_prerequisite_mastery={
            concept: readiness for concept in prerequisites[:assessed]
        },
        direct_prerequisite_readiness=prerequisite_axis,
        transitive_prerequisite_readiness=prerequisite_axis,
        two_stage_status="eligible",
        unweighted_combination=(
            (readiness + opportunity) / 2
            if readiness is not None and opportunity is not None
            else None
        ),
    )


def _presence(topic: str, concept: str, support_count: int = 1) -> BookConceptPresence:
    """Build one deduplicated concept-presence row."""

    return BookConceptPresence(
        topic_id=topic,
        concept_id=concept,
        supporting_evidence=[],
        support_count=support_count,
    )


def _mapping_book(
    book_id: str,
    title: str,
    topic: str,
    concepts: list[str],
    *,
    support_count: int = 1,
) -> BookEvidenceConceptMapping:
    """Build one source-aware mapping fixture without source weighting."""

    rows = [_presence(topic, concept, support_count) for concept in concepts]
    return BookEvidenceConceptMapping(
        book_id=book_id,
        title=title,
        topics=[topic],
        matcher_version="normalized_alias_span_v2",
        model_version="concept-matching-v2-overlap",
        matching_config_version="concept-matching-config-v3",
        matching_config_hash=HASH,
        evidence_row_count=max(1, len(rows)),
        matched_evidence_row_count=len(rows),
        ambiguous_evidence_row_count=0,
        concept_presence_count=len(rows),
        raw_match_occurrence_count=sum(row.support_count for row in rows),
        concepts=rows,
    )


def _mapping(books: list[BookEvidenceConceptMapping]) -> BookEvidenceConceptMappingReport:
    """Build a strict mapping report from compact fixture books."""

    return BookEvidenceConceptMappingReport(
        report_version="book-evidence-concept-presence-v1",
        book_evidence_contract_version="book-evidence-v1",
        book_evidence_hash=HASH,
        matcher_version="normalized_alias_span_v2",
        model_version="concept-matching-v2-overlap",
        matching_config_version="concept-matching-config-v3",
        matching_config_hash=HASH,
        total_books=len(books),
        books_with_matched_concepts=sum(bool(book.concepts) for book in books),
        books_without_matched_concepts=sum(not book.concepts for book in books),
        unique_book_concept_presence_count=sum(len(book.concepts) for book in books),
        raw_match_occurrence_count=sum(book.raw_match_occurrence_count for book in books),
        books=books,
    )


def _snapshot(
    book_id: str,
    prerequisite: float,
    opportunity: float | None,
) -> DiagnosticSnapshot:
    """Build the fixed fields used by exact human-pair prediction."""

    return DiagnosticSnapshot(
        book_id=book_id,
        title=book_id,
        covered_concepts=[],
        inferred_prerequisites=[],
        assessed_covered_mastery={},
        assessed_prerequisite_mastery={},
        direct_coverage=1,
        direct_learning_opportunity=opportunity,
        prerequisite_coverage=1,
        prerequisite_readiness=prerequisite,
        two_stage_status="eligible",
        unweighted_combination=(
            (prerequisite + opportunity) / 2 if opportunity is not None else None
        ),
    )


def test_projection_uses_only_accepted_edges_and_transitive_sets_are_deduplicated() -> None:
    """Rejected edges stay absent and converging graph paths do not duplicate concepts."""

    statuses = {
        (row.topic, row.prerequisite, row.dependent): row.review_status
        for row in REVIEWS.config.reviews
    }
    projection_keys = {(edge.topic, edge.prerequisite, edge.dependent) for edge in PROJECTION.edges}
    ancestors = prerequisite_ancestors(
        PROJECTION, "operating-systems", "concurrency", transitive=True
    )

    assert len(PROJECTION.edges) == 18
    assert all(statuses[key] == "accepted" for key in projection_keys)
    assert not projection_keys & {
        key for key, status in statuses.items() if status in {"rejected", "needs_revision"}
    }
    assert ancestors == {"process", "thread"}


def test_exact_lexicographic_ordering_and_stable_book_id_tiebreak() -> None:
    """Readiness is primary, opportunity secondary, and ID only stabilizes exact ties."""

    diagnostics = [
        _diagnostic("z-tie", readiness=0.8, opportunity=0.6),
        _diagnostic("lower", readiness=0.7, opportunity=0.99),
        _diagnostic("a-tie", readiness=0.8, opportunity=0.6),
        _diagnostic("secondary", readiness=0.8, opportunity=0.4),
    ]
    ranking = build_candidate_ranking("operating-systems", "fixture", "fixture", HASH, diagnostics)
    personalized = [book for book in ranking.books if book.rank is not None]

    assert [book.book_id for book in personalized] == [
        "a-tie",
        "z-tie",
        "secondary",
        "lower",
    ]
    assert [book.ordering_group for book in personalized] == [1, 1, 2, 3]
    assert ranking.statistics.final_ordering_group_count == 3
    assert ranking.statistics.largest_tie_group_size == 2


def test_classification_coverage_and_unknown_readiness_are_preserved() -> None:
    """All three pools preserve counts and never turn unknown readiness into zero."""

    diagnostics = [
        _diagnostic(
            "personalizable",
            readiness=0.75,
            opportunity=0.4,
            prerequisite_count=4,
            prerequisite_assessed=3,
        ),
        _diagnostic(
            "unassessed-prerequisite",
            readiness=None,
            opportunity=0.5,
            prerequisite_count=2,
            prerequisite_assessed=0,
        ),
        _diagnostic(
            "no-prerequisite",
            readiness=None,
            opportunity=0.5,
            prerequisite_count=0,
        ),
        _diagnostic(
            "no-evidence",
            readiness=None,
            opportunity=None,
            covered_count=0,
            prerequisite_count=0,
        ),
    ]
    ranking = build_candidate_ranking("operating-systems", "fixture", "fixture", HASH, diagnostics)
    by_id = {book.book_id: book for book in ranking.books}

    assert ranking.statistics.personalizable_count == 1
    assert ranking.statistics.concept_only_count == 2
    assert ranking.statistics.evidence_unavailable_count == 1
    assert by_id["personalizable"].prerequisite_coverage == 0.75
    assert by_id["unassessed-prerequisite"].prerequisite_readiness is None
    assert by_id["unassessed-prerequisite"].rank is None
    assert by_id["no-evidence"].availability_status == "evidence_unavailable"


def test_candidate_is_input_order_independent_and_does_not_use_occurrence_counts() -> None:
    """Only diagnostic concept sets matter; input order and evidence frequency do not."""

    first = [
        _diagnostic("book-b", readiness=0.6, opportunity=0.7),
        _diagnostic("book-a", readiness=0.8, opportunity=0.2),
    ]
    second = list(reversed(first))

    assert build_candidate_ranking(
        "operating-systems", "fixture", "fixture", HASH, first
    ) == build_candidate_ranking("operating-systems", "fixture", "fixture", HASH, second)
    assert not hasattr(first[0], "support_count")


def test_readiness_and_opportunity_are_calculated_from_assessed_concept_sets() -> None:
    """The reused evaluator calculates the two axes and complete scenario coverage."""

    profile = next(
        item
        for item in build_scenario_profiles(EXPERIMENT, PROJECTION)
        if item.reader_profile.topic_id == "operating-systems" and item.scenario_id == "beginner"
    )
    mapping = _mapping(
        [
            _mapping_book(
                "book",
                "Book",
                "operating-systems",
                ["deadlock", "virtual memory"],
                support_count=99,
            )
        ]
    )
    report = evaluate_concept_readiness_ranking(
        mapping,
        [profile.reader_profile],
        GRAPH,
        REVIEWS,
        MATCHING,
        mapping_report_hash=HASH,
        reader_profile_hashes={"operating-systems": profile.reader_profile_hash},
        input_hashes={"fixture": HASH},
    )
    book = report.books[0]

    assert book.direct.assessment_coverage == 1
    assert book.direct.learning_opportunity == 0.95
    assert book.transitive_prerequisite_readiness.assessment_coverage == 1
    assert book.transitive_prerequisite_readiness.mastery_mean == pytest.approx(0.2833333333)
    assert report.occurrence_count_used_as_weight is False


def test_scenario_change_can_reverse_exact_candidate_ordering() -> None:
    """The same two real-shaped LA books exchange order as readiness changes."""

    mapping = _mapping(
        [
            _mapping_book(
                "linear-algebra-dot",
                "Linear algebra.",
                "linear-algebra",
                [
                    "determinant",
                    "diagonalization",
                    "eigenvalue",
                    "eigenvector",
                    "matrix",
                    "vector space",
                ],
            ),
            _mapping_book(
                "applications",
                "Linear algebra with applications",
                "linear-algebra",
                [
                    "basis",
                    "determinant",
                    "diagonalization",
                    "eigenvalue",
                    "eigenvector",
                    "gaussian elimination",
                    "inner product",
                    "linear system",
                    "matrix",
                    "rank",
                    "systems of equations",
                    "vector",
                    "vector space",
                ],
            ),
        ]
    )
    profiles = build_scenario_profiles(EXPERIMENT, PROJECTION)
    ranked: dict[str, list[str]] = {}
    for scenario in ("beginner", "advanced"):
        profile = next(
            item
            for item in profiles
            if item.reader_profile.topic_id == "linear-algebra" and item.scenario_id == scenario
        )
        report = evaluate_concept_readiness_ranking(
            mapping,
            [profile.reader_profile],
            GRAPH,
            REVIEWS,
            MATCHING,
            mapping_report_hash=HASH,
            reader_profile_hashes={"linear-algebra": profile.reader_profile_hash},
            input_hashes={"fixture": HASH},
        )
        result = build_candidate_ranking(
            "linear-algebra",
            scenario,
            profile.description,
            profile.reader_profile_hash,
            report.books,
        )
        ranked[scenario] = [book.book_id for book in result.books if book.rank is not None]

    assert ranked["beginner"] == ["applications", "linear-algebra-dot"]
    assert ranked["advanced"] == ["linear-algebra-dot", "applications"]


def test_frozen_pair_labels_are_unchanged_and_exact_candidate_agrees_nine_of_ten() -> None:
    """Candidate comparison validates membership and never edits the frozen labels."""

    values = [
        ((0.95, 0.5), (0.7333333, 0.4166667)),
        ((0.95, 0.5), (0.9, 0.3571429)),
        ((0.9, 0.3571429), (0.7428571, 0.455)),
        ((0.88, 0.3772727), (0.7428571, 0.455)),
        ((0.7333333, 0.4166667), (0.9, 0.3571429)),
        ((0.9, 0.1), (0.575, 0.24)),
        ((0.62, 0.3428571), (0.9, 0.1)),
        ((0.8666667, 0.1), (0.6, 0.18)),
        ((0.8666667, 0.1), (0.575, 0.24)),
        ((0.6, 0.18), (0.62, 0.2333333)),
    ]
    pairs = []
    original_labels = [pair.human_preference for pair in EXPERIMENT.config.frozen_human_pairs]
    for frozen, (left, right) in zip(EXPERIMENT.config.frozen_human_pairs, values, strict=True):
        pairs.append(
            HumanBookPair(
                pair_id=frozen.pair_id,
                topic_id=frozen.topic_id,
                selection_reason="frozen fixture",
                left=_snapshot(frozen.left_book_id, left[0], left[1]),
                right=_snapshot(frozen.right_book_id, right[0], right[1]),
            )
        )
    evaluations, summaries = evaluate_candidate_human_pairs(
        SimpleNamespace(human_pair_review_packet=pairs), EXPERIMENT.config
    )
    by_topic = {summary.topic_id: summary.agreement_count for summary in summaries}

    assert [item.human_preference for item in evaluations] == original_labels
    assert by_topic == {"linear-algebra": 5, "operating-systems": 4, None: 9}


def test_pair_prediction_matches_missing_secondary_ordering() -> None:
    """Known opportunity ranks ahead of missing opportunity at equal readiness."""

    known = _snapshot("known", 0.8, 0.4)
    unknown = _snapshot("unknown", 0.8, None)

    assert _candidate_prediction(known, unknown) == "A"
    assert _candidate_prediction(unknown, known) == "B"
    assert _candidate_prediction(unknown, unknown) == "tie"


def test_demo_cli_smoke_uses_mapping_artifact_and_prints_fallback_counts(
    tmp_path: Path,
) -> None:
    """The meeting CLI reads a mapping artifact and emits concise candidate output."""

    mapping = _mapping(
        [
            _mapping_book(
                "personalized-book",
                "Personalized Book",
                "operating-systems",
                ["deadlock"],
            ),
            _mapping_book(
                "fallback-book",
                "Fallback Book",
                "operating-systems",
                [],
            ),
        ]
    )
    mapping_path = tmp_path / "mapping.json"
    mapping_path.write_text(mapping.model_dump_json(indent=2), encoding="utf-8")

    result = CliRunner().invoke(
        app,
        [
            "demo-concept-recommendation",
            "--topic",
            "operating-systems",
            "--scenario",
            "beginner",
            "--concept-mapping",
            str(mapping_path),
            "--limit",
            "1",
        ],
    )

    assert result.exit_code == 0, result.output
    assert CANDIDATE_VERSION in result.output
    assert "Personalized ranking available: 1/2" in result.output
    assert "Fallback / unavailable: 1/2" in result.output
    assert "Personalized Book" in result.output
