"""Regression tests for the isolated concept-readiness ranking experiment."""

import json
from pathlib import Path

import pytest
from pydantic import ValidationError
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
from bookmatch_ml.config import load_feature_config, load_reader_config
from bookmatch_ml.io import write_json
from bookmatch_ml.ranking.concept_readiness_experiments import (
    AcceptedGraphProjection,
    AcceptedPrerequisiteEdge,
    build_accepted_graph_projection,
    evaluate_concept_readiness_ranking,
    prerequisite_ancestors,
)
from bookmatch_ml.reader.profile import build_reader_profile, load_assessment

ROOT = Path(__file__).resolve().parents[1]
FEATURES = load_feature_config(ROOT / "configs/features.yaml")
GRAPH = load_concept_graph(ROOT / "configs/concept_graph.yaml", FEATURES)
REVIEWS = load_concept_graph_reviews(ROOT / "configs/concept_graph_reviews.yaml", GRAPH)
MATCHING = load_concept_matching_config(ROOT / "configs/concept_matching_v2.yaml")
READER_CONFIG = load_reader_config(ROOT / "configs/reader.yaml")
HASH = "sha256:" + "a" * 64


def _readers():
    return [
        build_reader_profile(load_assessment(ROOT / "examples/assessment.json"), READER_CONFIG),
        build_reader_profile(
            load_assessment(ROOT / "examples/concept_matching_la_assessment.json"),
            READER_CONFIG,
        ),
    ]


def _presence(topic: str, concept: str, support_count: int = 1) -> BookConceptPresence:
    return BookConceptPresence(
        topic_id=topic,
        concept_id=concept,
        supporting_evidence=[],
        support_count=support_count,
    )


def _book(
    book_id: str,
    title: str,
    topic: str,
    concepts: list[str],
    *,
    support_count: int = 1,
) -> BookEvidenceConceptMapping:
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


def _mapping(*, reverse: bool = False, support_count: int = 1):
    books = [
        _book(
            "os-book-1",
            "OS Concepts",
            "operating-systems",
            ["virtual memory", "deadlock"],
            support_count=support_count,
        ),
        _book(
            "os-book-2",
            "Programming Background",
            "operating-systems",
            ["programming"],
            support_count=support_count,
        ),
        _book(
            "la-book-1",
            "Linear Algebra Concepts",
            "linear-algebra",
            ["basis", "least squares"],
            support_count=support_count,
        ),
        _book("la-book-2", "Linear Algebra Metadata", "linear-algebra", []),
    ]
    if reverse:
        books = list(reversed(books))
        books = [
            book.model_copy(update={"concepts": list(reversed(book.concepts))}) for book in books
        ]
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


def _report(mapping=None, readers=None):
    return evaluate_concept_readiness_ranking(
        mapping or _mapping(),
        readers or _readers(),
        GRAPH,
        REVIEWS,
        MATCHING,
        mapping_report_hash=HASH,
        reader_profile_hashes={reader.topic_id: HASH for reader in (readers or _readers())},
        input_hashes={"fixture": HASH},
    )


def test_projection_uses_only_accepted_edges_and_is_sorted() -> None:
    projection = build_accepted_graph_projection(GRAPH, REVIEWS)
    decisions = {
        (row.topic, row.prerequisite, row.dependent): row.review_status
        for row in REVIEWS.config.reviews
    }

    assert len(projection.edges) == 18
    assert all(
        decisions[(edge.topic, edge.prerequisite, edge.dependent)] == "accepted"
        for edge in projection.edges
    )
    assert projection.edges == sorted(
        projection.edges, key=lambda edge: (edge.topic, edge.prerequisite, edge.dependent)
    )
    projected = {(edge.topic, edge.prerequisite, edge.dependent) for edge in projection.edges}
    assert ("linear-algebra", "vector space", "dimension") not in projected
    assert ("linear-algebra", "systems of equations", "linear system") not in projected


def test_projection_rejects_cycle_and_transitive_ancestors_are_deduplicated() -> None:
    projection = build_accepted_graph_projection(GRAPH, REVIEWS)

    assert prerequisite_ancestors(
        projection, "operating-systems", "deadlock", transitive=False
    ) == {"synchronization"}
    assert prerequisite_ancestors(projection, "operating-systems", "deadlock", transitive=True) == {
        "process",
        "thread",
        "concurrency",
        "synchronization",
    }
    with pytest.raises(ValidationError, match="cycle"):
        AcceptedGraphProjection(
            graph_version="cycle-test",
            graph_hash=HASH,
            review_config_version="cycle-test",
            review_config_hash=HASH,
            nodes={"topic": ["a", "b"]},
            edges=[
                AcceptedPrerequisiteEdge(topic="topic", prerequisite="a", dependent="b"),
                AcceptedPrerequisiteEdge(topic="topic", prerequisite="b", dependent="a"),
            ],
        )


def test_axes_keep_unknown_readiness_missing_and_calculate_coverage() -> None:
    report = _report()
    os_book = next(book for book in report.books if book.book_id == "os-book-1")
    metadata_only = next(book for book in report.books if book.book_id == "la-book-2")

    assert os_book.direct.total_concept_count == 2
    assert os_book.direct.assessed_concept_count == 1
    assert os_book.direct.assessment_coverage == 0.5
    assert os_book.direct.learning_opportunity == pytest.approx(0.5)
    assert os_book.transitive_prerequisites == [
        "computer architecture",
        "concurrency",
        "memory management",
        "process",
        "synchronization",
        "thread",
    ]
    assert os_book.transitive_prerequisite_readiness.assessed_concept_count == 5
    assert os_book.transitive_prerequisite_readiness.assessment_coverage == pytest.approx(5 / 6)
    assert "thread" not in os_book.assessed_transitive_prerequisite_mastery
    assert metadata_only.direct.semantics == "not_applicable"
    assert metadata_only.direct.learning_opportunity is None
    assert metadata_only.unweighted_combination is None


def test_direct_and_transitive_prerequisites_are_reported_separately() -> None:
    report = _report()
    os_book = next(book for book in report.books if book.book_id == "os-book-1")

    assert set(os_book.direct_prerequisites) == {"memory management", "synchronization"}
    assert set(os_book.direct_prerequisites) < set(os_book.transitive_prerequisites)
    os_summary = next(topic for topic in report.topics if topic.topic_id == "operating-systems")
    assert os_summary.books_with_transitive_only_additions == 1


def test_result_is_input_order_independent_and_ignores_occurrence_count() -> None:
    readers = _readers()
    first = _report(_mapping(support_count=1), readers)
    second = _report(_mapping(reverse=True, support_count=99), list(reversed(readers)))

    assert first == second
    assert first.occurrence_count_used_as_weight is False
    assert first.production_ranking_modified is False
    assert first.prose_difficulty_used is False


def test_cli_writes_byte_identical_report_and_blind_pair_packet(tmp_path: Path) -> None:
    mapping_path = tmp_path / "mapping.json"
    os_reader_path = tmp_path / "os-reader.json"
    la_reader_path = tmp_path / "la-reader.json"
    write_json(_mapping(), mapping_path)
    readers = _readers()
    write_json(
        next(reader for reader in readers if reader.topic_id == "operating-systems"),
        os_reader_path,
    )
    write_json(
        next(reader for reader in readers if reader.topic_id == "linear-algebra"),
        la_reader_path,
    )
    outputs = []
    for name in ("first", "second"):
        output = tmp_path / f"{name}.json"
        result = CliRunner().invoke(
            app,
            [
                "evaluate-concept-readiness-ranking",
                "--concept-mapping",
                str(mapping_path),
                "--reader",
                str(os_reader_path),
                "--reader",
                str(la_reader_path),
                "--output",
                str(output),
            ],
        )
        assert result.exit_code == 0, result.output
        outputs.append(output.read_bytes())

    assert outputs[0] == outputs[1]
    report = json.loads(outputs[0])
    assert all(pair["human_preference"] is None for pair in report["human_pair_review_packet"])
    assert all(pair["review_note"] is None for pair in report["human_pair_review_packet"])
