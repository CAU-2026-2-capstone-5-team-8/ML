"""Production contract tests for exact prerequisite-first ranking-v2."""

import json
from pathlib import Path

import pytest

from bookmatch_ml.concept_v2.book_evidence_mapping import (
    BookConceptPresence,
    BookEvidenceConceptMapping,
    BookEvidenceConceptMappingReport,
)
from bookmatch_ml.concept_v2.graph import load_concept_graph
from bookmatch_ml.concept_v2.profile import load_concept_matching_config
from bookmatch_ml.concept_v2.validation import load_concept_graph_reviews
from bookmatch_ml.config import (
    load_feature_config,
    load_ranking_v2_config,
    load_reader_config,
)
from bookmatch_ml.ranking.concept_readiness_experiments import (
    build_accepted_graph_projection,
)
from bookmatch_ml.ranking.matching import to_matching_reader_profile
from bookmatch_ml.ranking.multi_reader_concept_experiments import (
    build_scenario_profiles,
    load_multi_reader_concept_experiment_config,
)
from bookmatch_ml.ranking.prerequisite_first_candidate import (
    evaluate_candidate_scenario,
)
from bookmatch_ml.ranking.prerequisite_first_v2 import (
    PrerequisiteFirstBookProfile,
    RankingV2TargetUnavailableError,
    build_prerequisite_first_book_profiles,
    build_ranking_v2_projection,
    rank_prerequisite_first_v2,
)
from bookmatch_ml.reader.profile import build_reader_profile, load_assessment
from bookmatch_ml.schemas import (
    MatchingBookProfile,
    MatchingConcept,
    MatchingConceptReadiness,
    MatchingReaderProfile,
)

ROOT = Path(__file__).resolve().parents[1]
RANKING_V2 = load_ranking_v2_config(ROOT / "configs/ranking_v2.yaml")
READER = load_reader_config(ROOT / "configs/reader.yaml")
FEATURES = load_feature_config(ROOT / "configs/features.yaml")
GRAPH = load_concept_graph(ROOT / "configs/concept_graph.yaml", FEATURES)
REVIEWS = load_concept_graph_reviews(ROOT / "configs/concept_graph_reviews.yaml", GRAPH)
MATCHING = load_concept_matching_config(ROOT / "configs/concept_matching_v2.yaml")
EXPERIMENT = load_multi_reader_concept_experiment_config(
    ROOT / "configs/multi_reader_concept_ranking_v1.yaml"
)
PROJECTION = build_accepted_graph_projection(GRAPH, REVIEWS)
HASH = "sha256:" + "a" * 64


def test_frozen_production_projection_matches_reviewed_source_artifacts() -> None:
    """The packaged server projection must not drift from the reviewed graph inputs."""

    frozen = build_ranking_v2_projection(RANKING_V2)

    assert frozen == PROJECTION


def test_actual_scale50_snapshot_produces_expected_pools_and_top_five() -> None:
    """The real 50-book concept snapshot preserves reviewed pool and Top-5 behavior."""

    payload = json.loads(
        (ROOT / "tests/fixtures/ranking_v2/scale50_snapshot.json").read_text(encoding="utf-8")
    )
    assert payload["snapshot_version"] == "scale-50-ranking-v2-smoke-v1"
    matching_books = [
        MatchingBookProfile(
            book_id=row["book_id"],
            topic_distribution={row["topic_id"]: 1.0},
            covered_concepts=[
                MatchingConcept(concept=concept, weight=1.0) for concept in row["covered_concepts"]
            ],
            prerequisite_concepts=[],
            feature_version="book-v1",
            config_version="features-v1",
            config_hash=HASH,
        )
        for row in payload["books"]
    ]
    assert len(matching_books) == 50
    books = build_prerequisite_first_book_profiles(matching_books, PROJECTION)
    expected = {
        "operating-systems": {
            "counts": (11, 1, 13),
            "top": [
                "isbn13:9780132199087",
                "isbn13:9780070575721",
                "isbn13:9780130319999",
                "isbn13:9780471725954",
                "isbn13:9781292025773",
            ],
        },
        "linear-algebra": {
            "counts": (10, 0, 15),
            "top": [
                "book_09c91e7b456682eb19fb",
                "isbn13:9780131860612",
                "isbn13:9780135367971",
                "isbn13:9780205080106",
                "isbn13:9780070380233",
            ],
        },
    }
    for topic, expectation in expected.items():
        reader = MatchingReaderProfile(
            topic_id=topic,
            vocabulary=0.5,
            background_knowledge=0.5,
            comprehension=0.5,
            concept_readiness=[
                MatchingConceptReadiness(concept_id=concept, score=score)
                for concept, score in sorted(payload["readers"][topic].items())
            ],
            profile_version="reader-v1",
            config_version="reader-config-v1",
            config_hash=HASH,
        )
        first = rank_prerequisite_first_v2(reader, books, RANKING_V2, limit=5)
        reversed_input = rank_prerequisite_first_v2(
            reader, list(reversed(books)), RANKING_V2, limit=5
        )

        diagnostics = first.diagnostics
        assert diagnostics.topic_candidate_count == 25
        assert (
            diagnostics.personalizable_count,
            diagnostics.concept_only_count,
            diagnostics.evidence_unavailable_count,
        ) == expectation["counts"]
        assert [item.book_id for item in first.items] == expectation["top"]
        assert first.items == reversed_input.items
        assert all(item.prerequisite_assessed_count >= 1 for item in first.items)
        assert all(0 < item.prerequisite_coverage <= 1 for item in first.items)
        assert all(0 <= item.direct_coverage <= 1 for item in first.items)
        assert all(len(item.reasons) >= 2 for item in first.items)


def _reader(scores: dict[str, float]) -> MatchingReaderProfile:
    """Build a strict matching reader with only explicitly assessed concepts."""

    return MatchingReaderProfile(
        topic_id="operating-systems",
        vocabulary=0.5,
        background_knowledge=0.5,
        comprehension=0.5,
        concept_readiness=[
            MatchingConceptReadiness(concept_id=concept, score=score)
            for concept, score in sorted(scores.items())
        ],
        profile_version="reader-fixture-v1",
        config_version="reader-fixture-config-v1",
        config_hash=HASH,
    )


def _book(
    book_id: str,
    covered: list[str],
    prerequisites: list[str],
) -> PrerequisiteFirstBookProfile:
    """Build a production-v2 book input with sorted concept sets."""

    return PrerequisiteFirstBookProfile(
        book_id=book_id,
        topic_id="operating-systems",
        covered_concepts=sorted(covered),
        inferred_prerequisites=sorted(prerequisites),
        feature_version="source-aware-v1",
        config_version="source-aware-config-v1",
        config_hash=HASH,
    )


def _mapping_book(
    book_id: str,
    topic: str,
    concepts: list[str],
) -> BookEvidenceConceptMapping:
    """Build one source-aware mapping row for candidate/production parity."""

    rows = [
        BookConceptPresence(
            topic_id=topic,
            concept_id=concept,
            supporting_evidence=[],
            support_count=1,
        )
        for concept in concepts
    ]
    return BookEvidenceConceptMapping(
        book_id=book_id,
        title=f"Title {book_id}",
        topics=[topic],
        matcher_version="normalized_alias_span_v2",
        model_version="concept-matching-v2-overlap",
        matching_config_version="concept-matching-config-v3",
        matching_config_hash=HASH,
        evidence_row_count=max(1, len(rows)),
        matched_evidence_row_count=len(rows),
        ambiguous_evidence_row_count=0,
        concept_presence_count=len(rows),
        raw_match_occurrence_count=len(rows),
        concepts=rows,
    )


def _parity_inputs() -> tuple[BookEvidenceConceptMappingReport, list[MatchingBookProfile]]:
    """Build equivalent source-aware candidate-v1 and production-v2 inputs."""

    definitions = [
        ("os-personal", "operating-systems", ["deadlock", "virtual memory"]),
        ("os-concept", "operating-systems", ["programming"]),
        ("os-missing", "operating-systems", []),
        ("la-personal", "linear-algebra", ["basis", "least squares"]),
        ("la-concept", "linear-algebra", ["high school algebra"]),
        ("la-missing", "linear-algebra", []),
    ]
    mapped = [_mapping_book(book_id, topic, concepts) for book_id, topic, concepts in definitions]
    mapping = BookEvidenceConceptMappingReport(
        report_version="book-evidence-concept-presence-v1",
        book_evidence_contract_version="book-evidence-v1",
        book_evidence_hash=HASH,
        matcher_version="normalized_alias_span_v2",
        model_version="concept-matching-v2-overlap",
        matching_config_version="concept-matching-config-v3",
        matching_config_hash=HASH,
        total_books=len(mapped),
        books_with_matched_concepts=4,
        books_without_matched_concepts=2,
        unique_book_concept_presence_count=6,
        raw_match_occurrence_count=6,
        books=mapped,
    )
    matching_books = [
        MatchingBookProfile(
            book_id=book_id,
            topic_distribution={topic: 1.0},
            covered_concepts=[MatchingConcept(concept=concept, weight=1.0) for concept in concepts],
            prerequisite_concepts=[],
            lexical_difficulty=None,
            syntactic_complexity=None,
            concept_density=None,
            prerequisite_demand=None,
            feature_version="source-aware-v1",
            config_version="source-aware-config-v1",
            config_hash=HASH,
        )
        for book_id, topic, concepts in definitions
    ]
    return mapping, matching_books


def test_exact_ordering_uses_opportunity_only_for_readiness_ties_then_book_id() -> None:
    """Higher opportunity cannot overtake unequal prerequisite readiness."""

    reader = _reader(
        {
            "high": 0.9,
            "tie": 0.8,
            "known-low": 0.2,
            "known-high": 0.7,
        }
    )
    books = [
        _book("lower-readiness", ["known-low"], ["tie"]),
        _book("z-tie", ["known-low"], ["high"]),
        _book("a-tie", ["known-low"], ["high"]),
        _book("same-readiness-lower-opportunity", ["known-high"], ["high"]),
    ]

    response = rank_prerequisite_first_v2(reader, books, RANKING_V2, limit=10)

    assert [item.book_id for item in response.items] == [
        "a-tie",
        "z-tie",
        "same-readiness-lower-opportunity",
        "lower-readiness",
    ]
    assert [item.rank for item in response.items] == [1, 2, 3, 4]
    assert all(not hasattr(item, "score") for item in response.items)


def test_pool_filtering_shortage_and_unknown_readiness_are_explicit() -> None:
    """Fallback books remain unranked and do not fill a requested limit."""

    reader = _reader({"ready": 0.75, "covered": 0.3})
    books = [
        _book("personalizable", ["covered"], ["ready", "unknown"]),
        _book("concept-only", ["covered"], ["unassessed"]),
        _book("no-evidence", [], []),
    ]

    response = rank_prerequisite_first_v2(reader, books, RANKING_V2, limit=5)

    assert [item.book_id for item in response.items] == ["personalizable"]
    item = response.items[0]
    assert item.prerequisite_readiness == 0.75
    assert item.prerequisite_assessed_count == 1
    assert item.prerequisite_total_count == 2
    assert item.prerequisite_coverage == 0.5
    assert item.direct_learning_opportunity == 0.7
    assert response.diagnostics.personalizable_count == 1
    assert response.diagnostics.concept_only_count == 1
    assert response.diagnostics.evidence_unavailable_count == 1
    assert response.diagnostics.returned_count == 1
    assert response.diagnostics.personalized_candidate_shortage == 4


def test_no_personalizable_candidates_returns_empty_diagnostic_response() -> None:
    """An empty personalized pool is successful and never receives fallback fill."""

    reader = _reader({"covered": 0.4})
    books = [
        _book("concept-only", ["covered"], []),
        _book("no-evidence", [], []),
    ]

    response = rank_prerequisite_first_v2(reader, books, RANKING_V2, limit=5)

    assert response.items == []
    assert response.diagnostics.returned_count == 0
    assert response.diagnostics.fallback_count == 2
    assert response.diagnostics.personalized_candidate_shortage == 5


@pytest.mark.parametrize("target", ["concept-only", "no-evidence"])
def test_target_book_without_personalized_evidence_is_an_explicit_error(target: str) -> None:
    """Target scoring never returns a fake personalized value for fallback books."""

    reader = _reader({"covered": 0.4})
    books = [
        _book("concept-only", ["covered"], []),
        _book("no-evidence", [], []),
    ]

    with pytest.raises(RankingV2TargetUnavailableError, match="not personalizable"):
        rank_prerequisite_first_v2(
            reader,
            books,
            RANKING_V2,
            limit=5,
            book_id=target,
        )


def test_input_order_does_not_change_rank_v2_response() -> None:
    """Book input order cannot alter ranks or diagnostics."""

    reader = _reader({"p1": 0.9, "p2": 0.8, "covered": 0.2})
    books = [
        _book("first", ["covered"], ["p1"]),
        _book("second", ["covered"], ["p2"]),
    ]

    assert rank_prerequisite_first_v2(
        reader, books, RANKING_V2, limit=5
    ) == rank_prerequisite_first_v2(reader, list(reversed(books)), RANKING_V2, limit=5)


def test_reader_profile_projects_directly_into_production_rank_v2() -> None:
    """The real ReaderProfile pipeline supplies MatchingReaderProfile to v2."""

    full = build_reader_profile(
        load_assessment(ROOT / "examples/assessment.json"),
        READER,
    )
    reader = to_matching_reader_profile(full)
    assessed = {item.concept_id for item in reader.concept_readiness}
    prerequisite = next(iter(sorted(assessed)))
    book = _book("real-reader-book", [prerequisite], [prerequisite])

    response = rank_prerequisite_first_v2(reader, [book], RANKING_V2, limit=5)

    assert response.reader_profile_version == full.profile_version
    assert response.reader_config_hash == full.config_hash
    assert response.diagnostics.returned_count == 1
    assert response.items[0].prerequisite_coverage == 1


def test_target_personalizable_returns_one_item_without_scalar_score() -> None:
    """A supported target returns one diagnostic result with its global personalized rank."""

    reader = _reader({"ready": 0.8, "ahead-ready": 0.9, "covered": 0.3})
    books = [
        _book("target", ["covered"], ["ready"]),
        _book("ahead", ["covered"], ["ahead-ready"]),
    ]

    response = rank_prerequisite_first_v2(reader, books, RANKING_V2, limit=5, book_id="target")

    assert len(response.items) == 1
    assert response.items[0].rank == 2
    assert response.diagnostics.requested_limit == 1
    assert response.diagnostics.personalized_candidate_shortage == 0


def test_production_v2_matches_candidate_v1_across_all_eight_scenarios() -> None:
    """Production ordering and axes remain identical to the reviewed candidate."""

    mapping, matching_books = _parity_inputs()
    production_books = build_prerequisite_first_book_profiles(matching_books, PROJECTION)
    profiles = build_scenario_profiles(EXPERIMENT, PROJECTION)

    for profile in profiles:
        candidate = evaluate_candidate_scenario(
            mapping,
            profile,
            GRAPH,
            REVIEWS,
            MATCHING,
            mapping_report_hash=HASH,
            input_hashes={"fixture": HASH},
        )
        production = rank_prerequisite_first_v2(
            to_matching_reader_profile(profile.reader_profile),
            production_books,
            RANKING_V2,
            limit=25,
        )
        candidate_items = [book for book in candidate.books if book.rank is not None]

        assert [item.book_id for item in production.items] == [
            item.book_id for item in candidate_items
        ]
        assert [item.prerequisite_readiness for item in production.items] == [
            item.prerequisite_readiness for item in candidate_items
        ]
        assert [item.direct_learning_opportunity for item in production.items] == [
            item.direct_learning_opportunity for item in candidate_items
        ]
