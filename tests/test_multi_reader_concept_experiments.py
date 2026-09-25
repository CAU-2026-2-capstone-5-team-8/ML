"""Regression tests for deterministic multi-reader concept diagnostics."""

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
    DiagnosticSnapshot,
    HumanBookPair,
    build_accepted_graph_projection,
    evaluate_concept_readiness_ranking,
)
from bookmatch_ml.ranking.multi_reader_concept_experiments import (
    MultiReaderConceptExperimentError,
    build_scenario_profiles,
    build_variant_orderings,
    evaluate_frozen_human_pairs,
    load_multi_reader_concept_experiment_config,
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


def _presence(topic: str, concept: str, support_count: int) -> BookConceptPresence:
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
    support_count: int,
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
            "os-evidence",
            "OS Evidence",
            "operating-systems",
            ["virtual memory", "deadlock"],
            support_count,
        ),
        _book("os-missing", "OS Missing", "operating-systems", [], support_count),
        _book(
            "la-evidence",
            "LA Evidence",
            "linear-algebra",
            ["basis", "least squares"],
            support_count,
        ),
        _book("la-missing", "LA Missing", "linear-algebra", [], support_count),
    ]
    if reverse:
        books = [
            book.model_copy(update={"concepts": list(reversed(book.concepts))})
            for book in reversed(books)
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
        books_with_matched_concepts=2,
        books_without_matched_concepts=2,
        unique_book_concept_presence_count=4,
        raw_match_occurrence_count=sum(book.raw_match_occurrence_count for book in books),
        books=books,
    )


def _scenario_report(profile, mapping=None):
    return evaluate_concept_readiness_ranking(
        mapping or _mapping(),
        [profile.reader_profile],
        GRAPH,
        REVIEWS,
        MATCHING,
        mapping_report_hash=HASH,
        reader_profile_hashes={profile.reader_profile.topic_id: profile.reader_profile_hash},
        input_hashes={"fixture": HASH},
    )


def _snapshot(
    book_id: str,
    *,
    status: str,
    direct: float,
    prerequisite: float,
    d2: float,
) -> DiagnosticSnapshot:
    return DiagnosticSnapshot(
        book_id=book_id,
        title=book_id,
        covered_concepts=[],
        inferred_prerequisites=[],
        assessed_covered_mastery={},
        assessed_prerequisite_mastery={},
        direct_coverage=1.0,
        direct_learning_opportunity=direct,
        prerequisite_coverage=1.0,
        prerequisite_readiness=prerequisite,
        two_stage_status=status,
        unweighted_combination=d2,
    )


def test_scenario_profiles_are_deterministic_and_use_exact_canonical_ids() -> None:
    first = build_scenario_profiles(EXPERIMENT, PROJECTION)
    second = build_scenario_profiles(EXPERIMENT, PROJECTION)

    assert first == second
    assert len(first) == 8
    assert [profile.scenario_id for profile in first[:4]] == [
        "beginner",
        "intermediate",
        "advanced",
        "uneven",
    ]
    for profile in first:
        topic = profile.reader_profile.topic_id
        assert {item.concept_id for item in profile.reader_profile.concept_readiness} == set(
            PROJECTION.nodes[topic]
        )
        assert profile.description


def test_projection_contains_only_the_eighteen_accepted_edges() -> None:
    review_status = {
        (row.topic, row.prerequisite, row.dependent): row.review_status
        for row in REVIEWS.config.reviews
    }

    assert len(PROJECTION.edges) == 18
    assert all(
        review_status[(edge.topic, edge.prerequisite, edge.dependent)] == "accepted"
        for edge in PROJECTION.edges
    )
    assert not {(edge.topic, edge.prerequisite, edge.dependent) for edge in PROJECTION.edges} & {
        key for key, status in review_status.items() if status in {"rejected", "needs_revision"}
    }


def test_scenario_profile_rejects_missing_canonical_concept() -> None:
    config = EXPERIMENT.config
    first = config.scenarios["linear-algebra"][0]
    invalid = first.model_copy(
        update={
            "readiness": {key: value for key, value in first.readiness.items() if key != "basis"}
        }
    )
    scenarios = dict(config.scenarios)
    scenarios["linear-algebra"] = [invalid, *scenarios["linear-algebra"][1:]]
    loaded = EXPERIMENT.model_copy(
        update={"config": config.model_copy(update={"scenarios": scenarios})}
    )

    with pytest.raises(MultiReaderConceptExperimentError, match=r"missing=\['basis'\]"):
        build_scenario_profiles(loaded, PROJECTION)


def test_profile_changes_diagnostics_and_unknown_is_not_zero() -> None:
    profiles = build_scenario_profiles(EXPERIMENT, PROJECTION)
    la_beginner = next(
        profile
        for profile in profiles
        if profile.reader_profile.topic_id == "linear-algebra" and profile.scenario_id == "beginner"
    )
    la_advanced = next(
        profile
        for profile in profiles
        if profile.reader_profile.topic_id == "linear-algebra" and profile.scenario_id == "advanced"
    )
    beginner = _scenario_report(la_beginner)
    advanced = _scenario_report(la_advanced)
    beginner_book = next(book for book in beginner.books if book.book_id == "la-evidence")
    advanced_book = next(book for book in advanced.books if book.book_id == "la-evidence")
    missing = next(book for book in beginner.books if book.book_id == "la-missing")

    assert beginner_book.direct.learning_opportunity > advanced_book.direct.learning_opportunity
    assert beginner_book.direct.assessment_coverage == 1.0
    assert missing.direct.learning_opportunity is None
    assert missing.direct.semantics == "not_applicable"


def test_ordering_preserves_unavailable_group_and_is_input_order_independent() -> None:
    profile = next(
        profile
        for profile in build_scenario_profiles(EXPERIMENT, PROJECTION)
        if profile.reader_profile.topic_id == "operating-systems"
        and profile.scenario_id == "beginner"
    )
    report = _scenario_report(profile)
    first = build_variant_orderings(report.books, 0.05)
    second = build_variant_orderings(list(reversed(report.books)), 0.05)

    assert first == second
    direct = next(variant for variant in first if variant.variant_id == "direct_opportunity")
    assert direct.statistics.available_value_count == 1
    assert direct.statistics.unavailable_value_count == 1
    assert direct.groups[-1].available is False
    assert direct.groups[-1].group_key == "unavailable"


def test_occurrence_count_is_not_weighted_and_prerequisite_projections_stay_separate() -> None:
    profile = next(
        profile
        for profile in build_scenario_profiles(EXPERIMENT, PROJECTION)
        if profile.reader_profile.topic_id == "operating-systems"
        and profile.scenario_id == "intermediate"
    )
    first = _scenario_report(profile, _mapping(support_count=1))
    second = _scenario_report(profile, _mapping(reverse=True, support_count=99))
    first_books = sorted(first.books, key=lambda book: book.book_id)
    second_books = sorted(second.books, key=lambda book: book.book_id)
    evidence = next(book for book in first_books if book.book_id == "os-evidence")

    assert first_books == second_books
    assert first.occurrence_count_used_as_weight is False
    assert set(evidence.direct_prerequisites) < set(evidence.transitive_prerequisites)
    assert evidence.direct.assessment_coverage == 1.0
    assert evidence.transitive_prerequisite_readiness.assessment_coverage == 1.0


def test_frozen_human_predictions_and_agreement_are_unchanged() -> None:
    values = [
        (("eligible", 0.5, 0.95, 0.725), ("eligible", 0.4166667, 0.7333333, 0.575)),
        (("eligible", 0.5, 0.95, 0.725), ("eligible", 0.3571429, 0.9, 0.6285714)),
        (("eligible", 0.3571429, 0.9, 0.6285714), ("eligible", 0.455, 0.7428571, 0.5989286)),
        (("eligible", 0.3772727, 0.88, 0.6286364), ("eligible", 0.455, 0.7428571, 0.5989286)),
        (("eligible", 0.4166667, 0.7333333, 0.575), ("eligible", 0.3571429, 0.9, 0.6285714)),
        (("eligible", 0.1, 0.9, 0.5), ("insufficient_evidence", 0.24, 0.575, 0.4075)),
        (("insufficient_evidence", 0.3428571, 0.62, 0.4814286), ("eligible", 0.1, 0.9, 0.5)),
        (("insufficient_evidence", 0.1, 0.8666667, 0.4833333), ("eligible", 0.18, 0.6, 0.39)),
        (
            ("insufficient_evidence", 0.1, 0.8666667, 0.4833333),
            ("challenge_candidate", 0.24, 0.575, 0.4075),
        ),
        (("eligible", 0.18, 0.6, 0.39), ("insufficient_evidence", 0.2333333, 0.62, 0.4266667)),
    ]
    pairs = []
    for frozen, (left_values, right_values) in zip(
        EXPERIMENT.config.frozen_human_pairs, values, strict=True
    ):
        pairs.append(
            HumanBookPair(
                pair_id=frozen.pair_id,
                topic_id=frozen.topic_id,
                selection_reason="frozen fixture",
                left=_snapshot(
                    frozen.left_book_id,
                    status=left_values[0],
                    direct=left_values[1],
                    prerequisite=left_values[2],
                    d2=left_values[3],
                ),
                right=_snapshot(
                    frozen.right_book_id,
                    status=right_values[0],
                    direct=right_values[1],
                    prerequisite=right_values[2],
                    d2=right_values[3],
                ),
            )
        )
    results, summaries = evaluate_frozen_human_pairs(
        SimpleNamespace(human_pair_review_packet=pairs), EXPERIMENT.config
    )
    combined = {
        summary.variant_id: summary.agreement_count
        for summary in summaries
        if summary.topic_id is None
    }

    assert len(results) == 10
    assert combined == {
        "d1": 3,
        "d2": 9,
        "direct_only": 4,
        "e_prerequisite_first": 9,
        "prerequisite_only": 9,
    }


def test_cli_rejects_output_alias_before_loading_inputs(tmp_path: Path) -> None:
    mapping = tmp_path / "mapping.json"
    reader = tmp_path / "reader.json"
    mapping.write_text("{}", encoding="utf-8")
    reader.write_text("{}", encoding="utf-8")

    result = CliRunner().invoke(
        app,
        [
            "evaluate-multi-reader-concept-ranking",
            "--concept-mapping",
            str(mapping),
            "--fixed-reader",
            str(reader),
            "--output",
            str(mapping),
        ],
    )

    assert result.exit_code == 1
    assert "--output must not overwrite an experiment input" in result.output
