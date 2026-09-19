"""Structural validation artifacts remain deterministic and separate from ranking."""

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from bookmatch_ml.cli import app
from bookmatch_ml.concept_v2.graph import load_concept_graph
from bookmatch_ml.concept_v2.matching import match_book_concepts
from bookmatch_ml.concept_v2.profile import (
    build_book_concept_profile_v2,
    load_concept_matching_config,
)
from bookmatch_ml.concept_v2.validation import (
    ConceptGraphReviewFile,
    HumanEdgeReview,
    LoadedConceptGraphReviews,
    build_concept_validation_report,
    build_review_template,
    load_concept_graph_reviews,
)
from bookmatch_ml.config import (
    ConfigError,
    LoadedFeatureConfig,
    load_feature_config,
    load_reader_config,
)
from bookmatch_ml.data.evidence import assemble_book_evidence, calculate_evidence_coverage
from bookmatch_ml.data.loader import load_canonical_dataset
from bookmatch_ml.reader.profile import build_reader_profile, load_assessment

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "tests/fixtures/canonical"
FEATURES = load_feature_config(ROOT / "configs/features.yaml")
GRAPH = load_concept_graph(ROOT / "configs/concept_graph.yaml", FEATURES)
MATCHING = load_concept_matching_config(ROOT / "configs/concept_matching.yaml")
HASH = "sha256:" + "a" * 64


def _fixture_reader():
    return build_reader_profile(
        load_assessment(ROOT / "examples/assessment.json"),
        load_reader_config(ROOT / "configs/reader.yaml"),
    )


def _report(
    features: LoadedFeatureConfig = FEATURES,
    reviews: LoadedConceptGraphReviews | None = None,
):
    evidence = assemble_book_evidence(load_canonical_dataset(FIXTURE))
    return build_concept_validation_report(
        evidence=evidence,
        graph=GRAPH,
        features=features,
        matching=MATCHING,
        readers={"operating-systems": _fixture_reader()},
        toc_file_hash=HASH,
        input_hashes={"toc.jsonl": HASH},
        reviews=reviews,
    )


def _review_payload() -> dict:
    return {
        "config_version": "concept-graph-reviews-v1",
        "graph_version": GRAPH.graph.graph_version,
        "reviews": [
            {
                "topic": edge.topic,
                "prerequisite": edge.prerequisite,
                "dependent": edge.dependent,
                "review_status": "unreviewed",
                "review_note": None,
            }
            for edge in GRAPH.graph.edges
        ],
    }


def test_validation_report_is_semantically_stable_and_keeps_every_toc_entry():
    first = _report()
    second = _report()
    assert first.model_dump_json() == second.model_dump_json()
    assert sum(book.total_toc_entries for book in first.books) == 2
    assert all(
        book.matched_toc_entries + book.unmatched_toc_entries + book.ambiguous_toc_entries
        == book.total_toc_entries
        for book in first.books
    )
    os_book = next(book for book in first.books if book.topic == "operating-systems")
    assert os_book.matched_entries[0].toc_path == ["Processes"]
    assert os_book.matched_entries[0].matching_alias == "processes"
    assert os_book.matched_entries[0].traversal_position == 0
    assert os_book.unmatched_entries == []


def test_validation_cli_is_byte_stable(tmp_path: Path):
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"
    review = tmp_path / "review.json"
    common = ["build-concept-review", "--data-dir", str(FIXTURE)]
    first_result = CliRunner().invoke(
        app, [*common, "--output", str(first), "--review-template", str(review)]
    )
    first_review = review.read_bytes()
    second_result = CliRunner().invoke(
        app, [*common, "--output", str(second), "--review-template", str(review)]
    )
    assert first_result.exit_code == 0, first_result.output
    assert second_result.exit_code == 0, second_result.output
    assert first.read_bytes() == second.read_bytes()
    assert first_review == review.read_bytes()


def test_validation_cli_allows_missing_human_review_config(tmp_path: Path):
    output = tmp_path / "report.json"
    review = tmp_path / "review.json"
    result = CliRunner().invoke(
        app,
        [
            "build-concept-review",
            "--data-dir",
            str(FIXTURE),
            "--output",
            str(output),
            "--review-template",
            str(review),
            "--review-config",
            str(tmp_path / "missing.yaml"),
        ],
    )
    assert result.exit_code == 0, result.output
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["review_config_hash"] is None
    assert {row["review_status"] for row in report["graph_edges"]} == {"unreviewed"}


def test_validation_cli_overwrites_generated_snapshot_not_review_source(tmp_path: Path):
    output = tmp_path / "report.json"
    review = tmp_path / "review.json"
    review.write_text("{}", encoding="utf-8")
    result = CliRunner().invoke(
        app,
        [
            "build-concept-review",
            "--data-dir",
            str(FIXTURE),
            "--output",
            str(output),
            "--review-template",
            str(review),
        ],
    )
    assert result.exit_code == 0, result.output
    generated = ConceptGraphReviewFile.model_validate_json(review.read_text(encoding="utf-8"))
    assert len(generated.edges) == len(GRAPH.graph.edges)


def test_edge_evidence_uses_first_occurrence_path_and_observed_state():
    report = _report()
    edge = next(
        edge
        for edge in report.graph_edges
        if edge.topic == "operating-systems"
        and edge.prerequisite == "process"
        and edge.dependent == "scheduling"
    )
    book = next(item for item in edge.books if item.book_id == "book_11111111111111111111")
    assert book.state == "strict_before"
    assert book.legacy_state == "observed_before_in_toc"
    assert book.prerequisite_first.toc_title == "Processes"
    assert book.prerequisite_first.toc_path == ["Processes"]
    assert book.prerequisite_first.traversal_position == 0
    assert book.dependent_first.toc_path == ["Processes", "Scheduling"]
    assert book.dependent_first.traversal_position == 1


def test_same_entry_is_separate_from_strict_before():
    source = assemble_book_evidence(load_canonical_dataset(FIXTURE))[0]
    combined = source.toc[0].model_copy(update={"title": "Processes and Scheduling"})
    one_entry = source.model_copy(
        update={
            "toc": [combined],
            "coverage": calculate_evidence_coverage(source.documents, [combined]),
        }
    )
    report = build_concept_validation_report(
        evidence=[one_entry],
        graph=GRAPH,
        features=FEATURES,
        matching=MATCHING,
        readers={},
        toc_file_hash=HASH,
        input_hashes={"toc.jsonl": HASH},
    )
    edge = next(
        row
        for row in report.graph_edges
        if row.topic == "operating-systems"
        and row.prerequisite == "process"
        and row.dependent == "scheduling"
    )
    assert edge.books[0].state == "same_entry"
    assert edge.same_entry_count == 1
    assert edge.strict_before_count == 0


def test_artifact_preserves_unmatched_and_ambiguous_entries_separately():
    topics = dict(FEATURES.config.concept.topics)
    aliases = dict(topics["operating-systems"].root)
    aliases["thread"] = ["processes"]
    aliases["scheduling"] = ["cpu dispatch"]
    topics["operating-systems"] = topics["operating-systems"].model_copy(update={"root": aliases})
    concept = FEATURES.config.concept.model_copy(update={"topics": topics})
    features = FEATURES.model_copy(
        update={"config": FEATURES.config.model_copy(update={"concept": concept})}
    )
    report = _report(features)
    os_book = next(book for book in report.books if book.topic == "operating-systems")
    reasons = {item.toc_title: item.reason for item in os_book.unmatched_entries}
    assert reasons["Processes"] == "ambiguous_alias"
    assert reasons["Scheduling"] == "unmatched"
    assert os_book.ambiguous_toc_entries == 1
    assert os_book.unmatched_toc_entries == 1


def test_review_template_does_not_promote_source_type_or_change_v2_scores():
    evidence = assemble_book_evidence(load_canonical_dataset(FIXTURE))[0]
    profile = build_book_concept_profile_v2(
        evidence, "operating-systems", FEATURES, GRAPH, MATCHING, HASH
    )
    before = match_book_concepts(_fixture_reader(), profile, MATCHING)
    template = build_review_template(GRAPH)
    after = match_book_concepts(_fixture_reader(), profile, MATCHING)
    assert before == after
    assert {edge.review_status for edge in template.edges} == {"unreviewed"}
    assert {edge.source_type for edge in GRAPH.graph.edges} == {"proposed_seed"}
    assert all(edge.review_note is None for edge in template.edges)


def test_reviewed_status_requires_note_and_unreviewed_rejects_note():
    with pytest.raises(ValueError, match="require a non-blank review note"):
        HumanEdgeReview(
            topic="operating-systems",
            prerequisite="process",
            dependent="thread",
            review_status="accepted",
        )
    with pytest.raises(ValueError, match="must not carry"):
        HumanEdgeReview(
            topic="operating-systems",
            prerequisite="process",
            dependent="thread",
            review_status="unreviewed",
            review_note="automatic",
        )


def test_review_file_rejects_duplicate_edges():
    edge = build_review_template(GRAPH).edges[0]
    with pytest.raises(ValueError, match="duplicate"):
        ConceptGraphReviewFile(
            review_schema_version="concept-graph-review-v2",
            graph_version=GRAPH.graph.graph_version,
            graph_hash=GRAPH.content_hash,
            review_config_version=None,
            review_config_hash=None,
            edges=[edge, edge],
        )


def test_human_review_statuses_are_loaded_into_both_artifacts(tmp_path: Path):
    payload = _review_payload()
    statuses = ["unreviewed", "accepted", "rejected", "needs_revision"]
    expected = {}
    for row, status in zip(payload["reviews"][:4], statuses, strict=True):
        row["review_status"] = status
        row["review_note"] = None if status == "unreviewed" else f"human decision: {status}"
        expected[(row["topic"], row["prerequisite"], row["dependent"])] = status
    path = tmp_path / "reviews.yaml"
    path.write_text(json.dumps(payload), encoding="utf-8")
    reviews = load_concept_graph_reviews(path, GRAPH)
    report = _report(reviews=reviews)
    snapshot = build_review_template(GRAPH, reviews)
    assert {
        (row.topic, row.prerequisite, row.dependent): row.review_status
        for row in report.graph_edges
        if (row.topic, row.prerequisite, row.dependent) in expected
    } == expected
    assert {
        (row.topic, row.prerequisite, row.dependent): row.review_status
        for row in snapshot.edges
        if (row.topic, row.prerequisite, row.dependent) in expected
    } == expected


@pytest.mark.parametrize(
    ("failure", "message"),
    [
        ("invalid_status", "Input should be"),
        ("unknown_edge", "unknown graph edges"),
    ],
)
def test_human_review_config_rejects_invalid_status_and_unknown_edge(
    tmp_path: Path, failure: str, message: str
):
    payload = _review_payload()
    if failure == "invalid_status":
        payload["reviews"][0]["review_status"] = "uncertain"
    else:
        payload["reviews"][0]["dependent"] = "missing concept"
    path = tmp_path / "reviews.yaml"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ConfigError, match=message):
        load_concept_graph_reviews(path, GRAPH)


def test_human_review_config_wraps_malformed_yaml(tmp_path: Path):
    path = tmp_path / "reviews.yaml"
    path.write_text(
        "config_version: concept-graph-reviews-v1\nconfig_version: duplicate\n",
        encoding="utf-8",
    )
    with pytest.raises(ConfigError, match="invalid concept graph review config"):
        load_concept_graph_reviews(path, GRAPH)


def test_opportunity_descriptors_are_not_a_new_score():
    os_book = next(book for book in _report().books if book.topic == "operating-systems")
    assert os_book.learning_candidate_count is not None
    assert os_book.known_concept_count is not None
    assert os_book.unknown_mastery_count is not None
    assert os_book.measured_learning_weight is not None
    assert os_book.total_book_concept_weight > 0
    assert "score" not in os_book.model_fields_set
