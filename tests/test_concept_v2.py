"""Focused contract tests for independent TOC concept matching."""

from pathlib import Path

import pytest
from pydantic import ValidationError
from typer.testing import CliRunner

from bookmatch_ml.book.profile import build_book_profiles
from bookmatch_ml.cli import app
from bookmatch_ml.concept_v2.graph import ConceptGraph, LoadedConceptGraph, load_concept_graph
from bookmatch_ml.concept_v2.matching import match_book_concepts, order_matching_items
from bookmatch_ml.concept_v2.profile import (
    _map_entry,
    build_book_concept_profile_v2,
    load_concept_matching_config,
    validate_toc_mapping_rules,
)
from bookmatch_ml.concept_v2.toc import TocTreeError, reconstruct_toc
from bookmatch_ml.config import ConfigError, load_feature_config
from bookmatch_ml.data.evidence import assemble_book_evidence, calculate_evidence_coverage
from bookmatch_ml.data.loader import load_canonical_dataset
from bookmatch_ml.io import write_jsonl
from bookmatch_ml.ranking.loader import load_reader_profile
from bookmatch_ml.schemas import Book, BookEvidence, ConceptReadiness, TocEntry

ROOT = Path(__file__).resolve().parents[1]
FEATURES = load_feature_config(ROOT / "configs/features.yaml")
MATCHING = load_concept_matching_config(ROOT / "configs/concept_matching.yaml")
REAL_GRAPH = load_concept_graph(ROOT / "configs/concept_graph.yaml", FEATURES)
BOOK_ID = "book_11111111111111111111"
HASH = "sha256:" + "a" * 64


def entry(identifier: str, title: str, parent: str | None, level: int, order: int) -> TocEntry:
    return TocEntry(
        toc_entry_id=identifier,
        book_id=BOOK_ID,
        parent_entry_id=parent,
        level=level,
        order_index=order,
        label=None,
        title=title,
        source_id="synthetic",
    )


def graph(edges: list[tuple[str, str]], nodes: list[str] | None = None) -> LoadedConceptGraph:
    payload = {
        "config_version": "test-config-v1",
        "graph_version": "test-graph-v1",
        "relation_type": "prerequisite_candidate",
        "nodes": {
            "operating-systems": nodes or ["process", "thread", "concurrency", "synchronization"]
        },
        "edges": [
            {
                "topic": "operating-systems",
                "prerequisite": before,
                "dependent": after,
                "source_type": "proposed_seed",
                "version": "seed-v1",
            }
            for before, after in edges
        ],
    }
    return LoadedConceptGraph(graph=ConceptGraph.model_validate(payload), content_hash=HASH)


def evidence(entries: list[TocEntry]) -> BookEvidence:
    book = Book(
        book_id=BOOK_ID,
        isbn_10=None,
        isbn_13=None,
        title="Synthetic OS",
        subtitle=None,
        authors=[],
        publisher=None,
        published_year=None,
        language="en",
        topics=["operating-systems"],
    )
    return BookEvidence(
        book_id=BOOK_ID,
        metadata=book,
        toc=entries,
        documents=[],
        coverage=calculate_evidence_coverage([], entries),
    )


def profile(
    entries: list[TocEntry],
    loaded_graph: LoadedConceptGraph,
    topic: str = "operating-systems",
):
    book_evidence = evidence(entries)
    if topic != "operating-systems":
        book_evidence = book_evidence.model_copy(
            update={"metadata": book_evidence.metadata.model_copy(update={"topics": [topic]})}
        )
    compatible_matching = MATCHING
    if loaded_graph.graph.graph_version == "test-graph-v1":
        empty_rules = MATCHING.config.toc_mapping.model_copy(
            update={"alias_additions": {}, "exclusions": {}}
        )
        compatible_matching = MATCHING.model_copy(
            update={"config": MATCHING.config.model_copy(update={"toc_mapping": empty_rules})}
        )
    return build_book_concept_profile_v2(
        book_evidence, topic, FEATURES, loaded_graph, compatible_matching, HASH
    )


def reader(scores: dict[str, float]):
    source = load_reader_profile(ROOT / "examples/reader_profile.json")
    payload = source.model_dump(mode="python")
    payload["concept_readiness"] = [
        ConceptReadiness(
            concept_id=concept,
            score=score,
            response_count=1,
            earned_weight=score,
            available_weight=1,
        )
        for concept, score in sorted(scores.items())
    ]
    return type(source).model_validate(payload)


@pytest.mark.parametrize(
    ("edges", "nodes", "message"),
    [
        ([("process", "missing")], None, "unknown node"),
        ([("process", "process")], None, "self-loop"),
        ([("process", "thread"), ("process", "thread")], None, "duplicate"),
        ([("process", "thread"), ("thread", "process")], None, "cycle"),
    ],
)
def test_graph_rejects_invalid_edges(edges, nodes, message):
    with pytest.raises(ValidationError, match=message):
        graph(edges, nodes)


def test_graph_rejects_mixed_topic_edge():
    with pytest.raises(ValidationError, match="mixes topics"):
        ConceptGraph.model_validate(
            {
                "config_version": "x",
                "graph_version": "x",
                "relation_type": "prerequisite_candidate",
                "nodes": {"operating-systems": ["process"], "linear-algebra": ["matrix"]},
                "edges": [
                    {
                        "topic": "operating-systems",
                        "prerequisite": "process",
                        "dependent": "matrix",
                        "source_type": "proposed_seed",
                        "version": "x",
                    }
                ],
            }
        )


def test_graph_load_is_deterministic():
    first = load_concept_graph(ROOT / "configs/concept_graph.yaml", FEATURES)
    second = load_concept_graph(ROOT / "configs/concept_graph.yaml", FEATURES)
    assert first == second
    assert first.content_hash.startswith("sha256:")
    assert len(first.graph.edges) > 0


def test_missing_prerequisite_topic_is_a_config_error():
    prerequisite = FEATURES.config.prerequisite.model_copy(
        update={"topics": {"linear-algebra": FEATURES.config.prerequisite.topics["linear-algebra"]}}
    )
    features = FEATURES.model_copy(
        update={"config": FEATURES.config.model_copy(update={"prerequisite": prerequisite})}
    )
    with pytest.raises(ConfigError, match="feature prerequisite aliases missing topic"):
        load_concept_graph(ROOT / "configs/concept_graph.yaml", features)


def test_concept_cli_accepts_reordered_valid_v1_profiles(tmp_path: Path):
    fixture_dir = ROOT / "tests/fixtures/canonical"
    dataset = load_canonical_dataset(fixture_dir)
    profiles = build_book_profiles(assemble_book_evidence(dataset), FEATURES)
    books = tmp_path / "reversed_profiles.jsonl"
    write_jsonl(list(reversed(profiles)), books)
    result = CliRunner().invoke(
        app,
        [
            "evaluate-concept-matching",
            "--data-dir",
            str(fixture_dir),
            "--reader",
            str(ROOT / "examples/reader_profile.json"),
            "--books",
            str(books),
            "--output",
            str(tmp_path / "report.json"),
        ],
    )
    assert result.exit_code == 0, result.output


def test_toc_hierarchy_and_sibling_order_independent_of_input_order():
    entries = [
        entry("b", "Thread", "root", 2, 1),
        entry("grandchild", "Synchronization", "a", 3, 0),
        entry("root", "Process", None, 1, 0),
        entry("a", "Concurrency", "root", 2, 0),
    ]
    tree = reconstruct_toc(entries)
    assert [visit.entry.toc_entry_id for visit in tree.traversal] == [
        "root",
        "a",
        "grandchild",
        "b",
    ]
    assert tree.roots[0].children[0].children[0].entry.toc_entry_id == "grandchild"
    assert tree.traversal[2].parent_path == ("Process", "Concurrency")
    assert reconstruct_toc(list(reversed(entries))).traversal == tree.traversal


@pytest.mark.parametrize(
    ("entries", "message"),
    [
        ([entry("orphan", "Process", "missing", 2, 0)], "orphan"),
        ([entry("root", "Process", None, 2, 0)], "level 1"),
        ([entry("root", "Process", None, 1, 0), entry("child", "Thread", "root", 3, 0)], "level"),
        (
            [entry("a", "Process", None, 1, 0), entry("b", "Thread", None, 1, 0)],
            "duplicate sibling",
        ),
    ],
)
def test_toc_rejects_invalid_hierarchy(entries, message):
    with pytest.raises(TocTreeError, match=message):
        reconstruct_toc(entries)


def test_mapping_occurrences_unmatched_and_paths_are_stable():
    entries = [
        entry("root", "Processes", None, 1, 0),
        entry("child", "CPU Scheduling", "root", 2, 0),
        entry("repeat", "Processes", None, 1, 1),
        entry("unknown", "Appendix", None, 1, 2),
    ]
    loaded = graph([], nodes=["process", "scheduling"])
    result = profile(entries, loaded)
    covered = {item.concept_id: item for item in result.covered_concepts}
    assert covered["process"].occurrence_count == 2
    assert covered["process"].coverage_weight == pytest.approx(2 / 3)
    assert covered["process"].first_traversal_position == 0
    assert result.toc_mappings[1].toc_path == ["Processes", "CPU Scheduling"]
    assert result.diagnostics.matched_toc_entries == 3
    assert result.unmapped_entries[0].reason == "unmatched"
    assert result.model_dump() == profile(list(reversed(entries)), loaded).model_dump()
    assert result.graph_hash == HASH
    assert result.toc_file_hash == HASH


def test_ambiguous_alias_is_preserved_without_guessing():
    visit = reconstruct_toc([entry("p", "Processes", None, 1, 0)]).traversal[0]
    mappings, ambiguous, excluded = _map_entry(
        visit, {"process": ["processes"], "thread": ["processes"]}
    )
    assert mappings == []
    assert ambiguous is True
    assert excluded == set()


def test_toc_mapping_rules_remove_only_clear_false_positives():
    result = profile(
        [
            entry("compilation", "The compilation process", None, 1, 0),
            entry("process", "Process Management", None, 1, 1),
            entry("disk", "Disk Scheduling", None, 1, 2),
            entry("cpu", "CPU Scheduling", None, 1, 3),
        ],
        REAL_GRAPH,
    )
    mapped = {(row.toc_entry_id, row.concept_id) for row in result.toc_mappings}
    assert ("compilation", "process") not in mapped
    assert ("process", "process") in mapped
    assert ("disk", "scheduling") not in mapped
    assert ("disk", "storage") in mapped
    assert ("cpu", "scheduling") in mapped


def test_plural_systems_of_linear_equations_maps_to_linear_system():
    result = profile(
        [entry("systems", "Systems of Linear Equations", None, 1, 0)],
        REAL_GRAPH,
        topic="linear-algebra",
    )
    assert [(row.concept_id, row.matching_alias) for row in result.toc_mappings] == [
        ("linear system", "systems of linear equations")
    ]


@pytest.mark.parametrize("rule_name", ["alias_additions", "exclusions"])
@pytest.mark.parametrize("reference", ["unknown_topic", "unknown_concept"])
def test_toc_mapping_rules_reject_unknown_topics_and_concepts(rule_name, reference):
    rules = MATCHING.config.toc_mapping
    update = {topic: dict(concepts) for topic, concepts in getattr(rules, rule_name).items()}
    if reference == "unknown_topic":
        update["unknown-topic"] = {"process": ["process"]}
        message = "unknown topics"
    else:
        update.setdefault("operating-systems", {})["unknown concept"] = ["unknown"]
        message = "unknown operating-systems concepts"
    invalid_rules = rules.model_copy(update={rule_name: update})
    invalid_matching = MATCHING.model_copy(
        update={"config": MATCHING.config.model_copy(update={"toc_mapping": invalid_rules})}
    )
    with pytest.raises(ConfigError, match=message):
        validate_toc_mapping_rules(invalid_matching, REAL_GRAPH)


def test_prerequisite_before_use_missing_and_multiple_paths():
    loaded = graph(
        [
            ("process", "thread"),
            ("process", "concurrency"),
            ("thread", "concurrency"),
            ("concurrency", "synchronization"),
        ]
    )
    result = profile(
        [
            entry("p", "Processes", None, 1, 0),
            entry("t", "Threads", None, 1, 1),
            entry("s", "Synchronization", None, 1, 2),
        ],
        loaded,
    )
    taught = {item.concept_id: item for item in result.taught_before_use_candidates}
    external = {item.concept_id: item for item in result.prerequisite_requirements}
    assert "process" in taught
    assert "concurrency" in external
    assert len(taught["process"].graph_paths) >= 2
    assert "synchronization" not in external
    assert (
        result.model_dump()
        == profile(
            list(
                reversed(
                    [
                        entry("p", "Processes", None, 1, 0),
                        entry("t", "Threads", None, 1, 1),
                        entry("s", "Synchronization", None, 1, 2),
                    ]
                )
            ),
            loaded,
        ).model_dump()
    )


def test_missing_dependent_produces_no_requirement():
    result = profile([entry("p", "Processes", None, 1, 0)], graph([("process", "thread")]))
    assert result.prerequisite_requirements == []
    assert result.taught_before_use_candidates == []


def test_readiness_opportunity_and_missing_mastery_are_separate():
    result = profile(
        [
            entry("p", "Processes", None, 1, 0),
            entry("t", "Threads", None, 1, 1),
            entry("s", "Synchronization", None, 1, 2),
        ],
        graph([("process", "thread"), ("concurrency", "synchronization")]),
    )
    item = match_book_concepts(reader({"process": 0.8, "thread": 0.2}), result, MATCHING)
    assert item.readiness.score is None
    assert item.readiness.assessment_coverage == 0
    assert item.readiness.semantics == "unassessed"
    assert item.opportunity.score == pytest.approx(0.5)
    assert item.opportunity.assessment_coverage == pytest.approx(2 / 3)
    assert item.status == "insufficient_evidence"
    assert item.unknown_prerequisites == ["concurrency"]
    assert item.unknown_mastery_concepts == ["synchronization"]
    assert item.learning_candidate_concepts == ["thread"]
    assert all(
        0 <= value <= 1 for value in [item.opportunity.score, item.opportunity.assessment_coverage]
    )


def test_zero_mastery_is_assessed_and_no_prerequisite_is_not_applicable():
    result = profile([entry("p", "Processes", None, 1, 0)], graph([]))
    item = match_book_concepts(reader({"process": 0}), result, MATCHING)
    assert item.readiness.semantics == "not_applicable"
    assert item.readiness.score is None
    assert item.readiness.assessment_coverage is None
    assert item.opportunity.score == 1
    assert item.opportunity.assessment_coverage == 1
    assert item.status == "eligible"


def test_no_mapped_concept_has_unavailable_opportunity():
    result = profile([entry("x", "Appendix", None, 1, 0)], graph([]))
    item = match_book_concepts(reader({}), result, MATCHING)
    assert item.opportunity.semantics == "not_applicable"
    assert item.opportunity.score is None
    assert item.opportunity.assessment_coverage is None
    assert item.status == "insufficient_evidence"


def test_assessed_prerequisite_formula_and_stable_order():
    loaded = graph([("process", "synchronization"), ("thread", "synchronization")])
    result = profile(
        [
            entry("p", "Processes", None, 1, 0),
            entry("t", "Threads", None, 1, 1),
            entry("s", "Synchronization", None, 1, 2),
        ],
        loaded,
    )
    # Both are taught before use, so the readiness denominator is correctly absent.
    taught = {item.concept_id for item in result.taught_before_use_candidates}
    assert taught == {"process", "thread"}
    late = profile(
        [
            entry("s", "Synchronization", None, 1, 0),
            entry("p", "Processes", None, 1, 1),
            entry("t", "Threads", None, 1, 2),
        ],
        loaded,
    )
    item = match_book_concepts(
        reader({"process": 1.0, "thread": 0.0, "synchronization": 0.2}), late, MATCHING
    )
    assert item.readiness.score == pytest.approx(0.5)
    assert item.readiness.assessment_coverage == 1
    assert item.opportunity.score == pytest.approx((0 + 1 + 0.8) / 3)
    assert item.status == "challenge_candidate"
    partial = match_book_concepts(reader({"process": 1.0, "synchronization": 0.2}), late, MATCHING)
    assert partial.readiness.score == 1.0
    assert partial.readiness.assessment_coverage == pytest.approx(0.5)
    other = item.model_copy(update={"book_id": "book_22222222222222222222"})
    assert [value.book_id for value in order_matching_items([other, item])] == [
        item.book_id,
        other.book_id,
    ]
