"""Focused regression tests for matcher-v2 ablation variants."""

import hashlib
from pathlib import Path

from bookmatch_ml.concept_v2.evidence_evaluation import (
    EvidenceConceptGoldReview,
    LoadedEvidenceConceptGoldReview,
)
from bookmatch_ml.concept_v2.graph import load_concept_graph
from bookmatch_ml.concept_v2.matcher_experiment_evaluation import (
    build_matcher_v2_experiment_report,
)
from bookmatch_ml.concept_v2.matcher_experiments import (
    load_matcher_v2_experiment_config,
    match_concept_text_v2_a,
    match_concept_text_v2_b,
    match_concept_toc_v2_c1,
    match_concept_toc_v2_c2,
)
from bookmatch_ml.concept_v2.profile import load_concept_matching_config
from bookmatch_ml.config import load_feature_config

ROOT = Path(__file__).resolve().parents[1]
FEATURES = load_feature_config(ROOT / "configs/features.yaml")
GRAPH = load_concept_graph(ROOT / "configs/concept_graph.yaml", FEATURES)
MATCHING = load_concept_matching_config(ROOT / "configs/concept_matching.yaml")
EXPERIMENT = load_matcher_v2_experiment_config(ROOT / "configs/matcher_v2_experiments.yaml")


def _ids(text: str) -> set[str]:
    matches, ambiguous = match_concept_text_v2_a(
        text,
        "linear-algebra",
        FEATURES,
        GRAPH,
        MATCHING,
    )
    assert not ambiguous
    return {match.concept_id for match in matches}


def test_overlap_suppresses_vector_nested_inside_vector_space() -> None:
    assert _ids("Vector spaces") == {"vector space"}


def test_overlap_keeps_vector_at_an_independent_text_position() -> None:
    assert _ids("Vectors are used alongside vector spaces") == {"vector", "vector space"}


def test_overlap_keeps_distinct_concepts_supported_by_nested_phrases() -> None:
    assert _v2_b_ids("Multithreaded programming", "operating-systems") == {
        "programming",
        "thread",
    }


def _v2_b_ids(text: str, topic: str, evidence_type: str = "toc_exact") -> set[str]:
    matches, ambiguous = match_concept_text_v2_b(
        text,
        topic,
        evidence_type,
        FEATURES,
        GRAPH,
        MATCHING,
        EXPERIMENT,
    )
    assert not ambiguous
    return {match.concept_id for match in matches}


def test_reviewed_alias_and_form_variants_are_matched() -> None:
    assert _v2_b_ids("Linear maps", "linear-algebra") == {"linear transformation"}
    assert _v2_b_ids("Multithreaded", "operating-systems") == {"thread"}
    assert _v2_b_ids("Programming exercises", "operating-systems") == {"programming"}
    assert _v2_b_ids("Distributed UNIX systems", "operating-systems") == {"distributed systems"}
    assert _v2_b_ids("Multimedia Storage", "operating-systems") == {"storage"}


def test_bounded_modifier_patterns_recover_rephrased_linear_systems() -> None:
    assert _v2_b_ids(
        "Systems of Two Linear Equations in Two Variables",
        "linear-algebra",
    ) == {"linear system", "systems of equations"}


def test_virtual_machine_exercise_environment_is_not_course_content() -> None:
    text = (
        "A Linux virtual machine including C and Java source code and development tools "
        "allows students to complete programming exercises."
    )
    assert _v2_b_ids(text, "operating-systems", "description") == {"programming"}


def test_virtualization_learning_context_retains_virtual_machine_match() -> None:
    text = "This chapter explains virtualization, hypervisors, and virtual machines."
    assert _v2_b_ids(text, "operating-systems", "description") == {"virtualization"}


def _path_ids(
    text: str,
    path: list[str],
    topic: str,
    naive: bool = False,
) -> set[str]:
    matcher = match_concept_toc_v2_c1 if naive else match_concept_toc_v2_c2
    matches, ambiguous = matcher(
        text,
        path,
        topic,
        "toc_exact",
        FEATURES,
        GRAPH,
        MATCHING,
        EXPERIMENT,
    )
    assert not ambiguous
    return {match.concept_id for match in matches}


def test_context_assisted_path_recovers_reviewed_parent_concepts() -> None:
    assert _path_ids(
        "Permutations and Cofactors",
        ["Determinants", "Permutations and Cofactors"],
        "linear-algebra",
    ) == {"determinant"}
    assert _path_ids(
        "Causality",
        ["Distributed Algorithms", "Models of Distributed Computation", "Causality"],
        "operating-systems",
    ) == {"distributed systems"}
    assert _path_ids(
        "Main Memory",
        ["MEMORY MANAGEMENT", "Main Memory"],
        "operating-systems",
    ) == {"memory management"}
    assert _path_ids(
        "Input Software",
        ["Input/Output", "User Interfaces", "Input Software"],
        "operating-systems",
    ) == {"input/output"}
    assert _path_ids(
        "Hardware Issues",
        ["Input/Output", "Power Management", "Hardware Issues"],
        "operating-systems",
    ) == {"input/output"}


def test_context_assisted_path_does_not_inherit_when_leaf_matches() -> None:
    assert _path_ids(
        "The Rank and the Row Reduced Form",
        ["Vector Spaces and Subspaces", "The Rank and the Row Reduced Form"],
        "linear-algebra",
    ) == {"rank"}
    assert _path_ids(
        "Processes",
        ["Processes and Threads", "Processes"],
        "operating-systems",
    ) == {"process"}


def test_naive_path_union_exposes_parent_over_inheritance() -> None:
    assert _path_ids(
        "The Rank and the Row Reduced Form",
        ["Vector Spaces and Subspaces", "The Rank and the Row Reduced Form"],
        "linear-algebra",
        naive=True,
    ) == {"rank", "vector space"}
    assert _path_ids(
        "Processes",
        ["Processes and Threads", "Processes"],
        "operating-systems",
        naive=True,
    ) == {"process", "thread"}


def test_context_assisted_path_does_not_inherit_for_generic_leaf() -> None:
    assert (
        _path_ids(
            "Exercises",
            ["Vector Spaces", "Exercises"],
            "linear-algebra",
        )
        == set()
    )


def test_frozen_gold_ablation_is_stable_and_read_only() -> None:
    path = ROOT / "reviews/evidence_concept_gold_review_v1.json"
    before = path.read_bytes()
    review = EvidenceConceptGoldReview.model_validate_json(before)
    loaded = LoadedEvidenceConceptGoldReview(
        review=review,
        content_hash=f"sha256:{hashlib.sha256(before).hexdigest()}",
    )

    report = build_matcher_v2_experiment_report(
        loaded,
        FEATURES,
        GRAPH,
        MATCHING,
        EXPERIMENT,
    )

    totals = {
        variant.variant: (
            variant.metrics[0].true_positive_count,
            variant.metrics[0].false_positive_count,
            variant.metrics[0].false_negative_count,
        )
        for variant in report.variants
    }
    assert totals == {
        "v1": (51, 4, 13),
        "v2_a_overlap": (51, 1, 13),
        "v2_b_forms": (59, 0, 5),
        "v2_c1_naive_path_union": (64, 4, 0),
        "v2_c2_context_assisted": (64, 0, 0),
    }
    assert path.read_bytes() == before
