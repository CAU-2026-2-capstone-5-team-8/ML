"""Focused regression tests for matcher-v2 ablation variants."""

from pathlib import Path

from bookmatch_ml.concept_v2.graph import load_concept_graph
from bookmatch_ml.concept_v2.matcher_experiments import (
    load_matcher_v2_experiment_config,
    match_concept_text_v2_a,
    match_concept_text_v2_b,
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
