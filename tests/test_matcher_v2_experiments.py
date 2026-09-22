"""Focused regression tests for matcher-v2 ablation variants."""

from pathlib import Path

from bookmatch_ml.concept_v2.graph import load_concept_graph
from bookmatch_ml.concept_v2.matcher_experiments import match_concept_text_v2_a
from bookmatch_ml.concept_v2.profile import load_concept_matching_config
from bookmatch_ml.config import load_feature_config

ROOT = Path(__file__).resolve().parents[1]
FEATURES = load_feature_config(ROOT / "configs/features.yaml")
GRAPH = load_concept_graph(ROOT / "configs/concept_graph.yaml", FEATURES)
MATCHING = load_concept_matching_config(ROOT / "configs/concept_matching.yaml")


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
