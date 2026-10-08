"""Bounded alias candidate: real matching behavior and frozen-set regression."""

import json
from pathlib import Path

import pytest

from bookmatch_ml.concept_v2.graph import load_concept_graph
from bookmatch_ml.concept_v2.profile import (
    load_concept_matching_config,
    match_concept_text_production,
)
from bookmatch_ml.config import load_feature_config

ROOT = Path(__file__).resolve().parents[1]
FEATURES = load_feature_config(ROOT / "configs/features.yaml")
GRAPH = load_concept_graph(ROOT / "configs/concept_graph.yaml", FEATURES)
BASELINE = load_concept_matching_config(ROOT / "configs/concept_matching_v2.yaml")
CANDIDATE = load_concept_matching_config(ROOT / "configs/concept_matching_dot_product_v1.yaml")


def matches(text, config=CANDIDATE, topic="linear-algebra"):
    return match_concept_text_production(text, topic, FEATURES, GRAPH, config)


@pytest.mark.parametrize(
    "text",
    ["Dot product", "DOT PRODUCTS", "A dot-product calculation", "Dot\nproducts", "(dot product)"],
)
def test_dot_product_forms_map_to_inner_product(text):
    result, ambiguous = matches(text)
    assert not ambiguous
    assert [m.concept_id for m in result] == ["inner product"]
    assert result[0].matching_alias in {"dot product", "dot products"}
    assert matches(text, BASELINE) == ([], False)


@pytest.mark.parametrize(
    "text",
    [
        "Cross products",
        "Tensor product",
        "Product",
        "Dot production",
        "Dot productivity",
        "Dot productscope",
        "Antidot product",
        "Dotproduct",
        "Dot scalar product",
    ],
)
def test_candidate_does_not_expand_to_other_products_or_partial_words(text):
    assert matches(text) == matches(text, BASELINE) == ([], False)


def test_candidate_keeps_distinct_concepts_and_deduplicates_inner_product():
    result, ambiguous = matches("Vectors, vector spaces, inner products, and dot products")
    assert not ambiguous
    assert [m.concept_id for m in result] == ["inner product", "vector", "vector space"]
    assert matches("Dot product", topic="operating-systems") == ([], False)


@pytest.mark.parametrize(
    "filename",
    [
        "evidence_concept_gold_review_v1.json",
        "evidence_concept_holdout_general_v1_manifest.json",
        "evidence_concept_holdout_challenge_v1_manifest.json",
    ],
)
def test_candidate_preserves_every_frozen_prediction_and_ambiguity(filename):
    data = json.loads((ROOT / "reviews" / filename).read_text())
    for row in data["entries"]:
        before = matches(row["evidence_text"], BASELINE, row["topic"])
        after = matches(row["evidence_text"], CANDIDATE, row["topic"])
        assert after == before, row["evidence_id"]
