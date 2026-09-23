"""Regression checks for the production overlap-only matcher-v2."""

from pathlib import Path
from types import SimpleNamespace

import pytest

from bookmatch_ml.concept_v2.evidence_evaluation import (
    EvidenceConceptGoldReview,
    calculate_evidence_metrics,
)
from bookmatch_ml.concept_v2.graph import load_concept_graph
from bookmatch_ml.concept_v2.holdout import (
    EvidenceConceptHoldoutManifest,
    EvidenceConceptHoldoutPredictions,
    EvidenceConceptHoldoutReview,
)
from bookmatch_ml.concept_v2.profile import (
    load_concept_matching_config,
    match_concept_text,
    match_concept_text_production,
)
from bookmatch_ml.config import load_feature_config

ROOT = Path(__file__).resolve().parents[1]
FEATURES = load_feature_config(ROOT / "configs/features.yaml")
GRAPH = load_concept_graph(ROOT / "configs/concept_graph.yaml", FEATURES)
MATCHING_V1 = load_concept_matching_config(ROOT / "configs/concept_matching.yaml")
MATCHING_V2 = load_concept_matching_config(ROOT / "configs/concept_matching_v2.yaml")


def _production_ids(text: str, topic: str) -> list[str]:
    matches, ambiguous = match_concept_text_production(
        text,
        topic,
        FEATURES,
        GRAPH,
        MATCHING_V2,
    )
    assert not ambiguous
    return sorted(match.concept_id for match in matches)


def test_production_v2_suppresses_only_nested_overlap() -> None:
    assert _production_ids("Vector spaces", "linear-algebra") == ["vector space"]
    assert _production_ids("Vectors and vector spaces", "linear-algebra") == [
        "vector",
        "vector space",
    ]


def test_production_v2_retains_representative_v1_matches() -> None:
    assert _production_ids("Matrices and Eigenvalues", "linear-algebra") == [
        "eigenvalue",
        "matrix",
    ]
    assert _production_ids(
        "Processes, scheduling, file systems, and memory management",
        "operating-systems",
    ) == ["file system", "memory management", "process", "scheduling"]


def test_v1_remains_directly_callable_and_uses_its_original_semantics() -> None:
    matches, ambiguous = match_concept_text(
        "Vector spaces",
        "linear-algebra",
        FEATURES,
        GRAPH,
        MATCHING_V1,
    )
    assert not ambiguous
    assert {match.concept_id for match in matches} == {"vector", "vector space"}
    assert MATCHING_V1.config.matcher_version == "normalized_alias_phrase_v1"
    assert MATCHING_V2.config.matcher_version == "normalized_alias_span_v2"


def _metric_rows(review_rows, predictions_by_id: dict[str, list[str]]):
    return [
        SimpleNamespace(
            book_id=row.book_id,
            predicted_concept_ids=predictions_by_id[row.evidence_id],
            human_gold_concept_ids=row.human_gold_concept_ids,
        )
        for row in review_rows
        if row.review_outcome != "not_judgable"
    ]


def test_production_v2_reproduces_original_94_v2_a_metrics() -> None:
    review = EvidenceConceptGoldReview.model_validate_json(
        (ROOT / "reviews/evidence_concept_gold_review_v1.json").read_bytes()
    )
    predictions = {
        row.evidence_id: _production_ids(row.evidence_text, row.topic) for row in review.entries
    }
    metric = calculate_evidence_metrics(_metric_rows(review.entries, predictions), "all")

    assert (
        metric.true_positive_count,
        metric.false_positive_count,
        metric.false_negative_count,
    ) == (51, 1, 13)
    assert metric.micro_precision == 51 / 52
    assert metric.micro_recall == 51 / 64
    assert metric.micro_f1 == pytest.approx(0.8793103448)


def _assert_holdout_matches_fixed_v2_a(kind: str, expected: tuple[int, int, int]) -> None:
    base = ROOT / f"reviews/evidence_concept_holdout_{kind}_v1"
    manifest = EvidenceConceptHoldoutManifest.model_validate_json(
        base.with_name(f"{base.name}_manifest.json").read_bytes()
    )
    fixed = EvidenceConceptHoldoutPredictions.model_validate_json(
        base.with_name(f"{base.name}_predictions.json").read_bytes()
    )
    review = EvidenceConceptHoldoutReview.model_validate_json(
        base.with_suffix(".json").read_bytes()
    )
    fixed_by_id = {
        row.evidence_id: next(
            prediction.predicted_concept_ids
            for prediction in row.predictions
            if prediction.variant == "v2_a_overlap"
        )
        for row in fixed.entries
    }
    production_by_id = {
        row.evidence_id: _production_ids(row.evidence_text, row.topic) for row in manifest.entries
    }

    assert production_by_id == fixed_by_id
    metric = calculate_evidence_metrics(_metric_rows(review.entries, production_by_id), "all")
    assert (
        metric.true_positive_count,
        metric.false_positive_count,
        metric.false_negative_count,
    ) == expected
    expected_f1 = {
        "general": 0.864,
        "challenge": 0.7906976744,
    }[kind]
    assert metric.micro_f1 == pytest.approx(expected_f1)


def test_production_v2_reproduces_general_89_v2_a() -> None:
    _assert_holdout_matches_fixed_v2_a("general", (54, 0, 17))


def test_production_v2_reproduces_challenge_40_v2_a() -> None:
    _assert_holdout_matches_fixed_v2_a("challenge", (34, 1, 17))
