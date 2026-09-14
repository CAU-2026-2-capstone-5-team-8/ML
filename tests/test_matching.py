from pathlib import Path

import pytest

from bookmatch_ml.book.profile import build_book_profile, build_book_profiles
from bookmatch_ml.config import (
    load_feature_config,
    load_ranking_config,
    load_reader_config,
)
from bookmatch_ml.data.evidence import assemble_book_evidence, calculate_evidence_coverage
from bookmatch_ml.data.loader import load_canonical_dataset
from bookmatch_ml.ranking.matching import RankingError, rank_books, score_book_fit
from bookmatch_ml.reader.profile import build_reader_profile, load_assessment

ROOT = Path(__file__).parents[1]
FIXTURE_DIR = Path(__file__).parent / "fixtures" / "canonical"
FEATURE_CONFIG = load_feature_config(ROOT / "configs" / "features.yaml")
RANKING_CONFIG = load_ranking_config(ROOT / "configs" / "ranking.yaml")
READER_CONFIG = load_reader_config(ROOT / "configs" / "reader.yaml")


def _reader():
    assessment = load_assessment(ROOT / "examples" / "assessment.json")
    return build_reader_profile(assessment, READER_CONFIG)


def _evidence():
    dataset = load_canonical_dataset(FIXTURE_DIR)
    return assemble_book_evidence(dataset)


def _profiles():
    return build_book_profiles(_evidence(), FEATURE_CONFIG)


def _operating_systems_without_prose():
    evidence = _evidence()[0]
    documents = [
        document for document in evidence.documents if document.document_type == "description"
    ]
    evidence = evidence.model_copy(
        update={
            "documents": documents,
            "coverage": calculate_evidence_coverage(documents, evidence.toc),
        }
    )
    return build_book_profile(evidence, FEATURE_CONFIG)


def test_score_decomposition_matches_configured_formula() -> None:
    item = score_book_fit(_reader(), _profiles()[0], RANKING_CONFIG)
    weights = RANKING_CONFIG.config.component_weights
    expected = sum(getattr(item.components, name) * weight for name, weight in weights.items())

    assert item.score == pytest.approx(expected)
    assert item.active_weights == weights
    assert item.component_weight_coverage == 1.0
    assert item.knowledge_active_weights == RANKING_CONFIG.config.knowledge_weights
    assert item.knowledge_weight_coverage == 1.0
    assert item.unavailable_components == []
    assert item.components.topic_fit == 1.0
    assert item.knowledge_components.concept_density_fit is not None
    assert item.knowledge_components.prerequisite_demand_fit is not None
    assert item.knowledge_components.prerequisite_concept_fit is not None
    assert item.model_version == "rank-v1"
    assert item.config_version == "ranking-config-v1"


def test_missing_components_are_renormalized_and_not_treated_as_zero() -> None:
    item = score_book_fit(_reader(), _operating_systems_without_prose(), RANKING_CONFIG)

    assert item.score == 1.0
    assert item.components.topic_fit == 1.0
    assert item.components.vocabulary_fit is None
    assert item.components.knowledge_fit is None
    assert item.components.comprehension_fit is None
    assert item.active_weights == {"topic_fit": 1.0}
    assert item.component_weight_coverage == pytest.approx(0.35)
    assert item.knowledge_active_weights == {}
    assert item.knowledge_weight_coverage == 0.0
    assert item.unavailable_components == [
        "vocabulary_fit",
        "knowledge_fit",
        "comprehension_fit",
    ]
    assert any("실제 본문" in reason for reason in item.reasons)


def test_specific_cross_topic_book_retains_zero_topic_fit() -> None:
    linear_algebra = _profiles()[1]

    item = score_book_fit(_reader(), linear_algebra, RANKING_CONFIG)

    assert item.score == 0.0
    assert item.components.topic_fit == 0.0
    assert item.active_weights == {"topic_fit": 1.0}
    assert any("일치하지 않습니다" in reason for reason in item.reasons)


def test_top_k_filters_by_topic_and_is_deterministic() -> None:
    os_profile = _operating_systems_without_prose()
    second_os_profile = os_profile.model_copy(update={"book_id": "book_00000000000000000000"})
    books = [*_profiles(), second_os_profile]

    first = rank_books(_reader(), books, RANKING_CONFIG, limit=10)
    second = rank_books(_reader(), list(reversed(books)), RANKING_CONFIG, limit=10)

    assert first == second
    assert [item.book_id for item in first.items] == [
        "book_00000000000000000000",
        "book_11111111111111111111",
    ]
    assert all(item.components.topic_fit == 1.0 for item in first.items)
    assert first.model_version == "rank-v1"


def test_top_k_limit_is_applied_after_scoring() -> None:
    response = rank_books(_reader(), _profiles(), RANKING_CONFIG, limit=1)

    assert len(response.items) == 1
    assert response.items[0].book_id == "book_11111111111111111111"


def test_invalid_limit_fails_clearly() -> None:
    with pytest.raises(RankingError, match="at least 1"):
        rank_books(_reader(), _profiles(), RANKING_CONFIG, limit=0)


def test_explanations_follow_calculated_gaps_and_evidence() -> None:
    reader = _reader()
    item = score_book_fit(reader, _profiles()[0], RANKING_CONFIG)

    assert any("주제와 관련성이 높습니다" in reason for reason in item.reasons)
    assert item.diagnostics.lexical_demand is not None
    assert (
        abs(reader.vocabulary - item.diagnostics.lexical_demand)
        <= RANKING_CONFIG.config.explanation.close_gap
    )
    assert "어휘 난이도는 현재 준비도와 대체로 맞습니다." in item.reasons
    assert any("개념·선행지식 요구도" in reason for reason in item.reasons)
    assert any("문장 복잡도" in reason for reason in item.reasons)
    assert any("추정 선행 개념" in reason for reason in item.reasons)
    assert any("주요 개념" in reason for reason in item.reasons)
