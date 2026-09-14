from pathlib import Path

import pytest

from bookmatch_ml.book.concepts import build_concept_profile
from bookmatch_ml.book.difficulty import build_difficulty_profile
from bookmatch_ml.book.profile import build_book_profile, build_book_profiles
from bookmatch_ml.config import load_feature_config
from bookmatch_ml.data.evidence import assemble_book_evidence, calculate_evidence_coverage
from bookmatch_ml.data.loader import load_canonical_dataset
from bookmatch_ml.schemas import Document

ROOT = Path(__file__).parents[1]
FIXTURE_DIR = Path(__file__).parent / "fixtures" / "canonical"
CONFIG = load_feature_config(ROOT / "configs" / "features.yaml")


def _evidence():
    return assemble_book_evidence(load_canonical_dataset(FIXTURE_DIR))


def test_concept_profile_is_deterministic_and_retains_evidence_references() -> None:
    evidence = _evidence()[0]

    first = build_concept_profile(evidence, CONFIG)
    second = build_concept_profile(evidence, CONFIG)

    assert first == second
    assert first.topic_distribution == {"operating-systems": 1.0}
    process = next(item for item in first.covered_concepts if item.concept == "process")
    assert "toc" in process.evidence_types
    assert "preface" in process.evidence_types
    assert {reference.evidence_id for reference in process.evidence} >= {
        "toc_root",
        "doc_preface",
    }
    assert first.analyzed_toc_entry_ids == ["toc_root", "toc_child"]
    assert first.analyzed_document_ids == ["doc_description", "doc_preface"]
    assert first.profile_version == "concept-v1"
    assert first.config_hash == CONFIG.content_hash


def test_prerequisite_inference_is_labeled_as_a_proxy() -> None:
    profile = build_concept_profile(_evidence()[0], CONFIG)

    architecture = next(
        item for item in profile.prerequisite_concepts if item.concept == "computer architecture"
    )
    assert architecture.method == "explicit_and_early_prose_proxy"
    assert architecture.evidence[0].evidence_id == "doc_preface"


def test_difficulty_uses_only_valid_prose_and_retains_raw_features() -> None:
    profile = build_difficulty_profile(_evidence()[0], CONFIG)

    assert profile.analyzed_document_count == 1
    assert profile.documents[0].document_type == "preface"
    assert profile.documents[0].raw.mean_sentence_tokens > 0
    assert profile.documents[0].raw.vocabulary_diversity > 0
    assert profile.documents[0].raw.concept_mention_count > 0
    assert profile.analyzed_character_count == len(_evidence()[0].documents[1].text)
    for score in (
        profile.lexical_difficulty,
        profile.syntactic_complexity,
        profile.concept_density,
        profile.prerequisite_demand,
    ):
        assert score is not None
        assert 0 <= score <= 1


def test_book_without_prose_gets_null_difficulty_not_zero() -> None:
    profile = build_difficulty_profile(_evidence()[1], CONFIG)

    assert profile.analyzed_document_count == 0
    assert profile.lexical_difficulty is None
    assert profile.syntactic_complexity is None
    assert profile.concept_density is None
    assert profile.prerequisite_demand is None


def test_short_prose_is_excluded_with_a_visible_reason() -> None:
    evidence = _evidence()[0]
    short_document = evidence.documents[1].model_copy(update={"text": "Only a few words."})
    documents = [evidence.documents[0], short_document]
    evidence = evidence.model_copy(
        update={
            "documents": documents,
            "coverage": calculate_evidence_coverage(documents, evidence.toc),
        }
    )

    profile = build_difficulty_profile(evidence, CONFIG)

    assert profile.lexical_difficulty is None
    assert profile.excluded_documents[0].reason == "below_minimum_tokens"


def test_multiple_prose_documents_use_token_weighted_aggregation() -> None:
    evidence = _evidence()[0]
    added = Document(
        document_id="doc_sample",
        book_id=evidence.book_id,
        document_type="sample_chapter",
        text=("Processes and threads coordinate useful work in memory. " * 15),
        source_id="source_111",
        content_hash="sha256:" + "e" * 64,
    )
    documents = [*evidence.documents, added]
    evidence = evidence.model_copy(
        update={
            "documents": documents,
            "coverage": calculate_evidence_coverage(documents, evidence.toc),
        }
    )

    profile = build_difficulty_profile(evidence, CONFIG)
    expected = sum(
        document.scores.lexical_difficulty * document.token_count for document in profile.documents
    ) / sum(document.token_count for document in profile.documents)

    assert profile.analyzed_document_count == 2
    assert profile.lexical_difficulty == pytest.approx(expected)
    assert profile.aggregation_rule == "token_weighted_mean_v1"


def test_book_profile_keeps_coverage_and_all_versions() -> None:
    profile = build_book_profile(_evidence()[0], CONFIG)

    assert profile.evidence_coverage.has_toc is True
    assert profile.feature_version == "book-v1"
    assert profile.concept_profile.profile_version == "concept-v1"
    assert profile.difficulty_profile.feature_version == "difficulty-v1"
    assert profile.config_version == "features-v1"


def test_profile_list_is_sorted_by_book_id() -> None:
    evidence = list(reversed(_evidence()))

    profiles = build_book_profiles(evidence, CONFIG)

    assert [profile.book_id for profile in profiles] == sorted(
        profile.book_id for profile in profiles
    )
