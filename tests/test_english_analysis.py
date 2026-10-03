from pathlib import Path

import pytest

from bookmatch_ml.book.concepts import build_concept_profile
from bookmatch_ml.book.difficulty import build_difficulty_profile
from bookmatch_ml.book.english import analysis_text
from bookmatch_ml.config import load_feature_config
from bookmatch_ml.data.evidence import assemble_book_evidence
from bookmatch_ml.data.loader import load_canonical_dataset

ROOT = Path(__file__).parents[1]
CONFIG = load_feature_config(ROOT / "configs/features.yaml")


def evidence():
    return assemble_book_evidence(load_canonical_dataset(ROOT / "tests/fixtures/canonical"))[0]


def test_missing_translation_is_not_analyzed_as_english():
    assert analysis_text("행렬(rank)", None) == ""
    assert analysis_text("Matrix rank", None) == "Matrix rank"
    assert analysis_text("행렬", "Matrices") == "Matrices"
    with pytest.raises(ValueError, match="non-English"):
        analysis_text("행렬", "행렬")


def test_concepts_use_english_toc_and_preserve_original_title():
    book = evidence()
    book.metadata.title = "원문 도서 제목"
    book.metadata.en_title = "Translated book title"
    book.documents = []
    for entry in book.toc:
        entry.title = "원문 목차"
        entry.en_title = "Processes"
    profile = build_concept_profile(book, CONFIG)
    process = next(row for row in profile.covered_concepts if row.concept == "process")
    assert process.evidence_types == ["toc"]
    assert book.metadata.title == "원문 도서 제목"
    assert all(entry.title == "원문 목차" for entry in book.toc)


def test_difficulty_analyzes_english_text_and_preserves_original_document_hash():
    book = evidence()
    prose = next(row for row in book.documents if row.document_type == "preface")
    original_english = prose.text
    original_hash = prose.content_hash
    prose.text = "원문 문장입니다."
    prose.en_text = original_english
    profile = build_difficulty_profile(book, CONFIG)
    assert profile.analyzed_document_count == 1
    assert profile.analyzed_character_count == len(original_english)
    assert prose.text == "원문 문장입니다."
    assert prose.content_hash == original_hash


def test_absent_english_fields_do_not_change_legacy_json_or_hash_payload():
    book = evidence()
    assert "en_title" not in book.metadata.model_dump()
    assert all("en_text" not in row.model_dump() for row in book.documents)
