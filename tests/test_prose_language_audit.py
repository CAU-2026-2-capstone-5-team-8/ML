import hashlib
import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from bookmatch_ml.cli import app
from bookmatch_ml.config import load_feature_config
from bookmatch_ml.data.loader import load_canonical_dataset
from bookmatch_ml.evaluation.prose_language import build_prose_language_audit
from bookmatch_ml.schemas import DocumentDifficulty, ExcludedProseDocument, TextExtent

ROOT = Path(__file__).parents[1]
FIXTURE = ROOT / "tests/fixtures/canonical"
FEATURES = load_feature_config(ROOT / "configs/features.yaml")


def korean_fixture(language="ko", short=False):
    dataset = load_canonical_dataset(FIXTURE)
    doc = next(d for d in dataset.documents if d.document_type == "preface")
    text = (
        "합성 검증 자료입니다."
        if short
        else "합성 검증 자료입니다. 벡터와 행렬의 관계를 설명합니다. " * 20
    )
    docs = [
        d.model_copy(
            update={
                "text": text,
                "content_hash": "sha256:" + hashlib.sha256(text.encode()).hexdigest(),
                "text_extent": TextExtent(scope="excerpt", basis="Synthetic fixture excerpt."),
            }
        )
        if d.document_id == doc.document_id
        else d
        for d in dataset.documents
    ]
    books = [
        b.model_copy(update={"language": language}) if b.book_id == doc.book_id else b
        for b in dataset.books
    ]
    return dataset.model_copy(update={"books": books, "documents": docs})


def test_korean_numeric_scores_are_observations_not_validated_language_quality():
    dataset = korean_fixture()
    before = dataset.model_dump_json()
    report = build_prose_language_audit(dataset, FEATURES, "en")
    assert report.evaluation_status == "not_evaluated"
    assert report.total_books == 2 and report.books_without_prose == 1
    assert report.prose_document_count == 1
    row = report.documents[0]
    assert row.language_relation == "different_from_baseline"
    assert row.letter_counts["hangul"] > 0 and row.hangul_letter_share == 1
    assert row.text_extent.scope == "excerpt"
    assert isinstance(row.baseline_result, DocumentDifficulty)
    assert row.baseline_result.raw.concept_mention_count == 0
    assert "no_configured_concept_alias_matches" in row.review_flags
    assert (
        row.human_language_review is row.human_prose_review is row.human_relative_difficulty is None
    )
    assert dataset.model_dump_json() == before


def test_english_metadata_does_not_hide_hangul_or_assert_quality():
    report = build_prose_language_audit(korean_fixture("eng"), FEATURES, "en")
    row = report.documents[0]
    assert row.language_relation == "matches_declared_baseline"
    assert "hangul_present_in_english_baseline_input" in row.review_flags
    assert report.evaluation_status == "not_evaluated"


def test_unknown_language_and_short_excerpt_remain_visible_without_scores():
    row = build_prose_language_audit(korean_fixture("und", short=True), FEATURES, "en").documents[0]
    assert row.language_relation == "unknown"
    assert isinstance(row.baseline_result, ExcludedProseDocument)
    assert row.baseline_result.reason == "below_minimum_tokens"
    assert row.text_extent.scope == "excerpt"
    assert "no_configured_concept_alias_matches" not in row.review_flags


def test_no_prose_is_not_fabricated_from_descriptions():
    dataset = load_canonical_dataset(FIXTURE)
    dataset = dataset.model_copy(
        update={"documents": [d for d in dataset.documents if d.document_type == "description"]}
    )
    report = build_prose_language_audit(dataset, FEATURES, "en")
    assert report.books_without_prose == len(dataset.books)
    assert report.documents == [] and report.baseline_scored_document_count == 0


@pytest.mark.parametrize("language", ["", "english", "und", "mul", "zxx"])
def test_baseline_language_must_be_explicit(language):
    with pytest.raises(ValueError, match="baseline language"):
        build_prose_language_audit(load_canonical_dataset(FIXTURE), FEATURES, language)


def test_cli_is_deterministic_hashes_inputs_and_preserves_existing_files(tmp_path):
    outputs = [tmp_path / "first.json", tmp_path / "second.json"]
    runner = CliRunner()
    args = [
        "audit-prose-language",
        "--data-dir",
        str(FIXTURE),
        "--baseline-language",
        "en",
        "--config",
        str(ROOT / "configs/features.yaml"),
    ]
    for output in outputs:
        result = runner.invoke(app, [*args, "--output", str(output)])
        assert result.exit_code == 0, result.output
    assert outputs[0].read_bytes() == outputs[1].read_bytes()
    report = json.loads(outputs[0].read_text())
    assert set(report["canonical_hashes"]) == {
        "books.jsonl",
        "documents.jsonl",
        "toc.jsonl",
        "sources.jsonl",
    }
    original = (FIXTURE / "books.jsonl").read_bytes()
    for output in [outputs[0], FIXTURE / "books.jsonl"]:
        refused = runner.invoke(app, [*args, "--output", str(output)])
        assert refused.exit_code != 0 and "existing files are preserved" in refused.output
    assert (FIXTURE / "books.jsonl").read_bytes() == original
