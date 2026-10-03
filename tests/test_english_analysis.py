"""Account integration must preserve the original-text baseline and v3 boundary."""

from pathlib import Path

from bookmatch_ml.book.concepts import build_concept_profile
from bookmatch_ml.book.difficulty import build_difficulty_profile
from bookmatch_ml.cli import DEFAULT_CONCEPT_MATCHING_V2_CONFIG, DEFAULT_FEATURE_CONFIG
from bookmatch_ml.concept_v2.book_evidence_mapping import build_book_evidence_concept_mapping_report
from bookmatch_ml.concept_v2.graph import load_concept_graph
from bookmatch_ml.concept_v2.profile import load_concept_matching_config
from bookmatch_ml.config import load_feature_config
from bookmatch_ml.data.book_evidence import load_book_evidence
from bookmatch_ml.data.evidence import assemble_book_evidence
from bookmatch_ml.data.loader import load_canonical_dataset

ROOT = Path(__file__).parents[1]
CONFIG = load_feature_config(ROOT / "configs/features.yaml")


def evidence():
    return assemble_book_evidence(load_canonical_dataset(ROOT / "tests/fixtures/canonical"))[0]


def test_stored_english_does_not_change_default_concepts_or_difficulty():
    book = evidence()
    concepts = build_concept_profile(book, CONFIG)
    difficulty = build_difficulty_profile(book, CONFIG)
    for entry in book.toc:
        entry.en_title = "Unrelated translated heading"
    for document in book.documents:
        document.en_text = "Short translated text."
    assert build_concept_profile(book, CONFIG) == concepts
    assert build_difficulty_profile(book, CONFIG) == difficulty


def test_missing_english_does_not_silently_drop_mixed_original_text():
    book = evidence()
    book.documents = []
    for entry in book.toc:
        entry.title = "과정 소개: Processes"
    profile = build_concept_profile(book, CONFIG)
    assert any(row.concept == "process" for row in profile.covered_concepts)


def test_v3_nullable_english_survives_serialization_and_mapping_uses_original():
    records = load_book_evidence(ROOT / "tests/fixtures/book-evidence-v3.jsonl")
    record = records[0]
    for item in record.evidence:
        assert "en_text" in item.model_dump()
    graph = load_concept_graph(ROOT / "configs/concept_graph.yaml", CONFIG)
    matching = load_concept_matching_config(ROOT / "configs/concept_matching_v2.yaml")
    without_english = record.model_copy(
        update={"evidence": [item.model_copy(update={"en_text": None}) for item in record.evidence]}
    )
    args = ("sha256:" + "a" * 64, CONFIG, graph, matching)
    with_fields = build_book_evidence_concept_mapping_report(records, *args)
    without_fields = build_book_evidence_concept_mapping_report([without_english], *args)
    assert with_fields == without_fields


def test_cli_defaults_keep_baseline_and_absent_canonical_english_stays_omitted():
    assert Path("configs/features.yaml") == DEFAULT_FEATURE_CONFIG
    assert Path("configs/concept_matching_v2.yaml") == DEFAULT_CONCEPT_MATCHING_V2_CONFIG
    book = evidence()
    assert "en_title" not in book.metadata.model_dump()
    assert all("en_text" not in row.model_dump() for row in book.documents)
