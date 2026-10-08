import hashlib
import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from bookmatch_ml.cli import app
from bookmatch_ml.concept_v2.graph import load_concept_graph
from bookmatch_ml.concept_v2.profile import load_concept_matching_config
from bookmatch_ml.config import load_feature_config
from bookmatch_ml.data.book_evidence import (
    BookEvidenceImportError,
    build_concept_candidate_inputs,
    load_book_evidence,
)
from bookmatch_ml.evaluation.english_evidence import compare_english_evidence
from bookmatch_ml.schemas import Document

ROOT = Path(__file__).parents[1]
FIXTURE = ROOT / "tests/fixtures/book-evidence-v3.jsonl"
FEATURES = load_feature_config(ROOT / "configs/features.yaml")
GRAPH = load_concept_graph(ROOT / "configs/concept_graph.yaml", FEATURES)
MATCHING = load_concept_matching_config(ROOT / "configs/concept_matching_v2.yaml")


def test_v3_shared_fixture_exact_original_english_and_candidate_projection():
    rows = load_book_evidence(FIXTURE)
    candidate = next(c for c in build_concept_candidate_inputs(rows)[0].candidates if c.document_id)
    assert candidate.text.startswith(" \n") and candidate.text.endswith("\n ")
    assert candidate.en_text == " \nSynthetic vectors and matrices.\n "
    assert candidate.text_extent.scope == "excerpt"
    doc = Document(
        document_id="doc",
        book_id=rows[0].book.book_id,
        document_type="preface",
        text=candidate.text,
        en_text=candidate.en_text,
        source_id="s",
        content_hash="sha256:" + hashlib.sha256(candidate.text.encode()).hexdigest(),
    )
    assert doc.text == candidate.text and doc.en_text == candidate.en_text
    assert "en_text" not in doc.model_copy(update={"en_text": None}).model_dump()


def test_paired_comparison_never_uses_missing_english_or_changes_inputs():
    rows = load_book_evidence(FIXTURE)
    before = rows[0].model_dump_json()
    report = compare_english_evidence(rows, FEATURES, GRAPH, MATCHING)
    assert report["summary"]["toc_rows"] == 2
    assert report["summary"]["paired_rows"] == 1
    assert report["summary"]["english_book_concept_pairs"] == 1
    assert report["summary"]["original_book_concept_pairs"] == 0
    assert report["evaluation_status"] == "not_evaluated"
    missing = next(r for r in report["books"][0]["rows"] if r["toc_entry_id"] == "toc_1")
    assert missing["unpaired_reason"] == "missing_english"
    assert missing["original_matches"] == missing["english_matches"] == []
    assert rows[0].model_dump_json() == before


@pytest.mark.parametrize("change", ["tamper", "remove", "blank"])
def test_missing_blank_or_modified_english_rejected(tmp_path, change):
    row = json.loads(FIXTURE.read_text())
    if change == "remove":
        del row["evidence"][0]["en_text"]
    else:
        row["evidence"][0]["en_text"] = "changed" if change == "tamper" else "  "
    path = tmp_path / "changed.jsonl"
    path.write_text(json.dumps(row) + "\n")
    with pytest.raises(BookEvidenceImportError):
        load_book_evidence(path)


def test_v2_disguise_and_legacy_comparison_are_rejected(tmp_path):
    row = json.loads(FIXTURE.read_text())
    row.update(schema_version=2, contract_version="book-evidence-v2")
    path = tmp_path / "fake.jsonl"
    path.write_text(json.dumps(row) + "\n")
    with pytest.raises(BookEvidenceImportError):
        load_book_evidence(path)
    with pytest.raises(ValueError, match="book-evidence-v3"):
        compare_english_evidence(
            load_book_evidence(ROOT / "tests/fixtures/book-evidence-v2.jsonl"),
            FEATURES,
            GRAPH,
            MATCHING,
        )


def test_cli_deterministic_report_and_existing_file_protection(tmp_path):
    outputs = [tmp_path / "one.json", tmp_path / "two.json"]
    args = ["compare-english-evidence", "--input", str(FIXTURE)]
    for path in outputs:
        result = CliRunner().invoke(app, [*args, "--output", str(path)])
        assert result.exit_code == 0, result.output
    assert outputs[0].read_bytes() == outputs[1].read_bytes()
    before = FIXTURE.read_bytes()
    result = CliRunner().invoke(app, [*args, "--output", str(FIXTURE)])
    assert result.exit_code != 0 and FIXTURE.read_bytes() == before
