import hashlib
import json
from pathlib import Path

from typer.testing import CliRunner

from bookmatch_ml.cli import app
from bookmatch_ml.concept_v2.graph import load_concept_graph
from bookmatch_ml.concept_v2.profile import load_concept_matching_config
from bookmatch_ml.config import load_feature_config
from bookmatch_ml.data.book_evidence import load_book_evidence
from bookmatch_ml.evaluation.english_evidence import prepare_english_evidence_review

ROOT = Path(__file__).parents[1]
FIXTURE = ROOT / "tests/fixtures/book-evidence-v3.jsonl"
FEATURES = load_feature_config(ROOT / "configs/features.yaml")
GRAPH = load_concept_graph(ROOT / "configs/concept_graph.yaml", FEATURES)
MATCHING = load_concept_matching_config(ROOT / "configs/concept_matching_v2.yaml")


def test_packet_preserves_exact_evidence_and_keeps_unpaired_and_unmatched_rows():
    records = load_book_evidence(FIXTURE)
    before = records[0].model_dump_json()
    report = prepare_english_evidence_review(records, FEATURES, GRAPH, MATCHING)
    assert report["evaluation_status"] == "not_evaluated"
    assert report["translation_api_requests"] == 0
    assert report["summary"]["changed_toc_rows"] == 1
    book = report["books"][0]
    assert len(book["rows"]) == 2
    added = next(r for r in book["rows"] if r["toc_entry_id"] == "toc_0")
    expected = [{"topic": "linear-algebra", "concept_id": "matrix"}]
    assert added["added_concepts"] == added["supports_book_added_concepts"] == expected
    missing = next(r for r in book["rows"] if r["toc_entry_id"] == "toc_1")
    assert missing["translation_status"] == "missing_english"
    assert missing["added_concepts"] == missing["removed_concepts"] == []
    assert missing["evidence"]["en_text"] is None
    sample = book["prose_samples"][0]
    source = next(e for e in records[0].evidence if e.document_id)
    assert sample["evidence"] == source.model_dump(mode="json")
    assert (
        sample["original_text_hash"] == "sha256:" + hashlib.sha256(source.text.encode()).hexdigest()
    )
    assert (
        sample["english_text_hash"]
        == "sha256:" + hashlib.sha256(source.en_text.encode()).hexdigest()
    )
    for row in [*book["rows"], sample]:
        for field in ["human_translation_review", "human_match_review", "reviewer", "review_notes"]:
            assert row[field] is None
    assert records[0].model_dump_json() == before


def test_row_changes_are_not_mistaken_for_new_book_concepts():
    records = load_book_evidence(FIXTURE)
    record = records[0]
    items = []
    for item in record.evidence:
        if item.toc_entry_id == "toc_1":
            item = item.model_copy(update={"text": "Matrix", "en_text": "Overview"})
        elif item.document_id:
            item = item.model_copy(update={"en_text": None})
        items.append(item)
    report = prepare_english_evidence_review(
        [record.model_copy(update={"evidence": items})], FEATURES, GRAPH, MATCHING
    )
    book = report["books"][0]
    assert report["summary"]["changed_toc_rows"] == 2
    assert book["added_concepts"] == book["removed_concepts"] == []
    assert any(row["added_concepts"] for row in book["rows"])
    assert any(row["removed_concepts"] for row in book["rows"])
    for row in book["rows"]:
        assert row["supports_book_added_concepts"] == row["supports_book_removed_concepts"] == []
    sample = book["prose_samples"][0]
    assert sample["translation_status"] == "missing_english"
    assert sample["english_text_hash"] is None
    assert sample["evidence"]["en_text"] is None


def test_review_cli_replays_and_preserves_input_and_existing_output(tmp_path):
    args = ["compare-english-evidence", "--input", str(FIXTURE), "--review-packet"]
    outputs = [tmp_path / "one.json", tmp_path / "two.json"]
    before = FIXTURE.read_bytes()
    for output in outputs:
        result = CliRunner().invoke(app, [*args, "--output", str(output)])
        assert result.exit_code == 0, result.output
    content = outputs[0].read_bytes()
    assert content == outputs[1].read_bytes()
    report = json.loads(content)
    assert report["report_version"] == "english-evidence-review-v1"
    assert report["book_evidence_hash"] == "sha256:" + hashlib.sha256(before).hexdigest()
    for output in [outputs[0], FIXTURE]:
        result = CliRunner().invoke(app, [*args, "--output", str(output)])
        assert result.exit_code != 0
    assert outputs[0].read_bytes() == content
    assert FIXTURE.read_bytes() == before


def test_review_cli_rejects_tampered_input_without_creating_output(tmp_path):
    row = json.loads(FIXTURE.read_text())
    row["evidence"][0]["en_text"] = "Tampered English"
    input_path, output = tmp_path / "input.jsonl", tmp_path / "review.json"
    input_path.write_text(json.dumps(row) + "\n")
    result = CliRunner().invoke(
        app,
        [
            "compare-english-evidence",
            "--input",
            str(input_path),
            "--output",
            str(output),
            "--review-packet",
        ],
    )
    assert result.exit_code != 0
    assert not output.exists()
