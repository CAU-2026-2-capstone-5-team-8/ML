import csv
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest


def load_tool(name):
    spec = importlib.util.spec_from_file_location(name, Path("scripts") / f"{name}.py")
    tool = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tool)
    return tool


def experiment(tmp_path):
    # Three books: human prefers a > b > c; v2 reverses it; v3 ties a/b.
    report = {
        "version": "learning-order-comparison-v2",
        "snapshotId": "fixture",
        "candidatesHash": "sha256:fixture",
        "readerEvidence": "synthetic_scenarios_not_real_assessments",
        "scenarios": [
            {
                "name": "case",
                "observations": [],
                "before": [
                    {"bookId": b, "title": b, "rankGroup": i + 1}
                    for i, b in enumerate(["c", "b", "a"])
                ],
                "after": [
                    {"bookId": b, "title": b, "rankGroup": g}
                    for b, g in [("a", 1), ("b", 1), ("c", 2)]
                ],
            }
        ],
    }
    load_tool("compare_learning_order").write_outputs(report, tmp_path / "run")
    comparison = tmp_path / "run/comparison.json"
    sheet = tmp_path / "run/human-review.csv"
    with sheet.open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    return comparison, sheet, rows


def save(sheet, rows):
    with sheet.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def rated(rows, reviewer="r1"):
    return [
        {
            **r,
            "suitability_1_to_5": str(5 - i),
            "reason": "Fixture only",
            "reviewer": reviewer,
            "reviewed_at": "2026-10-06",
        }
        for i, r in enumerate(rows)
    ]


def test_worksheet_is_bound_to_exact_comparison_bytes(tmp_path):
    comparison, _, rows = experiment(tmp_path)
    expected = "sha256:" + hashlib.sha256(comparison.read_bytes()).hexdigest()
    assert all(row.get("comparison_hash") == expected for row in rows)


def test_blank_review_cannot_be_reported_as_accuracy(tmp_path):
    comparison, sheet, _ = experiment(tmp_path)
    result = load_tool("evaluate_learning_order").evaluate(comparison, [sheet])
    assert result["ratedRows"] == 0
    assert result["status"] == "awaiting_reviews"
    assert result["reviewers"] == []
    assert result["accuracyMeasured"] is False


def test_pairwise_counts_keep_model_ties_and_reviewers_separate(tmp_path):
    comparison, sheet, rows = experiment(tmp_path)
    save(sheet, rated(rows))
    second = tmp_path / "second.csv"
    reversed_rows = rated(rows, "r2")
    for i, row in enumerate(reversed_rows):
        row["suitability_1_to_5"] = str(i + 1)
    save(second, reversed_rows)
    result = load_tool("evaluate_learning_order").evaluate(comparison, [sheet, second])
    first = result["reviewers"][0]["scenarios"][0]
    assert first["before"]["disagree"] == 3
    assert first["after"]["agree"] == 2
    assert first["after"]["modelTie"] == 1
    assert first["after"]["agreementOnDecidedPairs"] == 1
    assert first["after"]["decidedPairCoverage"] == pytest.approx(2 / 3)
    assert result["reviewers"][1]["scenarios"][0]["after"]["disagree"] == 2
    assert result["reviewerCount"] == 2
    assert result["accuracyMeasured"] is False


@pytest.mark.parametrize(
    "field,value",
    [
        ("comparison_hash", "changed"),
        ("snapshot_id", "other"),
        ("candidates_hash", "other"),
        ("scenario", "unknown"),
        ("book_id", "unknown"),
        ("title", "changed"),
        ("observations_json", "[{}]"),
        ("suitability_1_to_5", "6"),
        ("suitability_1_to_5", "NaN"),
        ("reviewer", ""),
        ("reviewed_at", "yesterday"),
        ("reason", ""),
    ],
)
def test_mixed_or_invalid_reviews_are_rejected(tmp_path, field, value):
    comparison, sheet, rows = experiment(tmp_path)
    rows = rated(rows)
    rows[0][field] = value
    save(sheet, rows)
    with pytest.raises(ValueError):
        load_tool("evaluate_learning_order").evaluate(comparison, [sheet])


def test_duplicate_votes_and_changed_comparison_are_rejected(tmp_path):
    comparison, sheet, rows = experiment(tmp_path)
    save(sheet, rated(rows))
    tool = load_tool("evaluate_learning_order")
    with pytest.raises(ValueError, match="duplicate"):
        tool.evaluate(comparison, [sheet, sheet])
    data = json.loads(comparison.read_text(encoding="utf-8"))
    data["scenarios"][0]["after"][0]["rankGroup"] = 3
    comparison.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ValueError, match="hash"):
        tool.evaluate(comparison, [sheet])


def test_human_ties_and_missing_scores_are_not_model_errors(tmp_path):
    comparison, sheet, rows = experiment(tmp_path)
    rows = rated(rows)
    rows[0]["suitability_1_to_5"] = "4"
    rows[2]["suitability_1_to_5"] = ""
    save(sheet, rows)
    result = load_tool("evaluate_learning_order").evaluate(comparison, [sheet])
    stats = result["reviewers"][0]["scenarios"][0]["after"]
    assert stats["humanTie"] == 1
    assert stats["agree"] == stats["disagree"] == 0
    assert stats["agreementOnDecidedPairs"] is None
    assert result["ratedRows"] == 2


def test_cli_preserves_existing_report_and_is_deterministic(tmp_path):
    import subprocess
    import sys

    comparison, sheet, rows = experiment(tmp_path)
    save(sheet, rated(rows))
    outputs = [tmp_path / "first.json", tmp_path / "second.json"]
    for output in outputs:
        result = subprocess.run(
            [
                sys.executable,
                "scripts/evaluate_learning_order.py",
                "--comparison",
                str(comparison),
                "--reviews",
                str(sheet),
                "--output",
                str(output),
            ],
            capture_output=True,
        )
        assert result.returncode == 0, result.stderr
    assert outputs[0].read_bytes() == outputs[1].read_bytes()
    original = outputs[1].read_bytes()
    result = subprocess.run(
        [
            sys.executable,
            "scripts/evaluate_learning_order.py",
            "--comparison",
            str(comparison),
            "--reviews",
            str(sheet),
            "--output",
            str(outputs[1]),
        ],
        capture_output=True,
    )
    assert result.returncode != 0
    assert outputs[1].read_bytes() == original


def test_only_common_candidates_enter_paired_comparison(tmp_path):
    comparison, _, _ = experiment(tmp_path)
    report = json.loads(comparison.read_text(encoding="utf-8"))
    report["scenarios"][0]["before"].pop(0)  # c is only returned by v3
    load_tool("compare_learning_order").write_outputs(report, tmp_path / "subset")
    comparison = tmp_path / "subset/comparison.json"
    sheet = tmp_path / "subset/human-review.csv"
    with sheet.open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    save(sheet, rated(rows))
    result = load_tool("evaluate_learning_order").evaluate(comparison, [sheet])
    scenario = result["reviewers"][0]["scenarios"][0]
    assert scenario["commonCandidateCount"] == 2
    assert scenario["ratedOutsideCommonBooks"] == 1
    assert scenario["before"]["disagree"] == 1
    assert scenario["after"]["modelTie"] == 1
    assert scenario["after"]["agree"] == 0
