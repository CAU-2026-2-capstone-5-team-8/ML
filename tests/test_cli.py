import json
from pathlib import Path

from typer.testing import CliRunner

from bookmatch_ml.book.profile import build_book_profiles
from bookmatch_ml.cli import app
from bookmatch_ml.config import load_feature_config
from bookmatch_ml.data.evidence import assemble_book_evidence
from bookmatch_ml.data.loader import load_canonical_dataset
from bookmatch_ml.io import write_jsonl
from bookmatch_ml.schemas import BookProfile

FIXTURE_DIR = Path(__file__).parent / "fixtures" / "canonical"
CONFIG = Path(__file__).parents[1] / "configs" / "features.yaml"
READER_CONFIG = Path(__file__).parents[1] / "configs" / "reader.yaml"
ASSESSMENT = Path(__file__).parents[1] / "examples" / "assessment.json"
READER_PROFILE = Path(__file__).parents[1] / "examples" / "reader_profile.json"
RANKING_CONFIG = Path(__file__).parents[1] / "configs" / "ranking.yaml"
EVALUATION_CONFIG = Path(__file__).parents[1] / "configs" / "evaluation.yaml"


def _write_book_profiles(path: Path) -> None:
    evidence = assemble_book_evidence(load_canonical_dataset(FIXTURE_DIR))
    profiles = build_book_profiles(evidence, load_feature_config(CONFIG))
    write_jsonl(profiles, path)


def _write_evaluation_profiles(path: Path) -> list[BookProfile]:
    evidence = assemble_book_evidence(load_canonical_dataset(FIXTURE_DIR))
    profiles = build_book_profiles(evidence, load_feature_config(CONFIG))
    first, second = profiles
    second_concept = second.concept_profile.model_copy(
        update={"topic_distribution": {"operating-systems": 1.0}}
    )
    second_difficulty = first.difficulty_profile.model_copy(update={"book_id": second.book_id})
    second = second.model_copy(
        update={
            "concept_profile": second_concept,
            "difficulty_profile": second_difficulty,
        }
    )
    evaluation_profiles = [first, second]
    write_jsonl(evaluation_profiles, path)
    return evaluation_profiles


def test_inspect_data_prints_machine_readable_coverage_report() -> None:
    result = CliRunner().invoke(app, ["inspect-data", "--data-dir", str(FIXTURE_DIR)])

    assert result.exit_code == 0, result.output
    report = json.loads(result.stdout)
    assert report["book_count"] == 2
    assert report["document_count"] == 2
    assert report["toc_entry_count"] == 2
    assert report["source_count"] == 2
    assert report["books"][0]["coverage"]["prose_document_count"] == 1


def test_build_book_profiles_writes_deterministic_jsonl(tmp_path: Path) -> None:
    first = tmp_path / "first.jsonl"
    second = tmp_path / "second.jsonl"
    arguments = [
        "build-book-profiles",
        "--data-dir",
        str(FIXTURE_DIR),
        "--config",
        str(CONFIG),
    ]

    first_result = CliRunner().invoke(app, [*arguments, "--output", str(first)])
    second_result = CliRunner().invoke(app, [*arguments, "--output", str(second)])

    assert first_result.exit_code == 0, first_result.output
    assert second_result.exit_code == 0, second_result.output
    assert first.read_bytes() == second.read_bytes()
    profiles = [json.loads(line) for line in first.read_text(encoding="utf-8").splitlines()]
    assert len(profiles) == 2
    assert profiles[0]["feature_version"] == "book-v1"
    assert profiles[0]["difficulty_profile"]["analyzed_document_count"] == 1
    assert profiles[1]["difficulty_profile"]["lexical_difficulty"] is None


def test_ablation_command_writes_three_variants(tmp_path: Path) -> None:
    output = tmp_path / "ablation.json"

    result = CliRunner().invoke(
        app,
        [
            "ablate-book-profile",
            "--data-dir",
            str(FIXTURE_DIR),
            "--book-id",
            "book_11111111111111111111",
            "--output",
            str(output),
            "--config",
            str(CONFIG),
        ],
    )

    assert result.exit_code == 0, result.output
    report = json.loads(output.read_text(encoding="utf-8"))
    assert [variant["evidence_mode"] for variant in report["variants"]] == [
        "toc_only",
        "toc_description",
        "all_available",
    ]


def test_reader_profile_command_prints_profile() -> None:
    result = CliRunner().invoke(
        app,
        [
            "reader-profile",
            "--input",
            str(ASSESSMENT),
            "--config",
            str(READER_CONFIG),
        ],
    )

    assert result.exit_code == 0, result.output
    profile = json.loads(result.stdout)
    assert profile["vocabulary"] == 0.7
    assert profile["profile_version"] == "reader-v1"


def test_reader_profile_command_writes_deterministic_output(tmp_path: Path) -> None:
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"
    arguments = [
        "reader-profile",
        "--input",
        str(ASSESSMENT),
        "--config",
        str(READER_CONFIG),
    ]

    first_result = CliRunner().invoke(app, [*arguments, "--output", str(first)])
    second_result = CliRunner().invoke(app, [*arguments, "--output", str(second)])

    assert first_result.exit_code == 0, first_result.output
    assert second_result.exit_code == 0, second_result.output
    assert first.read_bytes() == second.read_bytes()
    assert json.loads(first.read_text(encoding="utf-8"))["response_count"] == 6


def test_rank_command_returns_topic_filtered_top_k(tmp_path: Path) -> None:
    books = tmp_path / "books.jsonl"
    _write_book_profiles(books)

    result = CliRunner().invoke(
        app,
        [
            "rank",
            "--reader",
            str(READER_PROFILE),
            "--books",
            str(books),
            "--limit",
            "5",
            "--config",
            str(RANKING_CONFIG),
        ],
    )

    assert result.exit_code == 0, result.output
    response = json.loads(result.stdout)
    assert response["model_version"] == "rank-v1"
    assert [item["book_id"] for item in response["items"]] == ["book_11111111111111111111"]
    assert set(response["items"][0]["components"]) == {
        "topic_fit",
        "vocabulary_fit",
        "knowledge_fit",
        "comprehension_fit",
    }


def test_rank_command_supports_specific_cross_topic_fit(tmp_path: Path) -> None:
    books = tmp_path / "books.jsonl"
    _write_book_profiles(books)

    result = CliRunner().invoke(
        app,
        [
            "rank",
            "--reader",
            str(READER_PROFILE),
            "--books",
            str(books),
            "--book-id",
            "book_22222222222222222222",
            "--config",
            str(RANKING_CONFIG),
        ],
    )

    assert result.exit_code == 0, result.output
    item = json.loads(result.stdout)["items"][0]
    assert item["book_id"] == "book_22222222222222222222"
    assert item["score"] == 0.0
    assert item["components"]["topic_fit"] == 0.0


def test_rank_command_writes_deterministic_output(tmp_path: Path) -> None:
    books = tmp_path / "books.jsonl"
    first = tmp_path / "first-ranking.json"
    second = tmp_path / "second-ranking.json"
    _write_book_profiles(books)
    arguments = [
        "rank",
        "--reader",
        str(READER_PROFILE),
        "--books",
        str(books),
        "--config",
        str(RANKING_CONFIG),
    ]

    first_result = CliRunner().invoke(app, [*arguments, "--output", str(first)])
    second_result = CliRunner().invoke(app, [*arguments, "--output", str(second)])

    assert first_result.exit_code == 0, first_result.output
    assert second_result.exit_code == 0, second_result.output
    assert first.read_bytes() == second.read_bytes()


def test_evaluate_command_writes_deterministic_combined_report(tmp_path: Path) -> None:
    books = tmp_path / "books.jsonl"
    labels = tmp_path / "labels.json"
    first = tmp_path / "first-evaluation.json"
    second = tmp_path / "second-evaluation.json"
    profiles = _write_evaluation_profiles(books)
    labels.write_text(
        json.dumps(
            {
                "label_version": "synthetic-test-v1",
                "is_synthetic": True,
                "topic_id": "operating-systems",
                "items": [
                    {"book_id": profiles[0].book_id, "human_rank": 1},
                    {"book_id": profiles[1].book_id, "human_rank": 2},
                ],
            }
        ),
        encoding="utf-8",
    )
    arguments = [
        "evaluate",
        "--data-dir",
        str(FIXTURE_DIR),
        "--books",
        str(books),
        "--reader",
        str(READER_PROFILE),
        "--difficulty-labels",
        str(labels),
        "--ablation-book-id",
        profiles[0].book_id,
        "--config",
        str(EVALUATION_CONFIG),
        "--feature-config",
        str(CONFIG),
        "--ranking-config",
        str(RANKING_CONFIG),
    ]

    first_result = CliRunner().invoke(app, [*arguments, "--output", str(first)])
    second_result = CliRunner().invoke(app, [*arguments, "--output", str(second)])

    assert first_result.exit_code == 0, first_result.output
    assert second_result.exit_code == 0, second_result.output
    assert first.read_bytes() == second.read_bytes()
    report = json.loads(first.read_text(encoding="utf-8"))
    assert report["evaluation_version"] == "evaluation-v1"
    assert report["difficulty"]["comparable_book_count"] == 2
    assert report["recommendation"]["candidate_count"] == 2
    assert len(report["ablation_comparisons"]) == 2
