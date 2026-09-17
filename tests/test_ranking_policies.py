"""Regression checks for the separate ranking-policy experiment."""

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from bookmatch_ml.book.profile import build_book_profiles
from bookmatch_ml.cli import app
from bookmatch_ml.config import (
    ConfigError,
    load_feature_config,
    load_ranking_config,
    load_ranking_policy_config,
)
from bookmatch_ml.data.evidence import assemble_book_evidence
from bookmatch_ml.data.loader import load_canonical_dataset
from bookmatch_ml.evaluation.ranking_policies import evaluate_ranking_policies
from bookmatch_ml.io import write_jsonl
from bookmatch_ml.ranking.loader import load_reader_profile
from bookmatch_ml.ranking.matching import rank_books

ROOT = Path(__file__).parents[1]
FIXTURE_DIR = Path(__file__).parent / "fixtures" / "canonical"
RANKING = load_ranking_config(ROOT / "configs" / "ranking.yaml")
POLICY = load_ranking_policy_config(ROOT / "configs" / "ranking_policies.yaml")
READER = load_reader_profile(ROOT / "examples" / "reader_profile.json")


def _profiles():
    books = build_book_profiles(
        assemble_book_evidence(load_canonical_dataset(FIXTURE_DIR)),
        load_feature_config(ROOT / "configs" / "features.yaml"),
    )
    sparse = books[1]
    sparse = sparse.model_copy(
        update={
            "concept_profile": sparse.concept_profile.model_copy(
                update={"topic_distribution": {"operating-systems": 1.0}}
            )
        }
    )
    return [books[0], sparse]


def _policy_config(*, minimum: float = 0.65, top_k: int = 1):
    return POLICY.model_copy(
        update={
            "config": POLICY.config.model_copy(
                update={"minimum_component_weight_coverage": minimum, "top_k": top_k}
            )
        }
    )


def test_renormalized_is_the_existing_ranking_and_limited_candidates_are_visible() -> None:
    books = _profiles()
    baseline = rank_books(READER, books, RANKING, limit=2).items
    report = evaluate_ranking_policies(
        READER, books, RANKING, _policy_config(), canonical_book_count=2
    )
    renormalized, minimum, two_stage = report.policies

    assert [item.book_id for item in renormalized.items] == [item.book_id for item in baseline]
    assert [item.readiness_score for item in renormalized.items] == [
        item.score for item in baseline
    ]
    assert all(item.group == "readiness" for item in renormalized.items)
    sparse_id = books[1].book_id
    assert baseline[0].book_id == sparse_id
    assert baseline[0].score == 1.0
    assert baseline[0].component_weight_coverage == 0.35
    for result in (minimum, two_stage):
        limited = next(item for item in result.items if item.book_id == sparse_id)
        assert limited.readiness_score is None
        assert limited.diagnostic_renormalized_score == 1.0
        assert limited.component_weight_coverage == 0.35
        assert limited.components.vocabulary_fit is None
        assert limited.components.comprehension_fit is None
        assert limited.unavailable_components == [
            "vocabulary_fit",
            "knowledge_fit",
            "comprehension_fit",
        ]
        assert limited.readiness_evidence == "partial"
        assert "below minimum" in limited.eligibility_reason
    assert minimum.items[-1].group == "ineligible"
    assert minimum.items[-1].topic_only_score == 1.0
    assert two_stage.items[-1].group == "topic_only"
    assert two_stage.items[-1].topic_only_score == 1.0
    assert report.policy_config_version == "ranking-policies-config-v1"
    assert report.policy_config_hash == POLICY.content_hash
    assert report.ranking_config_hash == RANKING.content_hash
    for result in report.policies:
        assert result.readiness_candidate_count + result.evidence_limited_count == 2
        assert {item.book_id for item in result.items} == {book.book_id for book in books}
        assert [item.display_position for item in result.items] == [1, 2]
        assert (
            result.readiness_top_k
            == [item.book_id for item in result.items if item.group == "readiness"][: report.top_k]
        )
        for item in result.items:
            assert (
                item.display_position_change == item.renormalized_position - item.display_position
            )


def test_minimum_coverage_boundary_is_inclusive_and_order_is_deterministic() -> None:
    books = _profiles()
    at_boundary = evaluate_ranking_policies(
        READER, books, RANKING, _policy_config(minimum=0.35), canonical_book_count=2
    )
    minimum = at_boundary.policies[1]
    assert minimum.readiness_candidate_count == 2
    assert minimum.evidence_limited_count == 0
    assert all(item.group == "readiness" for item in minimum.items)

    first = evaluate_ranking_policies(
        READER, books, RANKING, _policy_config(), canonical_book_count=2
    )
    second = evaluate_ranking_policies(
        READER, list(reversed(books)), RANKING, _policy_config(), canonical_book_count=2
    )
    assert first == second
    assert [item.display_position for item in first.policies[2].items] == [1, 2]
    assert first.policies[2].items[0].display_position_change == 1
    assert first.policies[2].items[1].display_position_change == -1
    assert all(difference.comparable for difference in first.top_k_differences)
    assert all(
        len(difference.left_only) == len(difference.right_only) == 1
        for difference in first.top_k_differences
    )


def test_top_k_difference_is_not_claimed_when_eligible_group_is_too_small() -> None:
    report = evaluate_ranking_policies(
        READER, _profiles(), RANKING, _policy_config(top_k=2), canonical_book_count=2
    )
    assert report.policies[0].readiness_top_k_complete
    assert not report.policies[1].readiness_top_k_complete
    assert all(not difference.comparable for difference in report.top_k_differences)
    assert all(
        difference.left_only == difference.right_only == []
        for difference in report.top_k_differences
    )


def test_two_stage_topic_only_ties_use_book_id_and_keep_null_readiness_scores() -> None:
    rich, sparse = _profiles()
    clone_id = "book_00000000000000000000"
    clone = sparse.model_copy(
        update={
            "book_id": clone_id,
            "concept_profile": sparse.concept_profile.model_copy(update={"book_id": clone_id}),
            "difficulty_profile": sparse.difficulty_profile.model_copy(
                update={"book_id": clone_id}
            ),
        }
    )
    report = evaluate_ranking_policies(
        READER,
        [sparse, rich, clone],
        RANKING,
        _policy_config(),
        canonical_book_count=3,
    )
    two_stage = report.policies[2]
    assert [item.book_id for item in two_stage.items[1:]] == [clone_id, sparse.book_id]
    assert [item.group for item in two_stage.items] == [
        "readiness",
        "topic_only",
        "topic_only",
    ]
    assert all(item.readiness_score is None for item in two_stage.items[1:])


@pytest.mark.parametrize(
    ("replacement", "message"),
    [
        ("minimum_component_weight_coverage: 1.1", "less than or equal to 1"),
        ("minimum_component_weight_coverage: 0", "greater than 0"),
        ("  - two_stage\n  - two_stage", "each supported policy exactly once"),
    ],
)
def test_policy_configuration_validation(tmp_path: Path, replacement: str, message: str) -> None:
    source = (ROOT / "configs" / "ranking_policies.yaml").read_text(encoding="utf-8")
    if replacement.startswith("  - two_stage"):
        invalid = source.replace("  - two_stage", replacement)
    else:
        invalid = source.replace("minimum_component_weight_coverage: 0.65", replacement)
    path = tmp_path / "ranking_policies.yaml"
    path.write_text(invalid, encoding="utf-8")
    with pytest.raises(ConfigError, match=message):
        load_ranking_policy_config(path)


def test_cli_produces_repeatable_report_and_rejects_mismatched_snapshot(tmp_path: Path) -> None:
    profiles = _profiles()
    books = tmp_path / "books.jsonl"
    write_jsonl(profiles, books)
    config = tmp_path / "ranking_policies.yaml"
    config.write_text(
        (ROOT / "configs" / "ranking_policies.yaml")
        .read_text(encoding="utf-8")
        .replace("top_k: 3", "top_k: 1"),
        encoding="utf-8",
    )
    arguments = [
        "evaluate-ranking-policies",
        "--data-dir",
        str(FIXTURE_DIR),
        "--books",
        str(books),
        "--reader",
        str(ROOT / "examples" / "reader_profile.json"),
        "--policy-config",
        str(config),
    ]
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"
    for output in (first, second):
        result = CliRunner().invoke(app, [*arguments, "--output", str(output)])
        assert result.exit_code == 0, result.output
    assert first.read_bytes() == second.read_bytes()
    report = json.loads(first.read_text(encoding="utf-8"))
    assert report["candidate_count"] == 2
    assert [policy["policy"] for policy in report["policies"]] == [
        "renormalized",
        "minimum_coverage",
        "two_stage",
    ]

    write_jsonl(profiles[:1], books)
    result = CliRunner().invoke(app, [*arguments, "--output", str(first)])
    assert result.exit_code == 1
    assert "do not match canonical dataset" in result.output
