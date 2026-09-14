import json
from pathlib import Path

import pytest

from bookmatch_ml.config import load_reader_config
from bookmatch_ml.reader.profile import AssessmentError, build_reader_profile, load_assessment
from bookmatch_ml.schemas import Assessment, AssessmentResponse

ROOT = Path(__file__).parents[1]
ASSESSMENT_PATH = ROOT / "examples" / "assessment.json"
CONFIG_PATH = ROOT / "configs" / "reader.yaml"


def _assessment() -> Assessment:
    return load_assessment(ASSESSMENT_PATH)


def test_reader_profile_uses_difficulty_weighted_scores() -> None:
    config = load_reader_config(CONFIG_PATH)

    profile = build_reader_profile(_assessment(), config)

    assert profile.topic_id == "operating-systems"
    assert profile.vocabulary == pytest.approx(0.7)
    assert profile.background_knowledge == pytest.approx(1.5 / 2.75)
    assert profile.comprehension == pytest.approx(2.0 / 2.25)
    assert profile.response_count == 6
    assert [detail.question_type for detail in profile.dimension_details] == [
        "vocabulary",
        "background_knowledge",
        "comprehension",
    ]
    assert profile.profile_version == "reader-v1"
    assert profile.config_version == "reader-config-v1"
    assert profile.config_hash == config.content_hash


def test_partial_scores_and_concept_tags_are_preserved_in_readiness() -> None:
    profile = build_reader_profile(_assessment(), load_reader_config(CONFIG_PATH))
    readiness = {item.concept_id: item for item in profile.concept_readiness}

    assert readiness["virtual memory"].score == pytest.approx(0.5)
    assert readiness["memory management"].score == pytest.approx(0.5)
    assert readiness["concurrency"].score == pytest.approx(0.8)
    assert list(readiness) == sorted(readiness)


def test_difficulty_weights_are_config_driven(tmp_path: Path) -> None:
    original = CONFIG_PATH.read_text(encoding="utf-8")
    path = tmp_path / "reader.yaml"
    path.write_text(original.replace("hard: 1.5", "hard: 3.0"), encoding="utf-8")

    baseline = build_reader_profile(_assessment(), load_reader_config(CONFIG_PATH))
    changed = build_reader_profile(_assessment(), load_reader_config(path))

    assert baseline.vocabulary == pytest.approx(0.7)
    assert changed.vocabulary == pytest.approx(0.625)
    assert baseline.config_hash != changed.config_hash


def test_missing_question_type_fails_clearly() -> None:
    assessment = _assessment()
    assessment = assessment.model_copy(
        update={
            "responses": [
                response
                for response in assessment.responses
                if response.question_type != "comprehension"
            ]
        }
    )

    with pytest.raises(AssessmentError, match="comprehension"):
        build_reader_profile(assessment, load_reader_config(CONFIG_PATH))


def test_unsupported_difficulty_fails_clearly() -> None:
    assessment = _assessment()
    responses = list(assessment.responses)
    responses[0] = responses[0].model_copy(update={"difficulty": "expert"})
    assessment = assessment.model_copy(update={"responses": responses})

    with pytest.raises(AssessmentError, match="unsupported question difficulties: expert"):
        build_reader_profile(assessment, load_reader_config(CONFIG_PATH))


@pytest.mark.parametrize(
    ("correct", "score"),
    [(None, None), (True, 1.0)],
)
def test_response_requires_exactly_one_result_value(
    correct: bool | None,
    score: float | None,
) -> None:
    with pytest.raises(ValueError, match="exactly one of correct or score"):
        AssessmentResponse(
            question_id="question-1",
            topic_id="operating-systems",
            question_type="vocabulary",
            difficulty="easy",
            correct=correct,
            score=score,
        )


def test_duplicate_questions_and_topic_mismatch_fail_during_loading(tmp_path: Path) -> None:
    payload = json.loads(ASSESSMENT_PATH.read_text(encoding="utf-8"))
    payload["responses"][1]["question_id"] = payload["responses"][0]["question_id"]
    duplicate_path = tmp_path / "duplicate.json"
    duplicate_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(AssessmentError, match="duplicate question_id"):
        load_assessment(duplicate_path)

    payload = json.loads(ASSESSMENT_PATH.read_text(encoding="utf-8"))
    payload["responses"][0]["topic_id"] = "linear-algebra"
    mismatch_path = tmp_path / "mismatch.json"
    mismatch_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(AssessmentError, match="do not match assessment topic"):
        load_assessment(mismatch_path)


def test_empty_assessment_fails_clearly(tmp_path: Path) -> None:
    payload = {
        "assessment_id": "empty",
        "topic_id": "operating-systems",
        "responses": [],
    }
    path = tmp_path / "empty.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(AssessmentError, match="at least 1 item"):
        load_assessment(path)


def test_reader_profile_is_deterministic() -> None:
    assessment = _assessment()
    config = load_reader_config(CONFIG_PATH)

    first = build_reader_profile(assessment, config).model_dump_json()
    second = build_reader_profile(assessment, config).model_dump_json()

    assert first == second


def test_documented_reader_profile_matches_current_config() -> None:
    generated = build_reader_profile(_assessment(), load_reader_config(CONFIG_PATH))
    documented = json.loads((ROOT / "examples" / "reader_profile.json").read_text(encoding="utf-8"))

    assert generated.model_dump(mode="json") == documented
