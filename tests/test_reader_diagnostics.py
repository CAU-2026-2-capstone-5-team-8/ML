"""Depth evidence must distinguish identical aggregate readiness scores."""

from pathlib import Path

import pytest

from bookmatch_ml.config import load_reader_config
from bookmatch_ml.reader.diagnostics import build_reader_diagnostics
from bookmatch_ml.schemas import Assessment, AssessmentResponse

CONFIG = load_reader_config(Path(__file__).parents[1] / "configs/reader.yaml")
TYPES = ("vocabulary", "background_knowledge", "comprehension")


def assessment(difficulty="easy", score=1.0):
    return Assessment(
        assessment_id="depth-example",
        topic_id="operating-systems",
        responses=[
            AssessmentResponse(
                question_id=f"q-{kind}",
                topic_id="operating-systems",
                concept_id="process",
                concept_tags=["process"],
                question_type=kind,
                difficulty=difficulty,
                score=score,
            )
            for kind in TYPES
        ],
    )


def test_easy_and_hard_perfect_scores_keep_different_evidence():
    easy = build_reader_diagnostics(assessment(), CONFIG)
    hard = build_reader_diagnostics(assessment("hard"), CONFIG)
    assert easy.concepts[0].observed_score == hard.concepts[0].observed_score == 1
    assert easy.concepts[0].response_count == hard.concepts[0].response_count == 3
    easy_cells = easy.concepts[0].evidence
    hard_cells = hard.concepts[0].evidence
    assert {c.difficulty for c in easy_cells if c.response_count} == {"easy"}
    assert {c.difficulty for c in hard_cells if c.response_count} == {"hard"}
    assert all(c.score is None for c in hard_cells if c.difficulty != "hard")
    assert easy.concepts[0].next_check.difficulty == "medium"
    assert hard.concepts[0].next_check.difficulty == "easy"
    assert hard.concepts[0].next_check.reason == "unassessed"


def test_partial_credit_preserved_and_review_precedes_unassessed_cells():
    data = assessment("medium", 0.5)
    result = build_reader_diagnostics(data, CONFIG).concepts[0]
    assert result.observed_score == 0.5
    assert all(c.score == 0.5 for c in result.evidence if c.response_count)
    assert result.next_check.reason == "review_observed_gap"
    assert result.next_check.difficulty == "medium"
    assert result.next_check.question_type == "vocabulary"


def test_duplicate_tag_and_concept_id_do_not_double_count():
    result = build_reader_diagnostics(assessment(), CONFIG)
    assert len(result.concepts) == 1
    assert result.concepts[0].response_count == 3
    assert sum(c.response_count for c in result.concepts[0].evidence) == 3


def test_order_is_deterministic_and_retains_question_ids():
    data = assessment()
    reversed_data = data.model_copy(update={"responses": list(reversed(data.responses))})
    first = build_reader_diagnostics(data, CONFIG)
    assert first == build_reader_diagnostics(reversed_data, CONFIG)
    assert sorted(q for c in first.concepts[0].evidence for q in c.question_ids) == [
        "q-background_knowledge",
        "q-comprehension",
        "q-vocabulary",
    ]


def test_untagged_answers_are_counted_but_not_invented_as_concepts():
    data = assessment()
    data.responses[0].concept_id = None
    data.responses[0].concept_tags = []
    result = build_reader_diagnostics(data, CONFIG)
    assert result.untagged_response_count == 1
    assert result.concepts[0].response_count == 2
    assert result.response_count == 3


def test_all_observed_full_credit_does_not_claim_mastery():
    data = assessment()
    for level in ("medium", "hard"):
        for response in assessment(level).responses:
            data.responses.append(
                response.model_copy(update={"question_id": f"{level}-{response.question_id}"})
            )
    result = build_reader_diagnostics(data, CONFIG).concepts[0]
    assert result.next_check is None
    assert result.fully_observed is True
    assert "mastery" not in result.model_dump()


def test_no_concept_answers_return_an_empty_concept_list():
    data = assessment()
    for response in data.responses:
        response.concept_id = None
        response.concept_tags = []
    result = build_reader_diagnostics(data, CONFIG)
    assert result.concepts == []
    assert result.untagged_response_count == 3


def test_unsupported_difficulty_is_rejected():
    with pytest.raises(ValueError, match="unsupported"):
        build_reader_diagnostics(assessment("unknown"), CONFIG)


def test_difficulty_order_is_not_inferred_from_numeric_weights():
    from bookmatch_ml.config import LoadedReaderConfig

    altered = CONFIG.config.model_copy(
        update={"difficulty_weights": {"hard": 1.0, "easy": 2.0, "medium": 3.0}}
    )
    config = LoadedReaderConfig(config=altered, content_hash=CONFIG.content_hash)
    assert (
        build_reader_diagnostics(assessment(), config).concepts[0].next_check.difficulty == "medium"
    )
