"""Policy regression tests: synthetic responses, not evidence of recommendation accuracy."""

from copy import deepcopy

import pytest
from fastapi.testclient import TestClient

from bookmatch_ml.api import create_app

CLIENT = TestClient(create_app())


def book(identity, concepts):
    return {
        "bookId": identity,
        "coveredConcepts": concepts,
        "sourceArtifactVersion": "synthetic-v1",
        "sourceArtifactHash": "sha256:" + "a" * 64,
    }


def observation(concept, correct, count=1, ability="application"):
    return {
        "conceptId": concept,
        "ability": ability,
        "responseCount": count,
        "correctCount": correct,
    }


def request(books, observations=(), version="concept-learning-v3", limit=20):
    return {
        "topicId": "operating-systems",
        "ability": "application",
        "modelVersion": version,
        "candidateBooks": books,
        "observations": list(observations),
        "limit": limit,
    }


def calculate(payload):
    response = CLIENT.post("/ml/learning-fit", json=payload)
    assert response.status_code == 200, response.text
    return response.json()


def test_different_readers_reverse_ranking_for_the_relevant_prerequisite():
    books = [book("a-memory", ["memory management"]), book("z-scheduling", ["scheduling"])]
    for process, architecture, expected in [(1, 0, "z-scheduling"), (0, 1, "a-memory")]:
        result = calculate(
            request(
                books,
                [
                    observation("process", process),
                    observation("computer architecture", architecture),
                ],
            )
        )
        assert result["items"][0]["bookId"] == expected


def test_same_category_compares_observed_error_rate_instead_of_book_id():
    books = [book("a-memory", ["memory management"]), book("z-scheduling", ["scheduling"])]
    obs = [observation("computer architecture", 0, 2), observation("process", 1, 2)]
    baseline = calculate(request(books, obs, "concept-learning-v2"))
    result = calculate(request(books, obs))
    assert [b["bookId"] for b in baseline["items"]] == ["a-memory", "z-scheduling"]
    assert [b["bookId"] for b in result["items"]] == ["z-scheduling", "a-memory"]
    assert result["items"][0]["personalization"]["prerequisiteErrorRate"] == 0.5
    assert result["items"][1]["personalization"]["prerequisiteErrorRate"] == 1


def test_more_observed_prerequisites_win_within_check_first():
    books = [book("a-unknown", ["virtual memory"]), book("z-partial", ["concurrency"])]
    result = calculate(request(books, [observation("process", 1)]))
    assert [b["bookId"] for b in result["items"]] == ["z-partial", "a-unknown"]
    metrics = result["items"][1]["personalization"]
    assert metrics["prerequisiteErrorRate"] is None
    assert metrics["prerequisiteCoverage"] == 0
    assert metrics["prerequisiteUnknownCount"] == 2


def test_practice_overlap_and_density_break_ties_without_rewarding_long_toc():
    books = [book("a-broad", ["scheduling", "thread"]), book("z-focused", ["scheduling"])]
    result = calculate(
        request(
            books,
            [observation("process", 1), observation("scheduling", 0), observation("thread", 1)],
        )
    )
    assert [b["bookId"] for b in result["items"]] == ["z-focused", "a-broad"]
    assert result["items"][0]["personalization"]["practiceCoverage"] == 1
    assert result["items"][1]["personalization"]["practiceDensity"] == 0.5
    assert result["items"][1]["personalization"]["observedCorrectTargetShare"] == 0.5


def test_unknown_targets_are_not_invented_as_new_learning_goals():
    books = [book("a-known", ["scheduling"]), book("z-unknown", ["thread"])]
    result = calculate(request(books, [observation("process", 1), observation("scheduling", 1)]))
    assert {b["rankGroup"] for b in result["items"]} == {1}
    assert all(b["tieCount"] == 2 for b in result["items"])
    assert result["items"][1]["personalization"]["targetUnknownCount"] == 1


def test_missing_prerequisite_graph_cannot_outrank_observed_foundation_gap():
    result = calculate(
        request(
            [book("a-no-graph", ["computer architecture"]), book("z-gap", ["scheduling"])],
            [observation("process", 0)],
        )
    )
    assert [item["bookId"] for item in result["items"]] == ["z-gap", "a-no-graph"]


def test_adding_correct_prerequisite_does_not_increase_observed_burden():
    payload = request([book("one", ["concurrency"])], [observation("thread", 0)])
    before = calculate(payload)["items"][0]
    payload["observations"].append(observation("process", 1))
    after = calculate(payload)["items"][0]
    assert (
        after["personalization"]["prerequisiteErrorRate"]
        <= before["personalization"]["prerequisiteErrorRate"]
    )
    assert after["inferredPrerequisites"] == before["inferredPrerequisites"]


def test_equivalent_books_tie_even_when_limit_splits_group_and_order_is_stable():
    books = [book("z", ["thread"]), book("a", ["scheduling"])]
    payload = request(books, limit=1)
    result = calculate(payload)
    assert result["items"][0]["bookId"] == "a"
    assert result["items"][0]["tieCount"] == 2
    payload["candidateBooks"].reverse()
    assert calculate(payload) == result


def test_duplicate_concepts_do_not_inflate_v3_but_v2_keeps_validation():
    payload = request([book("one", ["scheduling"])], [observation("process", 1)])
    once = calculate(payload)
    payload["candidateBooks"][0]["coveredConcepts"] *= 3
    assert calculate(payload) == once
    payload["modelVersion"] = "concept-learning-v2"
    assert CLIENT.post("/ml/learning-fit", json=payload).status_code == 422


@pytest.mark.parametrize("correct", [0, 1, 2, 3])
def test_correcting_answers_never_increases_same_book_observed_burden(correct):
    payload = request(
        [book("one", ["concurrency"])],
        [observation("process", correct, 3), observation("thread", 0)],
    )
    item = calculate(payload)["items"][0]
    assert item["personalization"]["prerequisiteErrorRate"] == pytest.approx(
        ((3 - correct) / 3 + 1) / 2
    )
    assert item["coveredConcepts"] == ["concurrency"]
    assert item["inferredPrerequisites"] == ["process", "thread"]


def test_unrelated_ability_cannot_change_order_and_legacy_outputs_have_no_new_fields():
    payload = request([book("one", ["scheduling"])])
    original = calculate(payload)
    payload["observations"] = [observation("process", 0, ability="reasoning")]
    assert calculate(payload) == original
    for version in ["concept-learning-v1", "concept-learning-v2"]:
        legacy = deepcopy(payload)
        legacy["modelVersion"] = version
        result = calculate(legacy)
        assert "personalization" not in result["items"][0]
        assert "orderingPolicy" not in result
