from pathlib import Path

import pytest
from test_concept_v2 import REAL_GRAPH, entry, profile, reader

from bookmatch_ml.concept_v2.difficulty import load_difficulty_policy, score_difficulty

POLICY_PATH = Path(__file__).parents[1] / "configs/concept_difficulty.yaml"


def assess(titles, mastery):
    book = profile([entry(str(i), title, None, 1, i) for i, title in enumerate(titles)], REAL_GRAPH)
    return score_difficulty(reader(mastery), book, load_difficulty_policy(POLICY_PATH))


def test_expertise_reduces_burden_without_changing_book_level():
    titles = ["Processes", "Threads", "Concurrency", "Synchronization", "Deadlock"]
    concepts = ["programming", "process", "thread", "concurrency", "synchronization", "deadlock"]
    novice = assess(titles, dict.fromkeys(concepts, 0.0))
    expert = assess(titles, dict.fromkeys(concepts, 1.0))
    assert novice["book_level_score"] == expert["book_level_score"]
    assert novice["burden_lower"] > expert["burden_upper"]
    assert novice["difficulty_band"] == "challenging"
    assert expert["difficulty_band"] == "easy"


def test_teaching_prerequisite_before_use_reduces_external_gap():
    mastery = {"programming": 1.0, "process": 0.0, "thread": 0.0}
    taught = assess(["Processes", "Threads"], mastery)
    assumed = assess(["Threads", "Processes"], mastery)
    assert taught["prerequisite_gap_lower"] < assumed["prerequisite_gap_lower"]


def test_unassessed_mastery_is_an_interval_not_a_fabricated_point():
    result = assess(["Processes"], {})
    assert result["burden_lower"] < result["burden_upper"]
    assert result["difficulty_band"] == "assessment_needed"
    assert result["next_assessment_concepts"]


def test_unmapped_titles_do_not_receive_a_perfect_fit():
    result = assess(["Power system operation"], {})
    assert result["book_level_score"] is None
    assert result["recommendation_score"] is None


def test_different_topic_is_rejected():
    book = profile([entry("1", "Processes", None, 1, 0)], REAL_GRAPH)
    wrong = reader({}).model_copy(update={"topic_id": "linear-algebra"})
    with pytest.raises(ValueError, match="topic"):
        score_difficulty(wrong, book, load_difficulty_policy(POLICY_PATH))


def test_duplicate_headings_do_not_raise_intrinsic_level():
    one = assess(["Processes"], {"programming": 1.0, "process": 0.0})
    repeated = assess(["Processes"] * 4, {"programming": 1.0, "process": 0.0})
    assert one["book_level_score"] == repeated["book_level_score"]


def test_more_advanced_concepts_increase_intrinsic_level():
    basic = assess(["Processes"], {})
    advanced = assess(["Deadlock", "Distributed systems"], {})
    assert advanced["book_level_score"] > basic["book_level_score"]


def test_learning_opportunity_is_preferred_to_already_mastered_material():
    titles = ["Processes", "Threads", "Concurrency", "Synchronization", "Deadlock"]
    concepts = ["programming", "process", "thread", "concurrency", "synchronization", "deadlock"]
    learning = assess(titles, dict.fromkeys(concepts, 0.5))
    mastered = assess(titles, dict.fromkeys(concepts, 1.0))
    assert learning["recommendation_score"] > mastered["recommendation_score"]


def test_all_graph_nodes_have_explicit_rubric_levels():
    policy, _ = load_difficulty_policy(POLICY_PATH)
    for topic, nodes in REAL_GRAPH.graph.nodes.items():
        assert set(policy.levels[topic]) == set(nodes)


def test_lower_mastery_never_reduces_burden():
    titles = ["Processes", "Threads"]
    burdens = [
        assess(titles, dict.fromkeys(["programming", "process", "thread"], score))["burden_lower"]
        for score in [0, 0.25, 0.5, 0.75, 1]
    ]
    assert burdens == sorted(burdens, reverse=True)
