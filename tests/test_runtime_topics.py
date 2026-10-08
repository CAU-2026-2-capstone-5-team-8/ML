import hashlib
import json
from pathlib import Path

import pytest

from bookmatch_ml.concept_v2.learning_fit import LearningFitRequest, recommend_learning
from bookmatch_ml.concept_v2.presentation import concept_graph
from bookmatch_ml.config import load_ranking_v2_config
from bookmatch_ml.runtime_topics import resolve_runtime_topic


@pytest.fixture
def registered(tmp_path):
    registry = tmp_path / "runtime-topics"
    registry.mkdir()
    topic = "field-microeconomics"
    base = load_ranking_v2_config(Path(__file__).parents[1] / "configs/ranking_v2.yaml")
    payload = {
        "contract_version": "topic-runtime-graph-v1",
        "topic_id": topic,
        "source_snapshot_id": "books-test",
        "content_report_hash": "sha256:" + "a" * 64,
        "review_report_hash": "sha256:" + "b" * 64,
        "reviewer_type": "ai",
        "nodes": {topic: [topic + ":supply", topic + ":equilibrium"]},
        "labels": {topic + ":supply": "공급", topic + ":equilibrium": "균형"},
        "accepted_edges": [
            {"topic": topic, "prerequisite": topic + ":supply", "dependent": topic + ":equilibrium"}
        ],
        "concept_graph_version": "concept-graph-v1",
        "concept_graph_hash": "sha256:" + "c" * 64,
        "graph_review_version": "ai-concept-graph-review-v1",
        "graph_review_hash": "sha256:" + "d" * 64,
    }
    graph = tmp_path / "artifacts/runtime-graph.json"
    graph.parent.mkdir()
    graph.write_text(json.dumps(payload))
    pointer = {
        "topicId": topic,
        "graphPath": str(graph),
        "graphHash": "sha256:" + hashlib.sha256(graph.read_bytes()).hexdigest(),
    }
    (registry / (topic + ".json")).write_text(json.dumps(pointer))
    return registry, topic, base, graph, payload


def test_new_topics_resolve_without_server_restart_and_preserve_production_graph(registered):
    registry, topic, base, _, _ = registered
    dynamic = resolve_runtime_topic(registry, topic, base)
    assert concept_graph(dynamic, topic)["reviewerType"] == "ai"
    assert resolve_runtime_topic(registry, "linear-algebra", base) is base
    request = LearningFitRequest.model_validate(
        {
            "modelVersion": "concept-learning-v2",
            "topicId": topic,
            "ability": "meaning",
            "observations": [
                {
                    "conceptId": topic + ":supply",
                    "ability": "meaning",
                    "responseCount": 1,
                    "correctCount": 1,
                }
            ],
            "candidateBooks": [
                {
                    "bookId": "test-book",
                    "coveredConcepts": [topic + ":equilibrium"],
                    "sourceArtifactVersion": "books-test",
                    "sourceArtifactHash": "sha256:" + "c" * 64,
                }
            ],
        }
    )
    result = recommend_learning(request, dynamic)
    assert (
        result["modelVersion"] == "concept-learning-v2"
        and result["items"][0]["bookId"] == "test-book"
    )


@pytest.mark.parametrize("failure", ["tampered", "escape", "wrong-topic", "cycle"])
def test_invalid_registry_fails_closed(registered, failure, tmp_path):
    registry, topic, base, graph, payload = registered
    if failure == "tampered":
        graph.write_text("{}")
    elif failure == "escape":
        (registry / (topic + ".json")).write_text(
            json.dumps(
                {"topicId": topic, "graphPath": "/etc/passwd", "graphHash": "sha256:" + "a" * 64}
            )
        )
    else:
        if failure == "wrong-topic":
            payload["topic_id"] = "other-topic"
        else:
            payload["accepted_edges"].append(
                {
                    "topic": topic,
                    "prerequisite": topic + ":equilibrium",
                    "dependent": topic + ":supply",
                }
            )
        graph.write_text(json.dumps(payload))
        pointer = json.loads((registry / (topic + ".json")).read_text())
        pointer["graphHash"] = "sha256:" + hashlib.sha256(graph.read_bytes()).hexdigest()
        (registry / (topic + ".json")).write_text(json.dumps(pointer))
    with pytest.raises((ValueError, FileNotFoundError)):
        resolve_runtime_topic(registry, topic, base)
