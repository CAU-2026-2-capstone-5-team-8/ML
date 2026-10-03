from pathlib import Path

from fastapi.testclient import TestClient

from bookmatch_ml.api import create_app
from bookmatch_ml.config import load_reader_config
from bookmatch_ml.reader.abilities import build_concept_ability_profile
from bookmatch_ml.schemas import Assessment, AssessmentResponse

ROOT = Path(__file__).parents[1]


def test_self_reports_and_missing_operations_do_not_become_verified_ability() -> None:
    common = dict(topic_id="linear-algebra", concept_id="matrix", difficulty="easy")
    assessment = Assessment(
        assessment_id="concept-test",
        topic_id="linear-algebra",
        responses=[
            AssessmentResponse(
                **common,
                question_id="definition",
                question_type="vocabulary",
                cognitive_operation="recognize",
                answer_mode="MULTIPLE_CHOICE",
                measurement_context="prior-knowledge",
                correct=True,
            ),
            AssessmentResponse(
                **common,
                question_id="application",
                question_type="comprehension",
                cognitive_operation="apply",
                answer_mode="MULTIPLE_CHOICE",
                measurement_context="prior-knowledge",
                correct=False,
            ),
            AssessmentResponse(
                **common,
                question_id="self-report",
                question_type="background_knowledge",
                cognitive_operation="apply",
                answer_mode="SELF_REPORT",
                correct=True,
            ),
            AssessmentResponse(
                **common,
                question_id="legacy",
                question_type="comprehension",
                answer_mode="MULTIPLE_CHOICE",
                measurement_context="prior-knowledge",
                correct=True,
            ),
        ],
    )
    result = build_concept_ability_profile(
        assessment, load_reader_config(ROOT / "configs/reader.yaml")
    )
    abilities = {a["ability"]: a for a in result["abilities"]}
    assert abilities["meaning"]["score"] == 1
    assert abilities["application"]["score"] == 0
    assert abilities["application"]["responseCount"] == 1
    assert "reasoning" not in abilities
    assert result["selfReports"][0]["positiveCount"] == 1
    assert result["unclassifiedResponseCount"] == 1


def test_graph_contains_only_reviewed_edges_and_preserves_shared_ids() -> None:
    client = TestClient(create_app())
    graph = client.get("/ml/concepts/linear-algebra").json()
    assert len(graph["nodes"]) == 20
    assert {"id": "matrix", "label": "행렬"} in graph["nodes"]
    assert {"source": "matrix", "target": "eigenvalue"} in graph["edges"]
    assert {"source": "eigenvalue", "target": "eigenvector"} not in graph["edges"]
    assert client.get("/ml/concepts/not-a-topic").status_code == 404
