from fastapi.testclient import TestClient

from bookmatch_ml.api import create_app


def book(identity, concepts):
    return {
        "bookId": identity,
        "coveredConcepts": concepts,
        "sourceArtifactVersion": "synthetic-v1",
        "sourceArtifactHash": "sha256:" + "a" * 64,
    }


def observation(concept, ability, correct):
    return {"conceptId": concept, "ability": ability, "responseCount": 1, "correctCount": correct}


def payload(observations=None):
    return {
        "topicId": "linear-algebra",
        "ability": "application",
        "limit": 5,
        "observations": observations or [],
        "candidateBooks": [
            book("already-known", ["matrix"]),
            book("next-learning", ["eigenvalue"]),
            book("unmapped", []),
        ],
    }


def test_known_book_does_not_always_beat_new_learning_and_unknown_is_explicit():
    client = TestClient(create_app())
    request = payload([observation("matrix", "application", 1)])
    response = client.post("/ml/learning-fit", json=request)
    assert response.status_code == 200, response.text
    result = response.json()
    assert result["modelVersion"] == "concept-learning-v1"
    assert result["unmappedCandidateCount"] == 1
    assert [row["bookId"] for row in result["items"]] == ["next-learning", "already-known"]
    assert result["items"][0]["unmeasuredConceptCount"] == 1
    assert result["items"][1]["reviewOnly"] is True
    assert "score" not in result["items"][0]
    assert result["depthStatus"] == "unverified"


def test_wrong_definition_is_not_wrong_application_and_self_report_is_rejected():
    client = TestClient(create_app())
    request = payload([observation("matrix", "meaning", 0)])
    result = client.post("/ml/learning-fit", json=request).json()
    eigenvalue = next(row for row in result["items"] if row["bookId"] == "next-learning")
    matrix = next(row for row in eigenvalue["foundation"] if row["conceptId"] == "matrix")
    assert matrix["state"] == "unmeasured"
    assert eigenvalue["status"] == "check-first"
    request["observations"][0]["answerMode"] = "SELF_REPORT"
    assert client.post("/ml/learning-fit", json=request).status_code == 422


def test_wrong_application_marks_foundation_gap_and_empty_observations_do_not():
    client = TestClient(create_app())
    wrong = client.post(
        "/ml/learning-fit", json=payload([observation("matrix", "application", 0)])
    ).json()
    assert (
        next(row for row in wrong["items"] if row["bookId"] == "next-learning")["status"]
        == "foundation-gap"
    )
    empty = client.post("/ml/learning-fit", json=payload()).json()
    assert (
        next(row for row in empty["items"] if row["bookId"] == "next-learning")["status"]
        == "check-first"
    )


def test_contract_rejects_unknown_concept_and_duplicate_cell_and_invalid_count():
    client = TestClient(create_app())
    cases = [
        payload([observation("invented", "application", 1)]),
        payload([observation("matrix", "application", 1)] * 2),
        payload([observation("matrix", "application", 2)]),
    ]
    for request in cases:
        assert client.post("/ml/learning-fit", json=request).status_code == 422


def test_unestablished_prerequisites_do_not_establish_readiness():
    client = TestClient(create_app())
    request = payload()
    request["candidateBooks"] = [book("graph-unlinked", ["singular value decomposition"])]
    result = client.post("/ml/learning-fit", json=request).json()["items"][0]
    assert result["foundation"] == []
    assert result["status"] == "check-first"
    assert result["foundationStatus"] == "not-established"
