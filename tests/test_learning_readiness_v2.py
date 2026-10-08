"""Regression tests for observable prerequisite checks, not calibrated mastery."""

from copy import deepcopy

from fastapi.testclient import TestClient

from bookmatch_ml.api import create_app

CLIENT = TestClient(create_app())
HASH = "sha256:" + "a" * 64


def request(observations=None, concepts=None):
    return {
        "topicId": "linear-algebra",
        "ability": "application",
        "modelVersion": "concept-learning-v2",
        "observations": observations or [],
        "candidateBooks": [
            {
                "bookId": "fixture",
                "coveredConcepts": concepts or ["matrix", "eigenvalue"],
                "sourceArtifactVersion": "fixture-v1",
                "sourceArtifactHash": HASH,
            }
        ],
        "limit": 5,
    }


def observation(concept, correct, count=1):
    return {
        "conceptId": concept,
        "ability": "application",
        "responseCount": count,
        "correctCount": correct,
    }


def calculate(payload):
    response = CLIENT.post("/ml/learning-fit", json=payload)
    assert response.status_code == 200, response.text
    return response.json()


def test_covered_prerequisite_does_not_hide_mistakes_and_preserves_counts():
    item = calculate(request([observation("matrix", 1, 2)]))["items"][0]
    assert item["status"] == "foundation-gap"
    assert item["inferredPrerequisites"] == ["matrix"]
    assert item["internalPrerequisites"] == ["matrix"]
    assert item["externalPrerequisites"] == []
    rows = item["readingChecklist"]["concepts"]
    assert [r["conceptId"] for r in rows] == ["matrix", "eigenvalue"]
    assert rows[0]["responseCount"] == 2
    assert rows[0]["correctCount"] == 1
    assert rows[0]["nextAction"] == "review-concept"
    assert rows[1]["state"] == "unmeasured"
    assert rows[1]["responseCount"] is None
    assert rows[1]["nextAction"] == "assess-concept"
    assert rows[1]["dependsOn"] == ["matrix"]
    assert rows[0]["requiredFor"] == ["eigenvalue"]


def test_unknown_and_more_correct_answers_never_change_book_facts():
    outcomes = [
        calculate(request(obs))["items"][0]
        for obs in ([], [observation("matrix", 0)], [observation("matrix", 1)])
    ]
    assert [r["status"] for r in outcomes] == ["check-first", "foundation-gap", "ready-to-explore"]
    for key in (
        "coveredConcepts",
        "inferredPrerequisites",
        "internalPrerequisites",
        "externalPrerequisites",
    ):
        assert outcomes[0][key] == outcomes[1][key] == outcomes[2][key]
    assert outcomes[2]["readingChecklist"]["concepts"][0]["state"] == "correct"
    assert (
        outcomes[2]["readingChecklist"]["interpretation"]
        == "observed_answers_not_calibrated_mastery"
    )


def test_v1_is_unchanged_and_explicit_v1_equals_default():
    payload = request([observation("matrix", 0)])
    payload.pop("modelVersion")
    omitted = calculate(payload)
    payload["modelVersion"] = "concept-learning-v1"
    assert calculate(payload) == omitted
    assert omitted["items"][0]["inferredPrerequisites"] == []
    assert "readingChecklist" not in omitted["items"][0]


def test_readiness_first_order_does_not_promote_unknown_books_over_ready_review():
    payload = request([observation("matrix", 1), observation("eigenvalue", 1)])
    other = deepcopy(payload["candidateBooks"][0])
    other.update(bookId="unchecked", coveredConcepts=["basis"])
    payload["candidateBooks"].append(other)
    result = calculate(payload)
    assert [i["bookId"] for i in result["items"]] == ["fixture", "unchecked"]
    payload["candidateBooks"].reverse()
    assert calculate(payload) == result


def support(concept="matrix"):
    return {
        "conceptId": concept,
        "evidenceId": "evidence_fixture",
        "sourceId": "source_fixture",
        "sourceUrl": "https://example.org/book",
        "evidenceType": "toc_same_work",
        "editionRelation": "same_work",
        "tocPath": ["1. Matrices"],
        "matchingAlias": "Matrices",
        "matchMethod": "normalized_alias_span_v2",
        "provenanceHash": HASH,
    }


def test_exact_duplicate_source_evidence_does_not_change_result_and_is_not_teaching():
    payload = request([observation("matrix", 0)])
    payload["candidateBooks"][0]["conceptEvidence"] = [support()]
    first = calculate(payload)
    payload["candidateBooks"][0]["conceptEvidence"].append(support())
    assert calculate(payload) == first
    row = first["items"][0]["readingChecklist"]["concepts"][0]
    assert row["evidence"] == [support()]
    assert row["teachingSufficiency"] == "unverified"
    assert first["items"][0]["status"] == "foundation-gap"


def test_provenance_rejects_foreign_concepts_unsafe_urls_and_conflicting_rows():
    for evidence in (
        [support("basis")],
        [dict(support(), sourceUrl="javascript:alert(1)")],
        [support(), dict(support(), tocPath=["different title"])],
    ):
        payload = request()
        payload["candidateBooks"][0]["conceptEvidence"] = evidence
        assert CLIENT.post("/ml/learning-fit", json=payload).status_code == 422
