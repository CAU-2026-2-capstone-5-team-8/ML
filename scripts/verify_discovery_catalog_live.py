"""Record actual catalog pagination and ML calculations against the frozen discovery handoff."""

import argparse
import copy
import json
from pathlib import Path

from verify_linear_algebra_live import Recorder, rank_request, resolve_ml_http


def verify(
    packet: Path,
    assessment_packet: Path,
    output: Path,
    backend: str,
    ml: str,
    user_id: int,
    previous_run: Path,
) -> dict:
    rec = Recorder(output)
    old = json.loads(previous_run.read_text())
    assert rec.request("previous-run", "GET", f"{backend}/api/recommendations/{old['id']}") == old
    summary = rec.request("catalog-summary", "GET", backend + "/api/books/summary")
    assert (
        summary["bookCount"],
        summary["tocBookCount"],
        summary["rankingCandidateCount"],
        summary["conceptBookCount"],
    ) == (471, 439, 83, 11)
    unique = {}
    for page in range(5):
        data = rec.request(
            f"catalog-page-{page}", "GET", f"{backend}/api/books?page={page}&size=100"
        )
        assert data["totalElements"] == 471 and data["totalPages"] == 5
        for book in data["content"]:
            assert book["id"] not in unique
            unique[book["id"]] = book
    assert len(unique) == 471
    assert (
        sum(any((t["tocEntryCount"] or 0) > 0 for t in b["topics"]) for b in unique.values()) == 439
    )
    la = next(t for t in summary["topics"] if t["mlTopicId"] == "linear-algebra")
    for topic in summary["topics"]:
        data = rec.request(
            "topic-" + topic["mlTopicId"],
            "GET",
            f"{backend}/api/books?topicId={topic['id']}&size=100",
        )
        assert data["totalElements"] == topic["bookCount"]
        assert (
            sum(b["topics"][0]["tocEntryCount"] > 0 for b in data["content"])
            == topic["tocBookCount"]
        )
        assert topic["assessmentReady"] == (topic["id"] == la["id"])
    assessment = json.loads((packet / "assessment-request.json").read_text())
    candidates = json.loads((packet / "wire-candidates.json").read_text())
    assert len(candidates) == 83
    profile = rec.request("ml-profile", "POST", ml + "/ml/reader-profile", assessment)
    rank_body = rank_request(profile, candidates)
    assert resolve_ml_http((packet / "ml-live.http").read_text(), profile) == rank_body
    ranked = rec.request("ml-rank", "POST", ml + "/ml/rank", rank_body)
    assert rec.request("ml-rank-repeat", "POST", ml + "/ml/rank", rank_body) == ranked
    assert ranked["diagnostics"]["topicCandidateCount"] == 83 and len(ranked["items"]) == 5
    changed = copy.deepcopy(assessment)
    for answer in changed["responses"]:
        answer["correct"] = False
    low = rec.request("ml-profile-changed", "POST", ml + "/ml/reader-profile", changed)
    assert low["conceptReadiness"] != profile["conceptReadiness"]
    rec.request("ml-rank-changed", "POST", ml + "/ml/rank", rank_request(low, candidates))
    unavailable = copy.deepcopy(rank_body)
    unavailable["readerProfile"]["conceptReadiness"] = []
    assert rec.request("ml-rank-unassessed", "POST", ml + "/ml/rank", unavailable)["items"] == []
    invalid = copy.deepcopy(rank_body)
    invalid["candidateBooks"].append(invalid["candidateBooks"][0])
    rec.request("ml-duplicate-candidate", "POST", ml + "/ml/rank", invalid, expected=422)
    correct = json.loads((assessment_packet / "generated.json").read_text())["correct_choice_index"]
    results = []
    for name, known in [("known", True), ("unfamiliar", False)]:
        session = rec.request(
            "start-" + name,
            "POST",
            backend + "/api/assessments",
            {"userId": user_id, "topicId": la["id"]},
            expected=201,
        )
        assert len(session["questions"]) == 9
        session_id = session["id"]
        for index, question in enumerate(session["questions"]):
            assert "correctChoiceIndex" not in question and "explanation" not in question
            answer = (
                {"knowsConcept": known}
                if question["answerMode"] == "SELF_REPORT"
                else {"selectedChoiceIndex": correct if known else (correct + 1) % 4}
            )
            rec.request(
                f"{name}-answer-{index}",
                "PUT",
                f"{backend}/api/assessments/{session_id}/answers/{question['id']}",
                answer,
            )
        completed = rec.request(
            "complete-" + name, "POST", f"{backend}/api/assessments/{session_id}/complete"
        )
        assert (
            rec.request(
                "complete-repeat-" + name,
                "POST",
                f"{backend}/api/assessments/{session_id}/complete",
            )
            == completed
        )
        assert completed["profile"]["calculationVersion"] == "reader-v1"
        body = {"userId": user_id, "topicId": la["id"], "challengeLevel": "BALANCED", "topK": 5}
        headers = {"Idempotency-Key": f"discovery-live-session-{session_id}"}
        recommendation = rec.request(
            "recommend-" + name,
            "POST",
            backend + "/api/recommendations",
            body,
            expected=201,
            headers=headers,
        )
        assert recommendation["diagnostics"]["topicCandidateCount"] == 83
        assert len(recommendation["items"]) == 5
        candidate_ids = {c["bookId"] for c in candidates}
        assert all(book["mlBookId"] in candidate_ids for book in recommendation["items"])
        assert (
            rec.request(
                "recommend-repeat-" + name,
                "POST",
                backend + "/api/recommendations",
                body,
                headers=headers,
            )
            == recommendation
        )
        assert (
            rec.request(
                "readback-" + name, "GET", f"{backend}/api/recommendations/{recommendation['id']}"
            )
            == recommendation
        )
        rec.request(
            "completed-answer-locked-" + name,
            "PUT",
            f"{backend}/api/assessments/{session_id}/answers/{session['questions'][0]['id']}",
            {"knowsConcept": not known},
            expected=409,
        )
        results.append(
            {
                "sessionId": session_id,
                "profile": completed["profile"],
                "runId": recommendation["id"],
                "diagnostics": recommendation["diagnostics"],
            }
        )
    assert (
        results[0]["profile"]["evidence"]["conceptReadiness"]
        != results[1]["profile"]["evidence"]["conceptReadiness"]
    )
    assert (
        rec.request("previous-run-after", "GET", f"{backend}/api/recommendations/{old['id']}")
        == old
    )
    result = {
        "catalog": summary,
        "backendRuns": results,
        "mlRepeatIdentical": True,
        "unassessedItems": 0,
        "previousRunPreserved": old["id"],
    }
    (output / "summary.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--packet-dir", type=Path, required=True)
    parser.add_argument("--assessment-packet-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--backend-base", required=True)
    parser.add_argument("--ml-base", required=True)
    parser.add_argument("--user-id", type=int, required=True)
    parser.add_argument("--previous-run-json", type=Path, required=True)
    args = parser.parse_args()
    result = verify(
        args.packet_dir,
        args.assessment_packet_dir,
        args.output_dir,
        args.backend_base.rstrip("/"),
        args.ml_base.rstrip("/"),
        args.user_id,
        args.previous_run_json,
    )
    print(
        json.dumps(
            {
                "books": result["catalog"]["bookCount"],
                "tocBooks": result["catalog"]["tocBookCount"],
                "candidateCount": 83,
                "runIds": [r["runId"] for r in result["backendRuns"]],
                "previousRunPreserved": result["previousRunPreserved"],
            }
        )
    )


if __name__ == "__main__":
    main()
