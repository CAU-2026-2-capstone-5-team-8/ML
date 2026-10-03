"""Record real HTTP calculations. Answers are synthetic; no scores are invented."""

import argparse
import copy
import json
import re
from pathlib import Path

import httpx

from bookmatch_ml.integration.schemas import MatchingReaderProfileDto


def matching_profile(profile: dict) -> dict:
    keys = [
        "userId",
        "topicId",
        "vocabulary",
        "backgroundKnowledge",
        "comprehension",
        "profileVersion",
        "configVersion",
        "configHash",
    ]
    result = {key: profile[key] for key in keys}
    result["conceptReadiness"] = [
        {"conceptId": row["conceptId"], "score": row["score"]}
        for row in profile["conceptReadiness"]
    ]
    return MatchingReaderProfileDto.model_validate(result).model_dump(mode="json", by_alias=True)


def rank_request(profile: dict, candidates: list) -> dict:
    return {
        "rankingModel": "rank-prerequisite-first-v2",
        "readerProfile": matching_profile(profile),
        "candidateBooks": candidates,
        "limit": 5,
    }


def resolve_ml_http(text: str, profile: dict) -> dict:
    """Resolve the fixed .http's response references against the actual API response."""
    body = text.split("# @name laRank\n", 1)[1].split("\n\n", 1)[1]

    def value(match: re.Match) -> str:
        path = match.group(1)
        row = re.fullmatch(r"conceptReadiness\[(\d+)\]\.(conceptId|score)", path)
        item = profile["conceptReadiness"][int(row[1])][row[2]] if row else profile[path]
        return str(item)

    return json.loads(re.sub(r"\{\{laProfile.response.body.\$\.([^}]+)\}\}", value, body))


class Recorder:
    def __init__(self, output: Path):
        output.mkdir(parents=True, exist_ok=False)
        self.output = output
        self.client = httpx.Client(timeout=20, trust_env=False)

    def request(self, name: str, method: str, url: str, body=None, expected=200, headers=None):
        response = self.client.request(method, url, json=body, headers=headers)
        data = response.json()
        artifact = {
            "method": method,
            "url": url,
            "requestHeaders": headers or {},
            "request": body,
            "status": response.status_code,
            "response": data,
        }
        (self.output / (name + ".json")).write_text(
            json.dumps(artifact, ensure_ascii=False, indent=2) + "\n"
        )
        assert response.status_code == expected, (name, response.status_code, data)
        return data


def verify_ml(rec: Recorder, packet: Path, base: str) -> dict:
    assessment = json.loads((packet / "assessment-request.json").read_text())
    candidates = json.loads((packet / "wire-candidates.json").read_text())
    profile = rec.request("ml-profile", "POST", base + "/ml/reader-profile", assessment)
    assert profile == rec.request(
        "ml-profile-repeat", "POST", base + "/ml/reader-profile", assessment
    )
    rank = rank_request(profile, candidates)
    # This is also a regression check of every response binding in the REST Client example.
    assert resolve_ml_http((packet / "ml-live.http").read_text(), profile) == rank
    ranked = rec.request("ml-rank", "POST", base + "/ml/rank", rank)
    assert ranked == rec.request("ml-rank-repeat", "POST", base + "/ml/rank", rank)
    assert len(ranked["items"]) == 3
    assert ranked["diagnostics"]["personalizedCandidateShortage"] == 2
    changed = copy.deepcopy(assessment)
    for row in changed["responses"]:
        row["correct"] = False
    low = rec.request("ml-profile-different-answers", "POST", base + "/ml/reader-profile", changed)
    assert low["conceptReadiness"] != profile["conceptReadiness"]
    assert low["comprehension"] != profile["comprehension"]
    low_rank = rank_request(low, candidates)
    assert low_rank["readerProfile"] != rank["readerProfile"]
    rec.request("ml-rank-different-answers", "POST", base + "/ml/rank", low_rank)
    missing = copy.deepcopy(rank)
    missing["readerProfile"]["conceptReadiness"] = []
    unavailable = rec.request("ml-rank-unassessed", "POST", base + "/ml/rank", missing)
    assert unavailable["items"] == []
    assert unavailable["diagnostics"]["personalizedCandidateShortage"] == 5
    missing["bookId"] = candidates[0]["bookId"]
    rec.request("ml-rank-unassessed-target", "POST", base + "/ml/rank", missing, expected=422)
    invalid = copy.deepcopy(rank)
    invalid["readerProfile"]["conceptReadiness"][0]["score"] = 2
    rec.request("ml-invalid-score", "POST", base + "/ml/rank", invalid, expected=422)
    duplicate = copy.deepcopy(rank)
    duplicate["candidateBooks"].append(duplicate["candidateBooks"][0])
    rec.request("ml-duplicate-candidate", "POST", base + "/ml/rank", duplicate, expected=422)
    return {
        "bookCount": len(ranked["items"]),
        "shortage": 2,
        "repeatIdentical": True,
        "differentAnswersChangeProfileAndRequest": True,
        "unassessedCount": 0,
        "invalidStatuses": [422, 422, 422],
    }


def verify_backend(rec: Recorder, packet: Path, base: str, user_id: int, topic_id: int) -> dict:
    expected_ids = set(json.loads((packet / "selection.json").read_text())["selected_book_ids"])
    correct_index = json.loads((packet / "generated.json").read_text())["correct_choice_index"]
    results = []
    for name, knows in [("known", True), ("unfamiliar", False)]:
        session = rec.request(
            "backend-start-" + name,
            "POST",
            base + "/api/assessments",
            {"userId": user_id, "topicId": topic_id},
            expected=201,
        )
        assert len(session["questions"]) == 9
        assert all(
            "correctChoiceIndex" not in row and "explanation" not in row
            for row in session["questions"]
        )
        session_id = session["id"]
        for i, q in enumerate(session["questions"]):
            answer = (
                {"knowsConcept": knows}
                if q["answerMode"] == "SELF_REPORT"
                else {"selectedChoiceIndex": correct_index if knows else (correct_index + 1) % 4}
            )
            rec.request(
                f"backend-{name}-answer-{i}",
                "PUT",
                f"{base}/api/assessments/{session_id}/answers/{q['id']}",
                answer,
            )
        done = rec.request(
            "backend-complete-" + name, "POST", f"{base}/api/assessments/{session_id}/complete"
        )
        assert done == rec.request(
            "backend-complete-repeat-" + name,
            "POST",
            f"{base}/api/assessments/{session_id}/complete",
        )
        assert done["profile"]["calculationVersion"] == "reader-v1"
        assert done["profile"]["evidence"]["method"] == "ml-http-reader-profile"
        headers = {"Idempotency-Key": f"la-live-verification-session-{session_id}"}
        request = {"userId": user_id, "topicId": topic_id, "challengeLevel": "BALANCED", "topK": 5}
        ranked = rec.request(
            "backend-recommend-" + name,
            "POST",
            base + "/api/recommendations",
            request,
            expected=201,
            headers=headers,
        )
        assert ranked["profileId"] == done["profile"]["id"]
        assert ranked["modelVersion"] == "rank-prerequisite-first-v2"
        assert ranked["status"] == "SUCCEEDED"
        assert {item["mlBookId"] for item in ranked["items"]} == expected_ids
        assert len(ranked["items"]) == 3
        assert ranked == rec.request(
            "backend-recommend-repeat-" + name,
            "POST",
            base + "/api/recommendations",
            request,
            headers=headers,
        )
        assert ranked == rec.request(
            "backend-readback-" + name, "GET", f"{base}/api/recommendations/{ranked['id']}"
        )
        invalid = {**request, "topK": 4}
        rec.request(
            "backend-idempotency-conflict-" + name,
            "POST",
            base + "/api/recommendations",
            invalid,
            expected=409,
            headers=headers,
        )
        rec.request(
            "backend-completed-answer-rejected-" + name,
            "PUT",
            f"{base}/api/assessments/{session_id}/answers/{session['questions'][0]['id']}",
            {"knowsConcept": False},
            expected=409,
        )
        results.append(
            {
                "sessionId": session_id,
                "profileId": done["profile"]["id"],
                "runId": ranked["id"],
                "profile": done["profile"],
            }
        )
    assert results[0]["profile"]["vocabulary"] != results[1]["profile"]["vocabulary"]
    assert results[0]["profile"]["comprehension"] != results[1]["profile"]["comprehension"]
    return {
        "sessions": [{k: v for k, v in r.items() if k != "profile"} for r in results],
        "mode": "http",
        "books": 3,
        "replayAndReadbackIdentical": True,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--packet-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--ml-url", default="http://127.0.0.1:8011")
    parser.add_argument("--backend-url")
    parser.add_argument("--user-id", type=int)
    parser.add_argument("--topic-id", type=int)
    args = parser.parse_args()
    if args.backend_url and (not args.user_id or not args.topic_id):
        parser.error("backend verification requires explicit bootstrap user-id and topic-id")
    rec = Recorder(args.output_dir)
    summary = {"ml": verify_ml(rec, args.packet_dir, args.ml_url.rstrip("/"))}
    if args.backend_url:
        summary["backend"] = verify_backend(
            rec, args.packet_dir, args.backend_url.rstrip("/"), args.user_id, args.topic_id
        )
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
