"""Prepare an immutable, explicit local handoff. Does not collect or change source data."""

import argparse
import hashlib
import json
from pathlib import Path

from bookmatch_ml.data.loader import load_canonical_dataset
from bookmatch_ml.integration.schemas import BookCandidateDto
from bookmatch_ml.ranking.loader import load_book_profiles
from bookmatch_ml.ranking.source_aware_adapter import (
    build_source_aware_matching_book_profiles,
    load_concept_mapping_report,
)
from bookmatch_ml.schemas import MatchingBookProfile

ROOT = Path(__file__).resolve().parents[1]
PIN = ROOT / "configs/experiments/linear-algebra-live-handoff-v1.json"
SELF_REPORTS = [
    ("matrix", "VOCABULARY", "행렬의 행과 열이라는 용어의 뜻을 알고 있나요?"),
    ("vector", "VOCABULARY", "벡터의 성분이라는 용어의 뜻을 알고 있나요?"),
    ("linear system", "VOCABULARY", "선형 연립방정식이라는 용어의 뜻을 알고 있나요?"),
    (
        "systems of equations",
        "BACKGROUND_KNOWLEDGE",
        "연립방정식의 해가 무엇을 뜻하는지 알고 있나요?",
    ),
    (
        "high school algebra",
        "BACKGROUND_KNOWLEDGE",
        "일차방정식의 이항과 식의 정리를 할 수 있나요?",
    ),
    (
        "vector space",
        "BACKGROUND_KNOWLEDGE",
        "벡터공간이 덧셈과 스칼라 곱에 닫혀 있다는 의미를 알고 있나요?",
    ),
    ("basis", "COMPREHENSION", "기저가 벡터공간의 모든 벡터를 표현한다는 의미를 설명할 수 있나요?"),
    ("eigenvalue", "COMPREHENSION", "고유값과 고유벡터의 관계를 설명할 수 있나요?"),
]


def digest(data: bytes) -> str:
    return "sha256:" + hashlib.sha256(data).hexdigest()


def checked(path: Path, expected: str) -> bytes:
    if not path.is_file():
        raise ValueError(f"missing snapshot file: {path}; obtain the pinned offline snapshot")
    raw = path.read_bytes()
    if digest(raw) != expected:
        raise ValueError(
            f"snapshot SHA-256 mismatch: {path}; original inputs are never overwritten"
        )
    return raw


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def build_ml_http(assessment: dict, wire_candidates: list) -> str:
    concept_rows = sorted({r["conceptId"] for r in assessment["responses"]})
    reader = {
        key: f"{{{{laProfile.response.body.$.{key}}}}}"
        for key in [
            "userId",
            "topicId",
            "vocabulary",
            "backgroundKnowledge",
            "comprehension",
            "profileVersion",
            "configVersion",
            "configHash",
        ]
    }
    reader["conceptReadiness"] = [
        {
            "conceptId": f"{{{{laProfile.response.body.$.conceptReadiness[{i}].conceptId}}}}",
            "score": f"{{{{laProfile.response.body.$.conceptReadiness[{i}].score}}}}",
        }
        for i in range(len(concept_rows))
    ]
    rank = json.dumps(
        {
            "rankingModel": "rank-prerequisite-first-v2",
            "readerProfile": reader,
            "candidateBooks": wire_candidates,
            "limit": 5,
        },
        ensure_ascii=False,
        indent=2,
    )
    for key in ["userId", "vocabulary", "backgroundKnowledge", "comprehension"]:
        rank = rank.replace('"' + reader[key] + '"', reader[key])
    for row in reader["conceptReadiness"]:
        rank = rank.replace('"' + row["score"] + '"', row["score"])
    http = (
        "# Synthetic answers; scores/readiness/config come from the preceding real response.\n"
        "# Run laProfile first. This fixed request produces eight sorted conceptReadiness rows.\n"
        "@baseUrl = http://127.0.0.1:8011\n\n###\n# @name laProfile\n"
        "POST {{baseUrl}}/ml/reader-profile\nContent-Type: application/json\n\n"
        + json.dumps(assessment, ensure_ascii=False, indent=2)
        + "\n\n###\n# @name laRank\nPOST {{baseUrl}}/ml/rank\nContent-Type: application/json\n\n"
        + rank
        + "\n"
    )
    return http


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--canonical-dir", type=Path, required=True)
    parser.add_argument("--pilot-dir", type=Path, required=True)
    parser.add_argument("--generated-question", type=Path, required=True)
    parser.add_argument("--reviews", type=Path, required=True)
    parser.add_argument("--grounding", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    pin = json.loads(PIN.read_text())
    if args.output_dir.exists():
        parser.error("output-dir exists; reuse its manifest or choose a new directory")
    for name, expected in pin["canonical_hashes"].items():
        checked(args.canonical_dir / name, expected)
    for name, expected in pin["pilot_hashes"].items():
        checked(args.pilot_dir / name, expected)
    canonical = load_canonical_dataset(args.canonical_dir)
    mapping = load_concept_mapping_report(args.pilot_dir / "concept-mapping.json")
    rebuilt = build_source_aware_matching_book_profiles(
        mapping, load_book_profiles(args.pilot_dir / "book-profiles.jsonl")
    )
    actual = [
        MatchingBookProfile.model_validate_json(line)
        for line in (args.pilot_dir / "matching-candidates.jsonl").read_text().splitlines()
    ]
    if actual != rebuilt:
        raise ValueError("pilot candidates differ from the source-aware mapping/profile join")
    books = {book.book_id: book for book in canonical.books}
    candidates = {c.book_id: c for c in actual}
    mapped = {book.book_id: book for book in mapping.books}
    selected = pin["selected_book_ids"]
    audit = []
    for book_id in selected:
        book, candidate, concept_book = books[book_id], candidates[book_id], mapped[book_id]
        if any(word in book.title.lower() for word in ["편입", "문제집", "해답", "work book"]):
            raise ValueError(f"selection contains a workbook: {book_id}")
        if not candidate.covered_concepts or not all(
            any(e.toc_path for e in concept.supporting_evidence)
            for concept in concept_book.concepts
        ):
            raise ValueError(f"selection lacks TOC-backed concept presence: {book_id}")
        audit.append(concept_book.model_dump(mode="json"))
    question_inputs = {
        "generated.json": args.generated_question,
        "reviews.jsonl": args.reviews,
        "grounding.json": args.grounding,
    }
    question_bytes = {
        name: checked(path, pin["approved_hashes"][name]) for name, path in question_inputs.items()
    }
    generated = json.loads(question_bytes["generated.json"])
    reviews = [json.loads(line) for line in question_bytes["reviews.jsonl"].splitlines()]
    review = [
        r for r in reviews if r["generated_question_id"] == generated["generated_question_id"]
    ]
    if len(review) != 1 or review[0]["status"] != "approve" or review[0]["correct"] is not True:
        raise ValueError("exact approved/correct review required")
    args.output_dir.mkdir(parents=True)
    # Minimized copies stay in ignored data/output; full prose is never put in .http or metadata.
    for name, raw in question_bytes.items():
        (args.output_dir / name).write_bytes(raw)
    for filename, objects in [
        ("books.jsonl", [books[key].model_dump(mode="json") for key in selected]),
        ("candidates.jsonl", [candidates[key].model_dump(mode="json") for key in selected]),
    ]:
        (args.output_dir / filename).write_text(
            "".join(json.dumps(obj, ensure_ascii=False, sort_keys=True) + "\n" for obj in objects)
        )
    write_json(args.output_dir / "selection.json", {**pin, "toc_support": audit})
    write_json(
        args.output_dir / "catalog-manifest.json",
        {
            "contract_version": "local-catalog-import-v1",
            "snapshot_id": pin["snapshot_id"],
            "books_path": "books.jsonl",
            "books_sha256": digest((args.output_dir / "books.jsonl").read_bytes()),
            "candidates_path": "candidates.jsonl",
            "candidates_sha256": digest((args.output_dir / "candidates.jsonl").read_bytes()),
            "selected_book_ids": selected,
            "topic": {
                "code": "LA",
                "name": "선형대수",
                "parent_code": "MAT",
                "parent_name": "수학",
                "ml_topic_id": "linear-algebra",
            },
        },
    )
    self_reports = [
        {
            "demo_key": f"la-local-validation-v1-{index}",
            "concept_id": concept,
            "measurement_area": area,
            "difficulty": 2,
            "prompt": "[자기평가] " + prompt,
            "version": "local-self-report-validation-v1",
        }
        for index, (concept, area, prompt) in enumerate(SELF_REPORTS, 1)
    ]
    write_json(
        args.output_dir / "assessment-manifest.json",
        {
            "contract_version": "local-assessment-bootstrap-v1",
            "ml_topic_id": "linear-algebra",
            "user_key": "la-live-validation-user-v1",
            "user_name": "선형대수 연동 검증 사용자",
            "self_report_questions": self_reports,
            "approved_questions": [
                {
                    "generated_path": "generated.json",
                    "generated_sha256": pin["approved_hashes"]["generated.json"],
                    "reviews_path": "reviews.jsonl",
                    "reviews_sha256": pin["approved_hashes"]["reviews.jsonl"],
                    "grounding_path": "grounding.json",
                    "grounding_sha256": pin["approved_hashes"]["grounding.json"],
                }
            ],
        },
    )
    responses = [
        {
            "questionId": q["demo_key"],
            "topicId": "linear-algebra",
            "conceptId": q["concept_id"],
            "questionType": q["measurement_area"].lower(),
            "difficulty": "easy",
            "correct": True,
        }
        for q in self_reports
    ]
    responses.append(
        {
            "questionId": generated["generated_question_id"],
            "topicId": "linear-algebra",
            "conceptId": "matrix",
            "questionType": "comprehension",
            "difficulty": "easy",
            "correct": True,
        }
    )
    assessment = {
        "userId": 1,
        "assessmentId": "synthetic-la-live-validation-v1",
        "topicId": "linear-algebra",
        "responses": responses,
    }
    write_json(args.output_dir / "assessment-request.json", assessment)
    wire_candidates = [
        BookCandidateDto(**candidates[key].model_dump()).model_dump(mode="json", by_alias=True)
        for key in selected
    ]
    write_json(args.output_dir / "wire-candidates.json", wire_candidates)
    (args.output_dir / "ml-live.http").write_text(build_ml_http(assessment, wire_candidates))
    print(
        f"Prepared {len(selected)} real books, 8 validation self-reports "
        f"and 1 approved v4 question: {args.output_dir}"
    )


if __name__ == "__main__":
    main()
