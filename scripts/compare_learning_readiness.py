"""Reproduce policy changes on committed book facts, never infer real reader ability."""

import argparse
import hashlib
import json
from pathlib import Path

from bookmatch_ml.concept_v2.learning_fit import LearningFitRequest, recommend_learning
from bookmatch_ml.concept_v2.presentation import concept_graph
from bookmatch_ml.config import load_ranking_v2_config


def compare(snapshot: Path) -> dict:
    raw = snapshot.read_bytes()
    data = json.loads(raw)
    digest = "sha256:" + hashlib.sha256(raw).hexdigest()
    config = load_ranking_v2_config(Path("configs/ranking_v2.yaml"))
    report = {
        "snapshotVersion": data["snapshot_version"],
        "snapshotHash": digest,
        "bookCount": len(data["books"]),
        "readerEvidence": "synthetic_scenarios_not_real_assessments",
        "accuracyMeasured": False,
        "limitations": [
            "Historical committed 50-book snapshot, not the current live catalog.",
            "Stored fractional reader scores are deliberately NOT converted to answers.",
            "Unknown graph concepts are explicitly excluded below; no source refs invented.",
            "All-wrong and all-correct single-response scenarios test rules, not mastery.",
        ],
        "topics": {},
    }
    keys = ("status", "coveredConcepts", "inferredPrerequisites", "foundationStatus")
    for topic in sorted({b["topic_id"] for b in data["books"]}):
        graph = concept_graph(config, topic)
        nodes = {n["id"] for n in graph["nodes"]}
        rows = []
        excluded = {}
        books = sorted(
            (b for b in data["books"] if b["topic_id"] == topic), key=lambda b: b["book_id"]
        )
        for book in books:
            covered = sorted(set(book["covered_concepts"]) & nodes)
            unknown = sorted(set(book["covered_concepts"]) - nodes)
            if unknown:
                excluded[book["book_id"]] = unknown
            if not covered:
                continue
            result = {"bookId": book["book_id"]}
            for label, version, correct in (
                ("v1Wrong", "concept-learning-v1", 0),
                ("v2Wrong", "concept-learning-v2", 0),
                ("v2Correct", "concept-learning-v2", 1),
            ):
                request = LearningFitRequest.model_validate(
                    {
                        "topicId": topic,
                        "modelVersion": version,
                        "observations": [
                            {
                                "conceptId": n,
                                "ability": "application",
                                "responseCount": 1,
                                "correctCount": correct,
                            }
                            for n in sorted(nodes)
                        ],
                        "candidateBooks": [
                            {
                                "bookId": book["book_id"],
                                "coveredConcepts": covered,
                                "sourceArtifactVersion": data["snapshot_version"],
                                "sourceArtifactHash": digest,
                            }
                        ],
                    }
                )
                item = recommend_learning(request, config)["items"][0]
                result[label] = {k: item[k] for k in keys}
                if label == "v2Wrong":
                    result["checklistOrder"] = [
                        c["conceptId"] for c in item["readingChecklist"]["concepts"]
                    ]
                    result["internalPrerequisites"] = item["internalPrerequisites"]
            rows.append(result)
        report["topics"][topic] = {
            "graphHash": graph["graphHash"],
            "graphReviewHash": graph["reviewHash"],
            "totalBooks": len(books),
            "mappedBooks": len(rows),
            "unmappedBooks": len(books) - len(rows),
            "excludedConcepts": excluded,
            "statusChanges": sum(r["v1Wrong"]["status"] != r["v2Wrong"]["status"] for r in rows),
            "books": rows,
        }
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--snapshot", type=Path, default=Path("tests/fixtures/ranking_v2/scale50_snapshot.json")
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = compare(args.snapshot)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    print(
        json.dumps(
            {
                topic: {k: v for k, v in row.items() if k != "books"}
                for topic, row in report["topics"].items()
            },
            ensure_ascii=False,
        )
    )
