"""Compare v2/v3 on hash-pinned catalog facts with explicitly synthetic reader answers."""

import argparse
import csv
import hashlib
import json
from pathlib import Path

from bookmatch_ml.concept_v2.learning_fit import LearningFitRequest, recommend_learning
from bookmatch_ml.concept_v2.learning_order import CRITERIA, POLICY
from bookmatch_ml.concept_v2.presentation import concept_graph
from bookmatch_ml.config import load_ranking_v2_config


def digest(raw):
    return "sha256:" + hashlib.sha256(raw).hexdigest()


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def records(base, name, expected):
    raw = (base / name).read_bytes()
    if digest(raw) != expected:
        raise ValueError(f"input hash mismatch: {name}")
    by_id = {}
    for line in raw.decode("utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line, object_pairs_hook=unique_object)
        if row["book_id"] in by_id:
            raise ValueError("duplicate book_id")
        by_id[row["book_id"]] = row
    return by_id


def compare(manifest_path, config_path):
    raw = manifest_path.read_bytes()
    manifest = json.loads(raw, object_pairs_hook=unique_object)
    if manifest["contract_version"] != "local-catalog-import-v1":
        raise ValueError("local-catalog-import-v1 manifest required")
    base = manifest_path.resolve().parent
    titles = records(base, manifest["books_path"], manifest["books_sha256"])
    candidates = records(base, manifest["candidates_path"], manifest["candidates_sha256"])
    selected = manifest["selected_book_ids"]
    if not selected or len(selected) != len(set(selected)):
        raise ValueError("nonempty unique selection required")
    topic = manifest["topic"]["ml_topic_id"]
    config = load_ranking_v2_config(config_path)
    graph = concept_graph(config, topic)
    nodes = sorted(n["id"] for n in graph["nodes"])
    books = []
    for identity in sorted(selected):
        if identity not in titles or identity not in candidates:
            raise ValueError("selected book missing from handoff")
        candidate = candidates[identity]
        if (
            set(candidate["topic_distribution"]) != {topic}
            or topic not in titles[identity]["topics"]
        ):
            raise ValueError("topic mapping mismatch")
        books.append(
            {
                "bookId": identity,
                "coveredConcepts": [c["concept"] for c in candidate["covered_concepts"]],
                "conceptEvidence": [
                    e for c in candidate["covered_concepts"] for e in c.get("evidence", [])
                ],
                "sourceArtifactVersion": manifest["snapshot_id"],
                "sourceArtifactHash": manifest["candidates_sha256"],
            }
        )
    scenarios = []
    for label, counts in [
        ("unmeasured", {}),
        ("all-incorrect", dict.fromkeys(nodes, 0)),
        ("all-correct", dict.fromkeys(nodes, 2)),
        ("alternating-A", {n: 2 if i % 2 == 0 else 0 for i, n in enumerate(nodes)}),
        ("alternating-B", {n: 0 if i % 2 == 0 else 2 for i, n in enumerate(nodes)}),
        ("partial-first-half", dict.fromkeys(nodes[: len(nodes) // 2], 2)),
    ]:
        observations = [
            {"conceptId": n, "ability": "application", "responseCount": 2, "correctCount": c}
            for n, c in sorted(counts.items())
        ]
        outputs = {}
        for key, version in [("before", "concept-learning-v2"), ("after", "concept-learning-v3")]:
            request = LearningFitRequest.model_validate(
                {
                    "topicId": topic,
                    "ability": "application",
                    "modelVersion": version,
                    "observations": observations,
                    "candidateBooks": books,
                    "limit": 20,
                }
            )
            response = recommend_learning(request, config)
            # Preserve semantic ties before the bookId display-order tiebreaker.
            # v2 sorts on these four fields; v3 already returns rankGroup.
            groups = {}
            for item in response["items"]:
                if key == "before":
                    group_key = (
                        item["status"],
                        item["reviewOnly"],
                        item["foundationStatus"] == "not-established",
                        item["practiceConceptCount"] == 0,
                    )
                    groups.setdefault(group_key, len(groups) + 1)
                    item["rankGroup"] = groups[group_key]
            outputs[key] = [
                {
                    **{
                        k: item[k]
                        for k in (
                            "bookId",
                            "rank",
                            "status",
                            "coveredConcepts",
                            "inferredPrerequisites",
                            "reasons",
                        )
                    },
                    **{
                        k: item[k]
                        for k in ("rankGroup", "tieCount", "personalization")
                        if k in item
                    },
                    "title": titles[item["bookId"]]["title"],
                }
                for item in response["items"]
            ]
        prior_rank = {r["bookId"]: r["rank"] for r in outputs["before"]}
        changed = sum(prior_rank.get(r["bookId"]) != r["rank"] for r in outputs["after"])
        scenarios.append(
            {
                "name": label,
                "observations": observations,
                "changedDisplayPositions": changed,
                **outputs,
            }
        )
    return {
        "version": "learning-order-comparison-v2",
        "orderingPolicy": POLICY,
        "orderingCriteria": CRITERIA,
        "snapshotId": manifest["snapshot_id"],
        "manifestHash": digest(raw),
        "booksHash": manifest["books_sha256"],
        "candidatesHash": manifest["candidates_sha256"],
        "graphHash": graph["graphHash"],
        "graphReviewHash": graph["reviewHash"],
        "configHash": graph["configHash"],
        "topicId": topic,
        "bookCount": len(books),
        "mappedBookCount": sum(bool(b["coveredConcepts"]) for b in books),
        "readerEvidence": "synthetic_scenarios_not_real_assessments",
        "accuracyMeasured": False,
        "limit": 20,
        "scenarios": scenarios,
    }


def write_outputs(report, directory):
    # Never overwrite an existing experiment or partially completed human evaluation.
    directory.mkdir(parents=True, exist_ok=False)
    with (directory / "comparison.json").open("x", encoding="utf-8") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    comparison_hash = digest((directory / "comparison.json").read_bytes())
    fields = [
        "comparison_hash",
        "snapshot_id",
        "candidates_hash",
        "scenario",
        "book_id",
        "title",
        "observations_json",
        "suitability_1_to_5",
        "reason",
        "reviewer",
        "reviewed_at",
    ]
    with (directory / "human-review.csv").open("x", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for scenario in report["scenarios"]:
            # Stable book identity order hides both algorithms' ranks from this worksheet.
            union = {r["bookId"]: r for side in ("before", "after") for r in scenario[side]}
            for identity, row in sorted(union.items()):
                writer.writerow(
                    {
                        "comparison_hash": comparison_hash,
                        "snapshot_id": report["snapshotId"],
                        "candidates_hash": report["candidatesHash"],
                        "scenario": scenario["name"],
                        "book_id": identity,
                        "title": row["title"],
                        "observations_json": json.dumps(
                            scenario["observations"], ensure_ascii=False
                        ),
                    }
                )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("configs/ranking_v2.yaml"))
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    result = compare(args.manifest, args.config)
    write_outputs(result, args.output_dir)
    print(
        json.dumps(
            {
                "bookCount": result["bookCount"],
                "mappedBookCount": result["mappedBookCount"],
                "changedDisplayPositions": {
                    s["name"]: s["changedDisplayPositions"] for s in result["scenarios"]
                },
            }
        )
    )
