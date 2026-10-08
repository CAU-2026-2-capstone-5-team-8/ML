"""Validate completed review worksheets and compare v2/v3 on shared book pairs.

No scores are generated here. Human agreement on synthetic reader scenarios is
not measured learning effectiveness or calibrated book difficulty.
"""

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from datetime import date
from itertools import combinations
from pathlib import Path


def digest(raw):
    return "sha256:" + hashlib.sha256(raw).hexdigest()


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def pairwise(scores, ranks):
    counts = dict(agree=0, disagree=0, humanTie=0, modelTie=0)
    for a, b in combinations(sorted(scores), 2):
        if scores[a] == scores[b]:
            counts["humanTie"] += 1
        elif ranks[a] == ranks[b]:
            counts["modelTie"] += 1
        elif (scores[a] > scores[b]) == (ranks[a] < ranks[b]):
            counts["agree"] += 1
        else:
            counts["disagree"] += 1
    decided = counts["agree"] + counts["disagree"]
    preferred = decided + counts["modelTie"]
    return {
        **counts,
        "agreementOnDecidedPairs": counts["agree"] / decided if decided else None,
        "decidedPairCoverage": decided / preferred if preferred else None,
    }


def evaluate(comparison_path, review_paths):
    raw = comparison_path.read_bytes()
    report = json.loads(raw, object_pairs_hook=unique_object)
    if report.get("version") != "learning-order-comparison-v2":
        raise ValueError("Regenerate comparison with v2 worksheet binding and semantic ties")
    comparison_hash = digest(raw)
    expected = {}
    scenarios = {}
    for scenario in report["scenarios"]:
        name = scenario["name"]
        if name in scenarios:
            raise ValueError("duplicate scenario")
        scenarios[name] = scenario
        for side in ("before", "after"):
            seen = set()
            for item in scenario[side]:
                identity = item["bookId"]
                group = item.get("rankGroup")
                if identity in seen or type(group) is not int or group < 1:
                    raise ValueError("invalid or duplicate ranked book")
                seen.add(identity)
                key = (name, identity)
                if key in expected and expected[key]["title"] != item["title"]:
                    raise ValueError("inconsistent book title")
                expected[key] = item

    votes = defaultdict(dict)
    seen_votes = set()
    input_hashes = []
    total_rows = 0
    for path in review_paths:
        input_hashes.append(digest(path.read_bytes()))
        with path.open(encoding="utf-8-sig", newline="") as stream:
            reader = csv.DictReader(stream)
            required = {
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
            }
            if not reader.fieldnames or set(reader.fieldnames) != required:
                raise ValueError("invalid worksheet columns; regenerate the worksheet")
            if len(reader.fieldnames) != len(required):
                raise ValueError("duplicate worksheet columns")
            file_rows = set()
            for row in reader:
                total_rows += 1
                if None in row or any(value is None for value in row.values()):
                    raise ValueError("invalid CSV row width")
                if row["comparison_hash"] != comparison_hash:
                    raise ValueError("comparison hash mismatch")
                if (
                    row["snapshot_id"] != report["snapshotId"]
                    or row["candidates_hash"] != report["candidatesHash"]
                ):
                    raise ValueError("snapshot or candidates hash mismatch")
                key = (row["scenario"], row["book_id"])
                if key not in expected or row["title"] != expected[key]["title"]:
                    raise ValueError("unknown or changed scenario/book/title")
                observations = json.loads(row["observations_json"], object_pairs_hook=unique_object)
                if observations != scenarios[key[0]]["observations"]:
                    raise ValueError("scenario observations changed")
                reviewer = row["reviewer"].strip()
                vote_key = (reviewer, *key)
                if vote_key in file_rows:
                    raise ValueError("duplicate worksheet row")
                file_rows.add(vote_key)
                score = row["suitability_1_to_5"].strip()
                if not score:
                    continue
                if score not in {"1", "2", "3", "4", "5"}:
                    raise ValueError("suitability must be an integer from 1 to 5 or blank")
                if not reviewer or not row["reason"].strip():
                    raise ValueError("scored rows require reviewer and reason")
                reviewed_at = row["reviewed_at"].strip()
                if date.fromisoformat(reviewed_at).isoformat() != reviewed_at:
                    raise ValueError("review date must be YYYY-MM-DD")
                if vote_key in seen_votes:
                    raise ValueError("duplicate reviewer vote across files")
                seen_votes.add(vote_key)
                votes[reviewer][key] = int(score)

    reviewers = []
    for reviewer, ratings in sorted(votes.items()):
        results = []
        for name, scenario in sorted(scenarios.items()):
            ranks = {
                side: {r["bookId"]: r["rankGroup"] for r in scenario[side]}
                for side in ("before", "after")
            }
            common = ranks["before"].keys() & ranks["after"].keys()
            scores = {b: s for (n, b), s in ratings.items() if n == name and b in common}
            results.append(
                {
                    "scenario": name,
                    "commonCandidateCount": len(common),
                    "ratedCommonBooks": len(scores),
                    "ratedOutsideCommonBooks": sum(
                        n == name and b not in common for n, b in ratings
                    ),
                    "ratingCoverage": len(scores) / len(common) if common else None,
                    **{side: pairwise(scores, ranks[side]) for side in ("before", "after")},
                }
            )
        reviewers.append({"reviewer": reviewer, "ratedRows": len(ratings), "scenarios": results})

    return {
        "version": "learning-order-human-evaluation-v1",
        "comparisonHash": comparison_hash,
        "reviewFileHashes": sorted(input_hashes),
        "snapshotId": report["snapshotId"],
        "readerEvidence": report["readerEvidence"],
        "status": "descriptive_review_results" if votes else "awaiting_reviews",
        "accuracyMeasured": False,
        "reviewerCount": len(votes),
        "reviewerIdentityVerified": False,
        "ratedRows": len(seen_votes),
        "unscoredRows": total_rows - len(seen_votes),
        "reviewers": reviewers,
        "limitations": [
            "Reviewer IDs are self-declared; distinct IDs do not prove independent people.",
            "Only shared, rated books are compared; this is not full-catalog accuracy.",
            "Human ties are excluded; model ties reduce decision coverage.",
            "Agreement must be read with pair counts and coverage, not alone.",
            "No automatic winner, production activation, or learning-effectiveness claim.",
        ],
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--comparison", type=Path, required=True)
    parser.add_argument("--reviews", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = evaluate(args.comparison, args.reviews)
    # Never overwrite previous results or accidentally destroy an input file.
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    print(json.dumps({k: result[k] for k in ("status", "ratedRows", "reviewerCount")}))
