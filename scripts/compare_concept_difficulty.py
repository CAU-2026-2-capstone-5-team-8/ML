"""Compare v1 with a versioned curriculum rubric; export a blind human review sheet."""

import argparse
import csv
import hashlib
import json
from pathlib import Path

from bookmatch_ml.book.profile import build_book_profiles
from bookmatch_ml.concept_v2.difficulty import load_difficulty_policy, score_difficulty
from bookmatch_ml.concept_v2.graph import load_concept_graph
from bookmatch_ml.concept_v2.profile import (
    build_book_concept_profile_v2,
    load_concept_matching_config,
)
from bookmatch_ml.config import load_feature_config, load_ranking_config
from bookmatch_ml.data.evidence import assemble_book_evidence
from bookmatch_ml.data.loader import load_canonical_dataset
from bookmatch_ml.ranking.loader import load_reader_profile
from bookmatch_ml.ranking.matching import score_book_fit
from bookmatch_ml.schemas import ConceptReadiness

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--reader", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        parser.error("output-dir exists; choose a new directory to preserve human reviews")
    configs = ROOT / "configs"
    features = load_feature_config(configs / "features.yaml")
    graph = load_concept_graph(configs / "concept_graph.yaml", features)
    matching = load_concept_matching_config(configs / "concept_matching.yaml")
    policy = load_difficulty_policy(configs / "concept_difficulty.yaml")
    for topic, nodes in graph.graph.nodes.items():
        if set(nodes) != set(policy[0].levels.get(topic, {})):
            raise ValueError(f"rubric and graph concept sets differ: {topic}")
    ranking = load_ranking_config(configs / "ranking.yaml")
    reader = load_reader_profile(args.reader)
    hashes = {
        name: "sha256:" + hashlib.sha256((args.data_dir / name).read_bytes()).hexdigest()
        for name in ("books.jsonl", "toc.jsonl", "documents.jsonl", "sources.jsonl")
    }
    evidence = assemble_book_evidence(load_canonical_dataset(args.data_dir))
    baseline = {b.book_id: b for b in build_book_profiles(evidence, features)}
    scenarios = {"provided_reader": reader}
    for name, score in [
        ("synthetic_novice", 0.0),
        ("synthetic_intermediate", 0.5),
        ("synthetic_expert", 1.0),
    ]:
        scenarios[name] = reader.model_copy(
            update={
                "concept_readiness": [
                    ConceptReadiness(
                        concept_id=c,
                        score=score,
                        response_count=1,
                        earned_weight=score,
                        available_weight=1,
                    )
                    for c in graph.graph.nodes[reader.topic_id]
                ]
            }
        )
    reports = {name: [] for name in scenarios}
    review = []
    for book in evidence:
        if reader.topic_id not in book.metadata.topics:
            continue
        profile = build_book_concept_profile_v2(
            book, reader.topic_id, features, graph, matching, hashes["toc.jsonl"]
        )
        for name, scenario in scenarios.items():
            old = score_book_fit(scenario, baseline[book.book_id], ranking)
            result = score_difficulty(scenario, profile, policy)
            # Baseline comparison uses the same reader dimensions, which are held fixed.
            result["v1_score"] = old.score
            result["v1_component_weight_coverage"] = old.component_weight_coverage
            result["matched_toc_entries"] = profile.diagnostics.matched_toc_entries
            result["total_toc_entries"] = profile.diagnostics.total_toc_entries
            reports[name].append(result)
        review.append(
            {
                "book_id": book.book_id,
                "title": book.metadata.title,
                "topic": reader.topic_id,
                "work_group": "",
                "split": "unassigned",
                "reviewer": "",
                "topic_relevant": "",
                "human_concept_level_1_to_3": "",
                "human_reader_band": "",
                "evidence_reference": "",
                "notes": "",
            }
        )
    for items in reports.values():
        items.sort(
            key=lambda x: (
                x["recommendation_score"] is None,
                -(x["recommendation_score"] or 0),
                x["book_id"],
            )
        )
    args.output_dir.mkdir(parents=True)
    report = {
        "input_hashes": hashes,
        "reader_sha256": hashlib.sha256(args.reader.read_bytes()).hexdigest(),
        "scenarios": reports,
        "quality_evaluation": "pending independent human labels",
    }
    (args.output_dir / "comparison.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    if review:
        with (args.output_dir / "blind-review.csv").open(
            "w", encoding="utf-8-sig", newline=""
        ) as f:
            writer = csv.DictWriter(f, fieldnames=list(review[0]))
            writer.writeheader()
            writer.writerows(review)
    print(
        json.dumps(
            {
                name: {
                    band: sum(i["difficulty_band"] == band for i in items)
                    for band in sorted({i["difficulty_band"] for i in items})
                }
                for name, items in reports.items()
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
