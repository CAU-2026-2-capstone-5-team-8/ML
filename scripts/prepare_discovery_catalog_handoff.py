"""Freeze the 471-book catalog and existing LA-83 candidates without new collection or inference."""

import argparse
import json
from collections import Counter
from pathlib import Path

from prepare_linear_algebra_handoff import build_ml_http, checked, digest, write_json

from bookmatch_ml.data.loader import load_canonical_dataset
from bookmatch_ml.integration.schemas import BookCandidateDto
from bookmatch_ml.ranking.loader import load_book_profiles
from bookmatch_ml.ranking.source_aware_adapter import (
    build_source_aware_matching_book_profiles,
    load_concept_mapping_report,
)
from bookmatch_ml.schemas import MatchingBookProfile

ROOT = Path(__file__).resolve().parents[1]
PIN = ROOT / "configs/experiments/discovery-catalog-live-v1.json"


def coverage_rows(dataset) -> list[dict]:
    """Counts from validated canonical ownership; availability is not recommendation eligibility."""
    toc = Counter(row.book_id for row in dataset.toc)
    descriptions = Counter(
        row.book_id for row in dataset.documents if row.document_type == "description"
    )
    sources = Counter(row.book_id for row in dataset.sources)
    return [
        {
            "book_id": book.book_id,
            "toc_entry_count": toc[book.book_id],
            "description_count": descriptions[book.book_id],
            "source_count": sources[book.book_id],
        }
        for book in sorted(dataset.books, key=lambda row: row.book_id)
    ]


def jsonl(rows: list[dict]) -> bytes:
    return "".join(
        json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows
    ).encode()


def prepare(topics_dir: Path, la_pilot_dir: Path, la_packet_dir: Path, output_dir: Path) -> dict:
    pin = json.loads(PIN.read_text())
    if output_dir.exists():
        raise ValueError("output-dir exists; reuse its manifest or choose a new directory")
    # Validate every source before creating any output. Never silently use another dataset.
    datasets = {}
    for entry in pin["topics"]:
        topic = entry["topic"]["ml_topic_id"]
        folder = topics_dir / topic / "processed"
        for name, expected in entry["canonical_hashes"].items():
            checked(folder / name, expected)
        dataset = load_canonical_dataset(folder)
        count = sum(row["toc_entry_count"] > 0 for row in coverage_rows(dataset))
        if len(dataset.books) != entry["book_count"] or count != entry["toc_book_count"]:
            raise ValueError(f"pinned coverage count differs: {topic}")
        datasets[topic] = dataset
    all_ids = [b.book_id for dataset in datasets.values() for b in dataset.books]
    if len(all_ids) != pin["book_count"] or len(set(all_ids)) != len(all_ids):
        raise ValueError("cross-topic duplicate identity or wrong catalog count")
    for name, expected in pin["la_pilot_hashes"].items():
        checked(la_pilot_dir / name, expected)
    mapping = load_concept_mapping_report(la_pilot_dir / "concept-mapping.json")
    candidates = [
        MatchingBookProfile.model_validate_json(line)
        for line in (la_pilot_dir / "matching-candidates.jsonl").read_text().splitlines()
    ]
    rebuilt = build_source_aware_matching_book_profiles(
        mapping, load_book_profiles(la_pilot_dir / "book-profiles.jsonl")
    )
    if candidates != rebuilt:
        raise ValueError("LA candidates differ from source-aware mapping/profile join")
    la_toc_ids = {
        row["book_id"]
        for row in coverage_rows(datasets["linear-algebra"])
        if row["toc_entry_count"] > 0
    }
    if {c.book_id for c in candidates} != la_toc_ids or len(candidates) != 83:
        raise ValueError("LA candidate identities must exactly match the 83 TOC books")
    previous = json.loads((la_packet_dir / "catalog-manifest.json").read_text())
    accepted = pin["accepted_projection_sources"][0]
    if (
        previous["snapshot_id"] != accepted["snapshot_id"]
        or previous["candidates_sha256"] != accepted["candidates_sha256"]
    ):
        raise ValueError("previous packet is not the pinned three-book handoff")
    checked(la_packet_dir / previous["candidates_path"], accepted["candidates_sha256"])
    assessment = json.loads((la_packet_dir / "assessment-request.json").read_text())
    if assessment["topicId"] != "linear-algebra" or len(assessment["responses"]) != 9:
        raise ValueError("explicit LA nine-answer validation request required")
    output_dir.mkdir(parents=True)
    batches = []
    totals = []
    for entry in pin["topics"]:
        topic = entry["topic"]["ml_topic_id"]
        folder = topics_dir / topic / "processed"
        books_name = topic + "-books.jsonl"
        evidence_name = topic + "-evidence.jsonl"
        (output_dir / books_name).write_bytes((folder / "books.jsonl").read_bytes())
        evidence = coverage_rows(datasets[topic])
        evidence_bytes = jsonl(evidence)
        (output_dir / evidence_name).write_bytes(evidence_bytes)
        batch = {
            "topic": entry["topic"],
            "books_path": books_name,
            "books_sha256": entry["canonical_hashes"]["books.jsonl"],
            "evidence_path": evidence_name,
            "evidence_sha256": digest(evidence_bytes),
            "candidates_path": None,
            "candidates_sha256": None,
            "canonical_hashes": entry["canonical_hashes"],
        }
        if topic == "linear-algebra":
            name = "linear-algebra-candidates.jsonl"
            raw = (la_pilot_dir / "matching-candidates.jsonl").read_bytes()
            (output_dir / name).write_bytes(raw)
            batch["candidates_path"] = name
            batch["candidates_sha256"] = digest(raw)
        batches.append(batch)
        totals.append(
            {
                "topic": topic,
                "books": len(evidence),
                "toc_books": sum(row["toc_entry_count"] > 0 for row in evidence),
                "candidates": 83 if topic == "linear-algebra" else 0,
            }
        )
    write_json(
        output_dir / "catalog-manifest.json",
        {
            "contract_version": "discovery-catalog-import-v1",
            "snapshot_id": pin["snapshot_id"],
            "topics": batches,
            "accepted_projection_sources": pin["accepted_projection_sources"],
        },
    )
    write_json(output_dir / "assessment-request.json", assessment)
    wire = [
        BookCandidateDto(**row.model_dump()).model_dump(mode="json", by_alias=True)
        for row in candidates
    ]
    write_json(output_dir / "wire-candidates.json", wire)
    (output_dir / "ml-live.http").write_text(build_ml_http(assessment, wire))
    summary = {
        "snapshot_id": pin["snapshot_id"],
        "book_count": len(all_ids),
        "toc_book_count": sum(row["toc_books"] for row in totals),
        "topics": totals,
        "la_candidate_count": len(candidates),
        "la_concept_book_count": sum(bool(row.covered_concepts) for row in candidates),
        "ranking_policy_changed": False,
        "quality_note": "Unfiltered discovery catalog: exam/general-interest books are retained.",
    }
    write_json(output_dir / "summary.json", summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--canonical-topics-dir", type=Path, required=True)
    parser.add_argument("--la-pilot-dir", type=Path, required=True)
    parser.add_argument("--la-packet-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    print(
        json.dumps(
            prepare(
                args.canonical_topics_dir, args.la_pilot_dir, args.la_packet_dir, args.output_dir
            ),
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
