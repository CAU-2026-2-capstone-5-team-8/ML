"""Join frozen candidates with source mapping into a NEW Backend candidate artifact."""

import argparse
import hashlib
import json
from pathlib import Path

from bookmatch_ml.concept_v2.learning_handoff import enrich_candidates
from bookmatch_ml.ranking.source_aware_adapter import load_concept_mapping_report
from bookmatch_ml.schemas import MatchingBookProfile


def prepare(candidates: Path, mapping: Path, output: Path) -> dict:
    if output.exists():
        raise ValueError("output exists; immutable inputs and outputs cannot be overwritten")
    raw_candidates = candidates.read_bytes()
    raw_mapping = mapping.read_bytes()
    books = [
        MatchingBookProfile.model_validate_json(line)
        for line in raw_candidates.splitlines()
        if line.strip()
    ]
    if not books:
        raise ValueError("candidate artifact is empty")
    rows = enrich_candidates(load_concept_mapping_report(mapping), books)
    if candidates.read_bytes() != raw_candidates or mapping.read_bytes() != raw_mapping:
        raise ValueError("input changed during preparation")
    content = "".join(
        json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows
    ).encode("utf-8")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("xb") as stream:
        stream.write(content)
    return {
        "artifactVersion": "learning-evidence-handoff-v1",
        "books": len(rows),
        "candidateInputHash": "sha256:" + hashlib.sha256(raw_candidates).hexdigest(),
        "mappingInputHash": "sha256:" + hashlib.sha256(raw_mapping).hexdigest(),
        "outputHash": "sha256:" + hashlib.sha256(content).hexdigest(),
        "rankingPolicyChanged": False,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--mapping", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(args.candidates, args.mapping, args.output), ensure_ascii=False))
