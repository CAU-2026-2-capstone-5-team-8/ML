"""Frozen candidates keep their scores; explicit new artifacts add source references."""

import importlib
import json
import runpy
from pathlib import Path

import pytest
from test_source_aware_adapter import _mapping, _profiles

from bookmatch_ml.ranking.source_aware_adapter import build_source_aware_matching_book_profiles


def test_handoff_preserves_concepts_and_weights_and_joins_all_support():
    mapping = _mapping()
    candidates = build_source_aware_matching_book_profiles(mapping, _profiles())
    before = [c.model_dump() for c in candidates]
    module = importlib.util.find_spec("bookmatch_ml.concept_v2.learning_handoff")
    assert module is not None, "explicit evidence handoff must be implemented"
    from bookmatch_ml.concept_v2.learning_handoff import enrich_candidates

    result = enrich_candidates(mapping, candidates)
    assert [c.model_dump() for c in candidates] == before
    os_book = next(b for b in result if "operating-systems" in b["topic_distribution"])
    process = next(c for c in os_book["covered_concepts"] if c["concept"] == "process")
    assert process["weight"] == 1.0
    assert len(process["evidence"]) == 3
    assert process["evidence"][0]["toc_path"][0] == "Fixture"
    assert result == enrich_candidates(mapping, list(reversed(candidates)))
    with pytest.raises(ValueError, match="identities"):
        enrich_candidates(mapping, candidates[:1])
    changed = candidates[0].model_copy(update={"covered_concepts": []})
    with pytest.raises(ValueError, match="concepts"):
        enrich_candidates(mapping, [changed, *candidates[1:]])


def test_handoff_command_is_repeatable_and_refuses_existing_output(tmp_path):
    mapping = _mapping()
    candidates = build_source_aware_matching_book_profiles(mapping, _profiles())
    mapping_path, candidate_path = tmp_path / "mapping.json", tmp_path / "candidates.jsonl"
    mapping_path.write_text(mapping.model_dump_json(), encoding="utf-8")
    candidate_path.write_text("\n".join(c.model_dump_json() for c in candidates), encoding="utf-8")
    prepare = runpy.run_path(
        str(Path(__file__).parents[1] / "scripts/prepare_learning_evidence.py")
    )["prepare"]
    first, second = tmp_path / "first.jsonl", tmp_path / "second.jsonl"
    report = prepare(candidate_path, mapping_path, first)
    assert prepare(candidate_path, mapping_path, second) == report
    assert first.read_bytes() == second.read_bytes()
    assert report["books"] == 2
    assert json.loads(first.read_text(encoding="utf-8").splitlines()[0])[
        "feature_version"
    ].endswith("-evidence-v1")
    with pytest.raises(ValueError, match="output exists"):
        prepare(candidate_path, mapping_path, first)
    assert first.read_bytes() == second.read_bytes()
