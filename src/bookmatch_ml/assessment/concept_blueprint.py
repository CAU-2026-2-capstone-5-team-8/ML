"""Build concept × ability targets from canonical TOC mappings, without prose."""

import hashlib
import json
from pathlib import Path

from bookmatch_ml.assessment.concept_contract import (
    OPERATIONS,
    ConceptBlueprint,
    ConceptQuestionSpec,
    content_hash,
)
from bookmatch_ml.concept_v2.graph import load_concept_graph
from bookmatch_ml.concept_v2.profile import (
    build_book_concept_profile_v2,
    load_concept_matching_config,
)
from bookmatch_ml.concept_v2.toc_source_selection import POLICY as TOC_SOURCE_POLICY
from bookmatch_ml.concept_v2.toc_source_selection import select_toc_source
from bookmatch_ml.config import load_feature_config
from bookmatch_ml.data.evidence import assemble_book_evidence
from bookmatch_ml.data.loader import load_canonical_dataset


def build_concept_blueprint(
    data_dir: Path, target_config: Path, config_dir: Path, *, toc_source_policy: str | None = None
):
    if toc_source_policy not in (None, TOC_SOURCE_POLICY):
        raise ValueError("unknown TOC source policy")
    targets = json.loads(target_config.read_text())
    if set(targets) != {"config_version", "topic_id", "concepts"}:
        raise ValueError("invalid target configuration")
    dataset = load_canonical_dataset(data_dir)
    hashes = {
        name: "sha256:" + hashlib.sha256((data_dir / name).read_bytes()).hexdigest()
        for name in ("books.jsonl", "documents.jsonl", "toc.jsonl", "sources.jsonl")
    }
    features = load_feature_config(config_dir / "features.yaml")
    graph = load_concept_graph(config_dir / "concept_graph.yaml", features)
    matching = load_concept_matching_config(config_dir / "concept_matching_v2.yaml")
    generation_config_hash = content_hash(
        {
            "targets": targets,
            "mapping_inputs": {
                name: "sha256:" + hashlib.sha256((config_dir / name).read_bytes()).hexdigest()
                for name in ("features.yaml", "concept_graph.yaml", "concept_matching_v2.yaml")
            },
        }
    )
    references = {}
    for book in assemble_book_evidence(dataset):
        if targets["topic_id"] not in book.metadata.topics:
            continue
        if toc_source_policy is not None:
            book, _ = select_toc_source(book)
        profile = build_book_concept_profile_v2(
            book, targets["topic_id"], features, graph, matching, hashes["toc.jsonl"]
        )
        for concept in profile.covered_concepts:
            references.setdefault(concept.concept_id, []).extend(
                {"book_id": book.book_id, "toc_entry_id": entry} for entry in concept.toc_entry_ids
            )
    specs = []
    for concept, abilities in targets["concepts"].items():
        if concept not in graph.graph.nodes.get(targets["topic_id"], []):
            raise ValueError(f"unknown canonical concept: {concept}")
        if not references.get(concept):
            raise ValueError(f"no TOC evidence for target: {concept}")
        if set(abilities) != set(OPERATIONS):
            raise ValueError(f"all three ability targets required: {concept}")
        for ability, target in abilities.items():
            if set(target) != {"objective", "misconceptions", "difficulty"}:
                raise ValueError("invalid ability target")
            data = dict(
                question_spec_version="concept-question-spec-v2",
                topic_id=targets["topic_id"],
                primary_concept=concept,
                ability=ability,
                cognitive_operation=OPERATIONS[ability],
                measurement_context="prior-knowledge",
                target_difficulty=target["difficulty"],
                assessment_objective=target["objective"],
                misconception_targets=target["misconceptions"],
                evidence_references=sorted(
                    references[concept], key=lambda r: (r["book_id"], r["toc_entry_id"])
                ),
                config_version=targets["config_version"],
                config_hash=generation_config_hash,
                canonical_file_hashes=hashes,
            )
            data["question_id"] = "q_" + content_hash(data).split(":")[1][:20]
            specs.append(ConceptQuestionSpec.model_validate(data))
    return ConceptBlueprint(
        blueprint_version="concept-assessment-blueprint-v2",
        topic_id=targets["topic_id"],
        question_specs=specs,
    )
