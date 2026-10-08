"""Validate proposed topic outlines and ground question targets in canonical TOCs.

No model calls, provider adapters, database writes or live ranking configuration changes.
The same function accepts a manually authored outline as an offline baseline.
"""

import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from bookmatch_ml.assessment.concept_blueprint import build_concept_blueprint
from bookmatch_ml.concept_v2.graph import ConceptGraph, load_concept_graph
from bookmatch_ml.concept_v2.profile import (
    build_book_concept_profile_v2,
    load_concept_matching_config,
)
from bookmatch_ml.concept_v2.toc_source_selection import POLICY as TOC_SOURCE_POLICY
from bookmatch_ml.concept_v2.toc_source_selection import select_toc_source
from bookmatch_ml.config import load_feature_config
from bookmatch_ml.data.evidence import assemble_book_evidence
from bookmatch_ml.data.loader import load_canonical_dataset

VERSION = "topic-content-preparation-v1"
FILES = ("books.jsonl", "documents.jsonl", "toc.jsonl", "sources.jsonl")


class Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True, allow_inf_nan=False)


class AbilityTarget(Strict):
    objective: str = Field(min_length=10, max_length=500)
    misconceptions: list[str] = Field(min_length=2, max_length=5)
    difficulty: Literal[1, 2, 3]

    @model_validator(mode="after")
    def distinct_misconceptions(self):
        if any(len(item) < 5 for item in self.misconceptions) or len(
            set(self.misconceptions)
        ) != len(self.misconceptions):
            raise ValueError("misconceptions must be substantive and distinct")
        return self


class AbilityTargets(Strict):
    meaning: AbilityTarget
    application: AbilityTarget
    reasoning: AbilityTarget


class ProposedConcept(Strict):
    id: str = Field(pattern=r"^[a-z][a-z0-9-]{1,59}$")
    name: str = Field(min_length=2, max_length=100)
    aliases: list[str] = Field(min_length=2, max_length=10)
    abilities: AbilityTargets


class ProposedEdge(Strict):
    prerequisite: str
    dependent: str
    reason: str = Field(min_length=10, max_length=500)


class TopicOutline(Strict):
    outline_version: Literal["topic-outline-v1"]
    topic_id: str = Field(pattern=r"^[a-z][a-z0-9-]{1,79}$")
    concepts: list[ProposedConcept] = Field(min_length=6, max_length=12)
    edges: list[ProposedEdge] = Field(max_length=24)

    @model_validator(mode="after")
    def validate_outline(self):
        if len({c.id for c in self.concepts}) != len(self.concepts):
            raise ValueError("duplicate concept id")
        aliases = set()
        for concept in self.concepts:
            for alias in concept.aliases:
                key = " ".join(alias.casefold().split())
                if len(key) < 2 or key in aliases:
                    raise ValueError("empty, duplicate or shared alias")
                aliases.add(key)
        # Reuse the real graph validator: unknown endpoints, cycles and duplicates fail closed.
        self.graph()
        return self

    def graph(self, proposal_source: Literal["ai_proposed", "proposed_seed"] = "proposed_seed"):
        return ConceptGraph.model_validate(
            {
                "config_version": VERSION,
                "graph_version": VERSION,
                "relation_type": "prerequisite_candidate",
                "nodes": {self.topic_id: [self.canonical_id(c.id) for c in self.concepts]},
                "edges": [
                    {
                        "topic": self.topic_id,
                        "prerequisite": self.canonical_id(e.prerequisite),
                        "dependent": self.canonical_id(e.dependent),
                        "source_type": proposal_source,
                        "version": VERSION,
                    }
                    for e in self.edges
                ],
            }
        )

    def canonical_id(self, concept_id: str):
        return f"{self.topic_id}:{concept_id}"


class PreparationPolicy(Strict):
    version: Literal["topic-target-selection-v1"] = "topic-target-selection-v1"
    minimum_books_per_target: int = Field(default=2, ge=1)
    minimum_targets: int = Field(default=3, ge=1)
    maximum_targets: int = Field(default=6, ge=1, le=12)

    @model_validator(mode="after")
    def valid_range(self):
        if self.minimum_targets > self.maximum_targets:
            raise ValueError("minimum targets exceeds maximum")
        return self


def digest(path: Path):
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n")


def prepare_topic_content(
    data_dir: Path,
    outline: TopicOutline,
    output_dir: Path,
    config_dir: Path,
    policy: PreparationPolicy | None = None,
    proposal_source: Literal["ai_proposed", "proposed_seed"] = "proposed_seed",
):
    """Return reproducible TOC evidence and a v2 blueprint, never generated questions."""
    policy = policy or PreparationPolicy()
    dataset = load_canonical_dataset(data_dir)
    books = [b for b in assemble_book_evidence(dataset) if outline.topic_id in b.metadata.topics]
    selections = [select_toc_source(book) for book in books]
    books = [selected for selected, _ in selections]
    if not books:
        raise ValueError("canonical dataset has no books in requested topic")
    hashes = {name: digest(data_dir / name) for name in FILES}
    output_dir.mkdir(parents=True, exist_ok=True)
    configs = output_dir / "configs"
    configs.mkdir(exist_ok=True)
    features = load_feature_config(config_dir / "features.yaml").config.model_dump()
    lexicon = {outline.canonical_id(c.id): c.aliases for c in outline.concepts}
    features["config_version"] = VERSION
    features["concept"]["topics"] = {outline.topic_id: lexicon}
    features["prerequisite"]["topics"] = {outline.topic_id: lexicon}
    (configs / "features.yaml").write_text(yaml.safe_dump(features, allow_unicode=True))
    (configs / "concept_graph.yaml").write_text(
        yaml.safe_dump(outline.graph(proposal_source).model_dump(), allow_unicode=True)
    )
    # Keep the established deterministic matcher and weights unchanged.
    matching = load_concept_matching_config(config_dir / "concept_matching_v2.yaml")
    matching_data = matching.config.model_dump()
    matching_data["toc_mapping"] = {"alias_additions": {}, "exclusions": {}}
    matching_data["toc_text_policy"] = "original_and_english"
    matching_data["config_version"] = "topic-bilingual-single-toc-config-v2"
    matching_data["profile_version"] = "topic-bilingual-single-toc-profile-v2"
    (configs / "concept_matching_v2.yaml").write_text(
        yaml.safe_dump(matching_data, allow_unicode=True)
    )
    loaded_features = load_feature_config(configs / "features.yaml")
    graph = load_concept_graph(configs / "concept_graph.yaml", loaded_features)
    matching = load_concept_matching_config(configs / "concept_matching_v2.yaml")
    profiles = [
        build_book_concept_profile_v2(
            b, outline.topic_id, loaded_features, graph, matching, hashes["toc.jsonl"]
        )
        for b in sorted(books, key=lambda b: b.book_id)
    ]
    evidence = defaultdict(list)
    for profile in profiles:
        for concept in profile.covered_concepts:
            evidence[concept.concept_id].extend(
                {"book_id": profile.book_id, "toc_entry_id": entry}
                for entry in concept.toc_entry_ids
            )
    concepts = []
    for concept in outline.concepts:
        concept_id = outline.canonical_id(concept.id)
        refs = sorted(evidence[concept_id], key=lambda r: (r["book_id"], r["toc_entry_id"]))
        concepts.append(
            {
                "id": concept_id,
                "name": concept.name,
                "bookCount": len({ref["book_id"] for ref in refs}),
                "tocCount": len(refs),
                "evidenceReferences": refs,
            }
        )
    selected = sorted(
        [c for c in concepts if c["bookCount"] >= policy.minimum_books_per_target],
        key=lambda c: (-c["bookCount"], -c["tocCount"], c["id"]),
    )[: policy.maximum_targets]
    ready = len(selected) >= policy.minimum_targets
    spec_count = 0
    if ready:
        by_id = {outline.canonical_id(c.id): c for c in outline.concepts}
        targets = {
            "config_version": VERSION,
            "topic_id": outline.topic_id,
            "concepts": {c["id"]: by_id[c["id"]].abilities.model_dump() for c in selected},
        }
        write_json(output_dir / "targets.json", targets)
        blueprint = build_concept_blueprint(
            data_dir, output_dir / "targets.json", configs, toc_source_policy=TOC_SOURCE_POLICY
        )
        write_json(output_dir / "blueprint.json", blueprint.model_dump())
        spec_count = len(blueprint.question_specs)
    elif (output_dir / "blueprint.json").exists():
        raise ValueError("refusing to overwrite an existing blueprint with insufficient evidence")
    (output_dir / "book-profiles.jsonl").write_text(
        "".join(
            json.dumps(p.model_dump(), ensure_ascii=False, sort_keys=True) + "\n" for p in profiles
        )
    )
    write_json(output_dir / "outline.json", outline.model_dump())
    report = {
        "contractVersion": VERSION,
        "topicId": outline.topic_id,
        "status": "CONCEPTS_READY" if ready else "NEEDS_EVIDENCE",
        "canonicalFileHashes": hashes,
        "outlineHash": digest(output_dir / "outline.json"),
        "policy": policy.model_dump(),
        "configHashes": {p.name: digest(p) for p in sorted(configs.glob("*.yaml"))},
        "bookCount": len(books),
        "mappedBookCount": sum(bool(p.covered_concepts) for p in profiles),
        "conceptCount": sum(c["bookCount"] > 0 for c in concepts),
        "questionSpecCount": spec_count,
        "selectedConceptIds": [c["id"] for c in selected] if ready else [],
        "concepts": concepts,
        "generatedQuestionCount": 0,
        "diagnosisReady": False,
        "humanReview": "unreviewed",
        "proposalSource": proposal_source,
        "tocSourcePolicy": TOC_SOURCE_POLICY,
        "tocSourceSelection": [selection for _, selection in selections],
    }
    write_json(output_dir / "report.json", report)
    return report
