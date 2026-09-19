"""Deterministic TOC-to-concept coverage and graph prerequisite projection."""

import hashlib
import re
from collections import defaultdict
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from bookmatch_ml.book.text import normalize_text
from bookmatch_ml.concept_v2.graph import LoadedConceptGraph
from bookmatch_ml.concept_v2.toc import TocVisit, reconstruct_toc
from bookmatch_ml.config import ConfigError, LoadedFeatureConfig, _load_unique_key_yaml
from bookmatch_ml.schemas import BookEvidence


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class ConceptMatchingPolicy(_Strict):
    minimum_readiness_assessment_coverage: float = Field(ge=0, le=1)
    minimum_opportunity_assessment_coverage: float = Field(ge=0, le=1)
    minimum_readiness_score: float = Field(ge=0, le=1)
    known_mastery_threshold: float = Field(ge=0, le=1)
    learning_mastery_threshold: float = Field(ge=0, le=1)


class TocMappingRules(_Strict):
    alias_additions: dict[str, dict[str, list[str]]] = Field(default_factory=dict)
    exclusions: dict[str, dict[str, list[str]]] = Field(default_factory=dict)

    @model_validator(mode="after")
    def rules_are_non_empty_and_unique(self) -> "TocMappingRules":
        for rule_name, topics in (
            ("alias additions", self.alias_additions),
            ("exclusions", self.exclusions),
        ):
            for topic, concepts in topics.items():
                if not topic.strip() or not concepts:
                    raise ValueError(f"TOC mapping {rule_name} require non-empty topics")
                for concept, phrases in concepts.items():
                    normalized = [normalize_text(phrase) for phrase in phrases]
                    if (
                        not concept.strip()
                        or not phrases
                        or any(not phrase for phrase in normalized)
                        or len(normalized) != len(set(normalized))
                    ):
                        raise ValueError(
                            f"TOC mapping {rule_name} require unique non-empty phrases"
                        )
        return self


class ConceptMatchingConfig(_Strict):
    config_version: str
    profile_version: str
    model_version: str
    coverage_weight_rule: Literal["min_occurrences_over_three_v1"]
    toc_mapping: TocMappingRules = Field(default_factory=TocMappingRules)
    policy: ConceptMatchingPolicy


class LoadedConceptMatchingConfig(_Strict):
    config: ConceptMatchingConfig
    content_hash: str


def load_concept_matching_config(path: Path) -> LoadedConceptMatchingConfig:
    try:
        content = path.read_bytes()
        config = ConceptMatchingConfig.model_validate(_load_unique_key_yaml(content))
    except (OSError, ValidationError, ValueError) as exc:
        raise ConfigError(f"invalid concept matching config: {path}: {exc}") from exc
    return LoadedConceptMatchingConfig(
        config=config, content_hash=f"sha256:{hashlib.sha256(content).hexdigest()}"
    )


class TocConceptMapping(_Strict):
    toc_entry_id: str
    source_id: str
    toc_title: str
    toc_label: str | None
    concept_id: str
    matching_alias: str
    match_method: Literal["normalized_alias_phrase_v1"]
    toc_level: int
    order_index: int
    parent_path: list[str]
    toc_path: list[str]
    traversal_position: int


class UnmappedTocEntry(_Strict):
    toc_entry_id: str
    title: str
    toc_path: list[str]
    reason: Literal["unmatched", "ambiguous_alias"]


class CoveredTocConcept(_Strict):
    concept_id: str
    coverage_weight: float = Field(ge=0, le=1)
    occurrence_count: int = Field(ge=1)
    minimum_toc_level: int = Field(ge=1)
    first_traversal_position: int = Field(ge=0)
    toc_entry_ids: list[str]
    toc_paths: list[list[str]]


class PrerequisiteCandidate(_Strict):
    concept_id: str
    weight: float = Field(ge=0, le=1)
    reason: Literal["taught_before_use_candidate", "external_or_late_candidate"]
    related_target_concepts: list[str]
    graph_paths: list[list[str]]
    evidence_toc_entry_ids: list[str]
    relation_source_types: list[str]


class TocCoverageDiagnostics(_Strict):
    total_toc_entries: int
    matched_toc_entries: int
    unmatched_toc_entries: int
    ambiguous_toc_entries: int
    concept_count: int


class BookConceptProfileV2(_Strict):
    book_id: str
    title: str
    topic_id: str
    toc_mappings: list[TocConceptMapping]
    unmapped_entries: list[UnmappedTocEntry]
    covered_concepts: list[CoveredTocConcept]
    prerequisite_requirements: list[PrerequisiteCandidate]
    taught_before_use_candidates: list[PrerequisiteCandidate]
    diagnostics: TocCoverageDiagnostics
    profile_version: str
    graph_version: str
    graph_config_version: str
    graph_hash: str
    feature_config_version: str
    feature_config_hash: str
    matching_config_version: str
    matching_config_hash: str
    toc_file_hash: str


def _mapping_inputs(
    features: LoadedFeatureConfig,
    graph: LoadedConceptGraph,
    matching: LoadedConceptMatchingConfig,
    topic: str,
) -> tuple[dict[str, list[str]], dict[str, list[str]]]:
    nodes = graph.graph.nodes[topic]
    covered = features.config.concept.topics[topic].root
    prerequisites = features.config.prerequisite.topics[topic].root
    aliases = {node: list(covered.get(node, prerequisites.get(node, []))) for node in nodes}
    rules = matching.config.toc_mapping
    additions = rules.alias_additions.get(topic, {})
    exclusions = rules.exclusions.get(topic, {})
    for concept, forms in additions.items():
        aliases[concept] = list(dict.fromkeys([*aliases[concept], *forms]))
    return aliases, exclusions


def validate_toc_mapping_rules(
    matching: LoadedConceptMatchingConfig, graph: LoadedConceptGraph
) -> None:
    """Reject mapping rules that cannot target the loaded concept graph."""

    rules = matching.config.toc_mapping
    configured_topics = set(rules.alias_additions) | set(rules.exclusions)
    unknown_topics = configured_topics - set(graph.graph.nodes)
    if unknown_topics:
        raise ConfigError(f"TOC mapping rules reference unknown topics: {sorted(unknown_topics)}")
    for rule_name, topics in (
        ("alias additions", rules.alias_additions),
        ("exclusions", rules.exclusions),
    ):
        for topic, concepts in topics.items():
            unknown_concepts = set(concepts) - set(graph.graph.nodes[topic])
            if unknown_concepts:
                raise ConfigError(
                    f"TOC mapping {rule_name} reference unknown {topic} concepts: "
                    f"{sorted(unknown_concepts)}"
                )


def _map_entry(
    visit: TocVisit,
    aliases: dict[str, list[str]],
    exclusions: dict[str, list[str]] | None = None,
) -> tuple[list[TocConceptMapping], bool]:
    title = normalize_text(visit.entry.title)
    excluded = {
        concept
        for concept, phrases in (exclusions or {}).items()
        if any(
            normalized and re.search(rf"(?<!\w){re.escape(normalized)}(?!\w)", title)
            for phrase in phrases
            if (normalized := normalize_text(phrase))
        )
    }
    found: dict[str, list[str]] = defaultdict(list)
    alias_to_concepts: dict[str, set[str]] = defaultdict(set)
    for concept, forms in aliases.items():
        if concept in excluded:
            continue
        for form in forms:
            normalized = normalize_text(form)
            if normalized and re.search(rf"(?<!\w){re.escape(normalized)}(?!\w)", title):
                found[concept].append(form)
                alias_to_concepts[normalized].add(concept)
    if any(len(concepts) > 1 for concepts in alias_to_concepts.values()):
        return [], True
    mappings = [
        TocConceptMapping(
            toc_entry_id=visit.entry.toc_entry_id,
            source_id=visit.entry.source_id,
            toc_title=visit.entry.title,
            toc_label=visit.entry.label,
            concept_id=concept,
            matching_alias=sorted(forms, key=lambda form: (-len(normalize_text(form)), form))[0],
            match_method="normalized_alias_phrase_v1",
            toc_level=visit.entry.level,
            order_index=visit.entry.order_index,
            parent_path=list(visit.parent_path),
            toc_path=list(visit.path),
            traversal_position=visit.traversal_position,
        )
        for concept, forms in sorted(found.items())
    ]
    return mappings, False


def _ancestor_paths(
    target: str, edges: list[tuple[str, str, str]]
) -> list[tuple[str, tuple[str, ...], tuple[str, ...]]]:
    incoming: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for before, after, source in edges:
        incoming[after].append((before, source))
    result: list[tuple[str, tuple[str, ...], tuple[str, ...]]] = []

    def walk(node: str, path: tuple[str, ...], sources: tuple[str, ...]) -> None:
        for ancestor, source in sorted(incoming[node]):
            next_path = (ancestor, *path)
            next_sources = (source, *sources)
            result.append((ancestor, next_path, next_sources))
            walk(ancestor, next_path, next_sources)

    walk(target, (target,), ())
    return result


def build_book_concept_profile_v2(
    evidence: BookEvidence,
    topic: str,
    features: LoadedFeatureConfig,
    graph: LoadedConceptGraph,
    matching: LoadedConceptMatchingConfig,
    toc_file_hash: str,
) -> BookConceptProfileV2:
    if topic not in evidence.metadata.topics or topic not in graph.graph.nodes:
        raise ValueError(f"book {evidence.book_id} does not belong to graph topic {topic}")
    validate_toc_mapping_rules(matching, graph)
    tree = reconstruct_toc(evidence.toc)
    aliases, exclusions = _mapping_inputs(features, graph, matching, topic)
    mappings: list[TocConceptMapping] = []
    unmapped: list[UnmappedTocEntry] = []
    matched_entry_ids: set[str] = set()
    for visit in tree.traversal:
        entry_mappings, ambiguous = _map_entry(visit, aliases, exclusions)
        mappings.extend(entry_mappings)
        if entry_mappings:
            matched_entry_ids.add(visit.entry.toc_entry_id)
        else:
            unmapped.append(
                UnmappedTocEntry(
                    toc_entry_id=visit.entry.toc_entry_id,
                    title=visit.entry.title,
                    toc_path=list(visit.path),
                    reason="ambiguous_alias" if ambiguous else "unmatched",
                )
            )
    by_concept: dict[str, list[TocConceptMapping]] = defaultdict(list)
    for mapping in mappings:
        by_concept[mapping.concept_id].append(mapping)
    covered = [
        CoveredTocConcept(
            concept_id=concept,
            coverage_weight=min(1.0, len(items) / 3.0),
            occurrence_count=len(items),
            minimum_toc_level=min(item.toc_level for item in items),
            first_traversal_position=items[0].traversal_position,
            toc_entry_ids=[item.toc_entry_id for item in items],
            toc_paths=[item.toc_path for item in items],
        )
        for concept, items in sorted(by_concept.items())
    ]
    covered_by_id = {item.concept_id: item for item in covered}
    edges = [
        (edge.prerequisite, edge.dependent, edge.source_type)
        for edge in graph.graph.edges
        if edge.topic == topic
    ]
    grouped: dict[tuple[str, str], dict[str, object]] = {}
    for target in covered:
        for ancestor, path, sources in _ancestor_paths(target.concept_id, edges):
            prior = covered_by_id.get(ancestor)
            taught = (
                prior is not None
                and prior.first_traversal_position < target.first_traversal_position
            )
            reason = "taught_before_use_candidate" if taught else "external_or_late_candidate"
            key = (ancestor, reason)
            bucket = grouped.setdefault(
                key,
                {
                    "weights": [],
                    "targets": set(),
                    "paths": set(),
                    "evidence": set(),
                    "sources": set(),
                },
            )
            bucket["weights"].append(target.coverage_weight)
            bucket["targets"].add(target.concept_id)
            bucket["paths"].add(path)
            bucket["evidence"].update(target.toc_entry_ids)
            bucket["sources"].update(sources)

    def candidates(reason: str) -> list[PrerequisiteCandidate]:
        return [
            PrerequisiteCandidate(
                concept_id=concept,
                weight=max(bucket["weights"]),
                reason=reason,
                related_target_concepts=sorted(bucket["targets"]),
                graph_paths=[list(path) for path in sorted(bucket["paths"])],
                evidence_toc_entry_ids=sorted(bucket["evidence"]),
                relation_source_types=sorted(bucket["sources"]),
            )
            for (concept, status), bucket in sorted(grouped.items())
            if status == reason
        ]

    return BookConceptProfileV2(
        book_id=evidence.book_id,
        title=evidence.metadata.title,
        topic_id=topic,
        toc_mappings=mappings,
        unmapped_entries=unmapped,
        covered_concepts=covered,
        prerequisite_requirements=candidates("external_or_late_candidate"),
        taught_before_use_candidates=candidates("taught_before_use_candidate"),
        diagnostics=TocCoverageDiagnostics(
            total_toc_entries=len(tree.traversal),
            matched_toc_entries=len(matched_entry_ids),
            unmatched_toc_entries=sum(item.reason == "unmatched" for item in unmapped),
            ambiguous_toc_entries=sum(item.reason == "ambiguous_alias" for item in unmapped),
            concept_count=len(covered),
        ),
        profile_version=matching.config.profile_version,
        graph_version=graph.graph.graph_version,
        graph_config_version=graph.graph.config_version,
        graph_hash=graph.content_hash,
        feature_config_version=features.config.config_version,
        feature_config_hash=features.content_hash,
        matching_config_version=matching.config.config_version,
        matching_config_hash=matching.content_hash,
        toc_file_hash=toc_file_hash,
    )
