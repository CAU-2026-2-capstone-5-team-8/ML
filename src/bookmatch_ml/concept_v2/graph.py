"""Small, versioned prerequisite-candidate DAG independent of book TOCs."""

import hashlib
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from bookmatch_ml.config import (
    ConfigError,
    LoadedFeatureConfig,
    _load_unique_key_yaml,
)


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class GraphEdge(_Strict):
    topic: str = Field(min_length=1)
    prerequisite: str = Field(min_length=1)
    dependent: str = Field(min_length=1)
    relation_type: Literal["prerequisite_candidate"] = "prerequisite_candidate"
    source_type: Literal["proposed_seed", "human_reviewed_seed"]
    version: str = Field(min_length=1)


class ConceptGraph(_Strict):
    config_version: str = Field(min_length=1)
    graph_version: str = Field(min_length=1)
    relation_type: Literal["prerequisite_candidate"]
    nodes: dict[str, list[str]]
    edges: list[GraphEdge]

    @model_validator(mode="after")
    def validate_dag(self) -> "ConceptGraph":
        if not self.nodes or any(
            not topic.strip()
            or not nodes
            or any(not node.strip() for node in nodes)
            or len(nodes) != len(set(nodes))
            for topic, nodes in self.nodes.items()
        ):
            raise ValueError("graph topics require distinct nodes")
        membership = {node: topic for topic, nodes in self.nodes.items() for node in nodes}
        if len(membership) != sum(map(len, self.nodes.values())):
            raise ValueError("graph node belongs to multiple topics")
        seen: set[tuple[str, str, str]] = set()
        adjacency: dict[str, list[str]] = {node: [] for node in membership}
        for edge in self.edges:
            if edge.prerequisite not in membership or edge.dependent not in membership:
                raise ValueError("graph edge references unknown node")
            if edge.prerequisite == edge.dependent:
                raise ValueError("graph self-loop")
            if (
                membership[edge.prerequisite] != edge.topic
                or membership[edge.dependent] != edge.topic
            ):
                raise ValueError("graph edge mixes topics")
            key = (edge.topic, edge.prerequisite, edge.dependent)
            if key in seen:
                raise ValueError("duplicate graph edge")
            seen.add(key)
            adjacency[edge.prerequisite].append(edge.dependent)
        visited: set[str] = set()
        active: set[str] = set()

        def visit(node: str) -> None:
            if node in active:
                raise ValueError("graph prerequisite cycle")
            if node in visited:
                return
            active.add(node)
            for dependent in sorted(adjacency[node]):
                visit(dependent)
            active.remove(node)
            visited.add(node)

        for node in sorted(membership):
            visit(node)
        return self


class LoadedConceptGraph(_Strict):
    graph: ConceptGraph
    content_hash: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")


def load_concept_graph(path: Path, features: LoadedFeatureConfig) -> LoadedConceptGraph:
    """Reject graph nodes outside the existing topic lexicons and hash exact config bytes."""

    try:
        content = path.read_bytes()
        graph = ConceptGraph.model_validate(_load_unique_key_yaml(content))
        if set(graph.nodes) != set(features.config.concept.topics):
            raise ValueError("graph topics differ from feature concept topics")
        for topic, nodes in graph.nodes.items():
            available = set(features.config.concept.topics[topic].root)
            prerequisite_topic = features.config.prerequisite.topics.get(topic)
            if prerequisite_topic is None:
                raise ValueError(f"feature prerequisite aliases missing topic: {topic}")
            available.update(prerequisite_topic.root)
            unknown = set(nodes) - available
            if unknown:
                raise ValueError(f"graph nodes missing from feature aliases: {sorted(unknown)}")
    except (OSError, ValidationError, ValueError) as exc:
        raise ConfigError(f"invalid concept graph: {path}: {exc}") from exc
    return LoadedConceptGraph(
        graph=graph, content_hash=f"sha256:{hashlib.sha256(content).hexdigest()}"
    )
