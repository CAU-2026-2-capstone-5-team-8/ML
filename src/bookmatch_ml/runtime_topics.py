"""Local reviewed-topic registry. Never relax the fixed production ranking config."""

import hashlib
import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from bookmatch_ml.concept_v2.graph import ConceptGraph
from bookmatch_ml.config import RankingV2AcceptedEdge


class RuntimeGraph(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    contract_version: Literal["topic-runtime-graph-v1"]
    topic_id: str = Field(pattern=r"^([a-z][a-z0-9-]{1,79})$")
    source_snapshot_id: str
    content_report_hash: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    review_report_hash: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    nodes: dict[str, list[str]]
    labels: dict[str, str]
    accepted_edges: list[RankingV2AcceptedEdge]
    concept_graph_version: Literal["concept-graph-v1"]
    concept_graph_hash: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    graph_review_version: Literal["ai-concept-graph-review-v1"]
    graph_review_hash: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    reviewer_type: Literal["ai"]

    @model_validator(mode="after")
    def validate_graph(self):
        if set(self.nodes) != {self.topic_id} or set(self.labels) != set(self.nodes[self.topic_id]):
            raise ValueError("runtime graph topic/labels differ")
        if any(not c.startswith(self.topic_id + ":") for c in self.nodes[self.topic_id]):
            raise ValueError("runtime graph requires namespaced concepts")
        ConceptGraph.model_validate(
            dict(
                config_version="runtime-topic-v1",
                graph_version="concept-graph-v1",
                relation_type="prerequisite_candidate",
                nodes=self.nodes,
                edges=[
                    dict(**e.model_dump(), source_type="ai_proposed", version="runtime-topic-v1")
                    for e in self.accepted_edges
                ],
            )
        )
        return self


class LoadedRuntimeGraph(BaseModel):
    config: RuntimeGraph
    content_hash: str


def resolve_runtime_topic(
    registry: Path | None, topic: str, baseline, override_topics: frozenset[str] = frozenset()
):
    """Prefer the fixed graph unless an operator explicitly opts this topic into the registry."""
    if baseline is not None and topic in baseline.config.nodes and topic not in override_topics:
        return baseline
    if registry is None:
        return baseline
    # User topic text is not a filename. Validate before resolving any local path.
    import re

    if not re.fullmatch(r"[a-z][a-z0-9-]{1,79}", topic):
        raise ValueError("invalid runtime topic")
    pointer = registry / (topic + ".json")
    if not pointer.exists():
        return baseline
    marker = json.loads(pointer.read_text())
    if set(marker) != {"graphPath", "graphHash", "topicId"} or marker["topicId"] != topic:
        raise ValueError("invalid runtime topic pointer")
    graph_path = Path(marker["graphPath"]).resolve(strict=True)
    # Backend owns both registry and immutable artifact root. No remote fetching.
    if (
        not graph_path.is_relative_to(registry.parent.resolve())
        or graph_path.name != "runtime-graph.json"
    ):
        raise ValueError("runtime graph outside artifact root")
    raw = graph_path.read_bytes()
    digest = "sha256:" + hashlib.sha256(raw).hexdigest()
    if digest != marker["graphHash"]:
        raise ValueError("runtime graph hash differs")
    config = RuntimeGraph.model_validate_json(raw)
    if config.topic_id != topic:
        raise ValueError("runtime graph topic differs")
    return LoadedRuntimeGraph(config=config, content_hash=digest)
