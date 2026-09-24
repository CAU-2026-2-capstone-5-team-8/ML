"""Versioned deterministic matcher experiments evaluated against frozen gold."""

import hashlib
import re
from collections import defaultdict
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from bookmatch_ml.book.text import normalize_text
from bookmatch_ml.concept_v2.graph import LoadedConceptGraph
from bookmatch_ml.concept_v2.profile import (
    LoadedConceptMatchingConfig,
    _mapping_inputs,
    match_alias_spans_v2,
    validate_toc_mapping_rules,
)
from bookmatch_ml.config import ConfigError, LoadedFeatureConfig, _load_unique_key_yaml

GENERIC_PATH_LEAVES = frozenset(
    {
        "appendix",
        "bibliography",
        "exercises",
        "introduction",
        "notes",
        "preface",
        "problems",
        "references",
        "review questions",
        "summary",
    }
)


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class ExperimentalConceptMatch(_Strict):
    """One span-backed concept match emitted by an experiment variant."""

    concept_id: str
    matching_alias: str
    match_method: str
    span_start: int = Field(ge=0)
    span_end: int = Field(gt=0)


class ModifierTolerantPattern(_Strict):
    phrase: str
    max_gap_tokens: int = Field(ge=1, le=3)


class ContextExclusionRule(_Strict):
    aliases: list[str]
    evidence_types: list[str]
    usage_cues: list[str]
    learning_cues: list[str]

    @model_validator(mode="after")
    def values_are_non_empty(self) -> "ContextExclusionRule":
        values = (self.aliases, self.evidence_types, self.usage_cues, self.learning_cues)
        if any(not items or any(not normalize_text(item) for item in items) for items in values):
            raise ValueError("context exclusion rules require non-empty values")
        return self


class MatcherV2ExperimentConfig(_Strict):
    config_version: str
    alias_additions: dict[str, dict[str, list[str]]] = Field(default_factory=dict)
    modifier_tolerant_patterns: dict[str, dict[str, list[ModifierTolerantPattern]]] = Field(
        default_factory=dict
    )
    context_exclusions: dict[str, dict[str, ContextExclusionRule]] = Field(default_factory=dict)


class LoadedMatcherV2ExperimentConfig(_Strict):
    config: MatcherV2ExperimentConfig
    content_hash: str


def load_matcher_v2_experiment_config(path: Path) -> LoadedMatcherV2ExperimentConfig:
    """Load the isolated matcher-v2 experiment policy without changing v1 config."""

    try:
        content = path.read_bytes()
        config = MatcherV2ExperimentConfig.model_validate(_load_unique_key_yaml(content))
    except (OSError, ValidationError, ValueError) as exc:
        raise ConfigError(f"invalid matcher-v2 experiment config: {path}: {exc}") from exc
    return LoadedMatcherV2ExperimentConfig(
        config=config,
        content_hash=f"sha256:{hashlib.sha256(content).hexdigest()}",
    )


def validate_matcher_v2_experiment_config(
    experiment: LoadedMatcherV2ExperimentConfig,
    graph: LoadedConceptGraph,
) -> None:
    """Reject experiment rules that do not reference canonical graph concepts."""

    sections = (
        experiment.config.alias_additions,
        experiment.config.modifier_tolerant_patterns,
        experiment.config.context_exclusions,
    )
    for section in sections:
        unknown_topics = set(section) - set(graph.graph.nodes)
        if unknown_topics:
            raise ConfigError(
                f"matcher-v2 rules reference unknown topics: {sorted(unknown_topics)}"
            )
        for topic, concepts in section.items():
            unknown_concepts = set(concepts) - set(graph.graph.nodes[topic])
            if unknown_concepts:
                raise ConfigError(
                    f"matcher-v2 rules reference unknown {topic} concepts: "
                    f"{sorted(unknown_concepts)}"
                )


def _excluded_concepts(
    normalized_text: str,
    exclusions: dict[str, list[str]],
) -> set[str]:
    return {
        concept
        for concept, phrases in exclusions.items()
        if any(
            normalized and re.search(rf"(?<!\w){re.escape(normalized)}(?!\w)", normalized_text)
            for phrase in phrases
            if (normalized := normalize_text(phrase))
        )
    }


def _span_matches(
    text: str,
    aliases: dict[str, list[str]],
    exclusions: dict[str, list[str]],
    match_method: str,
) -> tuple[list[ExperimentalConceptMatch], bool]:
    normalized_text = normalize_text(text)
    excluded = _excluded_concepts(normalized_text, exclusions)
    matches: list[ExperimentalConceptMatch] = []
    matched_alias_concepts: dict[str, set[str]] = defaultdict(set)
    seen: set[tuple[str, str, int, int]] = set()
    for concept, forms in aliases.items():
        if concept in excluded:
            continue
        for form in forms:
            normalized_alias = normalize_text(form)
            if not normalized_alias:
                continue
            for occurrence in re.finditer(
                rf"(?<!\w){re.escape(normalized_alias)}(?!\w)", normalized_text
            ):
                key = (concept, normalized_alias, occurrence.start(), occurrence.end())
                if key in seen:
                    continue
                seen.add(key)
                matched_alias_concepts[normalized_alias].add(concept)
                matches.append(
                    ExperimentalConceptMatch(
                        concept_id=concept,
                        matching_alias=form,
                        match_method=match_method,
                        span_start=occurrence.start(),
                        span_end=occurrence.end(),
                    )
                )
    if any(len(concepts) > 1 for concepts in matched_alias_concepts.values()):
        return [], True
    return matches, False


def _suppress_nested_overlaps(
    matches: list[ExperimentalConceptMatch],
) -> list[ExperimentalConceptMatch]:
    """Suppress a nested span only when its concept is part of the longer concept."""

    ordered = sorted(
        matches,
        key=lambda match: (
            -(match.span_end - match.span_start),
            match.span_start,
            match.concept_id,
            normalize_text(match.matching_alias),
        ),
    )
    kept: list[ExperimentalConceptMatch] = []
    for candidate in ordered:
        nested = any(
            existing.concept_id != candidate.concept_id
            and existing.span_start <= candidate.span_start
            and candidate.span_end <= existing.span_end
            and (existing.span_end - existing.span_start)
            > (candidate.span_end - candidate.span_start)
            and _contains_phrase(normalize_text(existing.concept_id), candidate.concept_id)
            for existing in kept
        )
        if not nested:
            kept.append(candidate)
    return kept


def _one_match_per_concept(
    matches: list[ExperimentalConceptMatch],
) -> list[ExperimentalConceptMatch]:
    by_concept: dict[str, list[ExperimentalConceptMatch]] = defaultdict(list)
    for match in matches:
        by_concept[match.concept_id].append(match)
    return [
        sorted(
            concept_matches,
            key=lambda match: (
                -(match.span_end - match.span_start),
                match.span_start,
                normalize_text(match.matching_alias),
            ),
        )[0]
        for _, concept_matches in sorted(by_concept.items())
    ]


def _aliases_with_experiment_additions(
    aliases: dict[str, list[str]],
    experiment: LoadedMatcherV2ExperimentConfig,
    topic: str,
) -> dict[str, list[str]]:
    merged = {concept: list(forms) for concept, forms in aliases.items()}
    for concept, additions in experiment.config.alias_additions.get(topic, {}).items():
        merged[concept] = list(dict.fromkeys([*merged[concept], *additions]))
    return merged


def _token_spans(text: str) -> list[tuple[str, int, int]]:
    return [
        (match.group(), match.start(), match.end())
        for match in re.finditer(r"\w+", normalize_text(text))
    ]


def _modifier_tolerant_matches(
    text: str,
    patterns: dict[str, list[ModifierTolerantPattern]],
) -> list[ExperimentalConceptMatch]:
    tokens = _token_spans(text)
    matches: list[ExperimentalConceptMatch] = []
    for concept, concept_patterns in patterns.items():
        for pattern in concept_patterns:
            expected = [token for token, _, _ in _token_spans(pattern.phrase)]
            if not expected:
                continue
            for start_index, (token, span_start, _) in enumerate(tokens):
                if token != expected[0]:
                    continue
                cursor = start_index
                span_end = tokens[start_index][2]
                for expected_token in expected[1:]:
                    next_index = next(
                        (
                            index
                            for index in range(
                                cursor + 1,
                                min(len(tokens), cursor + pattern.max_gap_tokens + 2),
                            )
                            if tokens[index][0] == expected_token
                        ),
                        None,
                    )
                    if next_index is None:
                        break
                    cursor = next_index
                    span_end = tokens[cursor][2]
                else:
                    matches.append(
                        ExperimentalConceptMatch(
                            concept_id=concept,
                            matching_alias=pattern.phrase,
                            match_method="modifier_tolerant_alias_v2_b",
                            span_start=span_start,
                            span_end=span_end,
                        )
                    )
    return matches


def _contains_phrase(text: str, phrase: str) -> bool:
    normalized = normalize_text(phrase)
    return bool(normalized and re.search(rf"(?<!\w){re.escape(normalized)}(?!\w)", text))


def _apply_context_exclusions(
    matches: list[ExperimentalConceptMatch],
    text: str,
    evidence_type: str,
    rules: dict[str, ContextExclusionRule],
) -> list[ExperimentalConceptMatch]:
    normalized_text = normalize_text(text)
    kept: list[ExperimentalConceptMatch] = []
    for match in matches:
        rule = rules.get(match.concept_id)
        applies = (
            rule is not None
            and evidence_type in rule.evidence_types
            and normalize_text(match.matching_alias)
            in {normalize_text(alias) for alias in rule.aliases}
        )
        usage_context = applies and any(
            _contains_phrase(normalized_text, cue) for cue in rule.usage_cues
        )
        learning_context = applies and any(
            _contains_phrase(normalized_text, cue) for cue in rule.learning_cues
        )
        if not (usage_context and not learning_context):
            kept.append(match)
    return kept


def match_concept_text_v2_a(
    text: str,
    topic: str,
    features: LoadedFeatureConfig,
    graph: LoadedConceptGraph,
    matching: LoadedConceptMatchingConfig,
) -> tuple[list[ExperimentalConceptMatch], bool]:
    """Run v1 aliases with general longest-span overlap suppression."""

    if topic not in graph.graph.nodes:
        raise ValueError(f"unsupported graph topic: {topic}")
    validate_toc_mapping_rules(matching, graph)
    aliases, exclusions = _mapping_inputs(features, graph, matching, topic)
    matches, ambiguous = match_alias_spans_v2(text, aliases, exclusions)
    if ambiguous:
        return [], True
    return (
        [
            ExperimentalConceptMatch(
                concept_id=match.concept_id,
                matching_alias=match.matching_alias,
                match_method="normalized_alias_span_v2_a",
                span_start=match.span_start,
                span_end=match.span_end,
            )
            for match in matches
        ],
        False,
    )


def match_concept_text_v2_b(
    text: str,
    topic: str,
    evidence_type: str,
    features: LoadedFeatureConfig,
    graph: LoadedConceptGraph,
    matching: LoadedConceptMatchingConfig,
    experiment: LoadedMatcherV2ExperimentConfig,
) -> tuple[list[ExperimentalConceptMatch], bool]:
    """Add reviewed forms and bounded modifier matching on top of v2-A."""

    if topic not in graph.graph.nodes:
        raise ValueError(f"unsupported graph topic: {topic}")
    validate_toc_mapping_rules(matching, graph)
    validate_matcher_v2_experiment_config(experiment, graph)
    aliases, exclusions = _mapping_inputs(features, graph, matching, topic)
    aliases = _aliases_with_experiment_additions(aliases, experiment, topic)
    exact, ambiguous = _span_matches(
        text,
        aliases,
        exclusions,
        "normalized_alias_span_v2_b",
    )
    if ambiguous:
        return [], True
    tolerant = _modifier_tolerant_matches(
        text,
        experiment.config.modifier_tolerant_patterns.get(topic, {}),
    )
    contextual = _apply_context_exclusions(
        [*exact, *tolerant],
        text,
        evidence_type,
        experiment.config.context_exclusions.get(topic, {}),
    )
    return _one_match_per_concept(_suppress_nested_overlaps(contextual)), False


def _path_components(text: str, toc_path: list[str] | None) -> list[str]:
    if not toc_path:
        return [text]
    components = list(toc_path)
    if normalize_text(components[-1]) != normalize_text(text):
        components.append(text)
    return components


def _merge_component_matches(
    matches: list[ExperimentalConceptMatch],
) -> list[ExperimentalConceptMatch]:
    """Deduplicate path-component matches by concept without comparing local spans."""

    return _one_match_per_concept(matches)


def match_concept_toc_v2_c1(
    text: str,
    toc_path: list[str] | None,
    topic: str,
    evidence_type: str,
    features: LoadedFeatureConfig,
    graph: LoadedConceptGraph,
    matching: LoadedConceptMatchingConfig,
    experiment: LoadedMatcherV2ExperimentConfig,
) -> tuple[list[ExperimentalConceptMatch], bool]:
    """Diagnostic only: union every match from the leaf and all TOC parents."""

    all_matches: list[ExperimentalConceptMatch] = []
    ambiguous = False
    for component in _path_components(text, toc_path):
        component_matches, component_ambiguous = match_concept_text_v2_b(
            component,
            topic,
            evidence_type,
            features,
            graph,
            matching,
            experiment,
        )
        all_matches.extend(component_matches)
        ambiguous = ambiguous or component_ambiguous
    if ambiguous:
        return [], True
    return _merge_component_matches(all_matches), False


def match_concept_toc_v2_c2(
    text: str,
    toc_path: list[str] | None,
    topic: str,
    evidence_type: str,
    features: LoadedFeatureConfig,
    graph: LoadedConceptGraph,
    matching: LoadedConceptMatchingConfig,
    experiment: LoadedMatcherV2ExperimentConfig,
) -> tuple[list[ExperimentalConceptMatch], bool]:
    """Keep leaf matches primary; otherwise use only the nearest matching parent."""

    leaf_matches, ambiguous = match_concept_text_v2_b(
        text,
        topic,
        evidence_type,
        features,
        graph,
        matching,
        experiment,
    )
    if ambiguous or leaf_matches:
        return leaf_matches, ambiguous
    if normalize_text(text) in GENERIC_PATH_LEAVES:
        return [], False
    components = _path_components(text, toc_path)
    parents = components[:-1]
    for parent in reversed(parents):
        parent_matches, parent_ambiguous = match_concept_text_v2_b(
            parent,
            topic,
            evidence_type,
            features,
            graph,
            matching,
            experiment,
        )
        if parent_ambiguous:
            return [], True
        if parent_matches:
            return [
                match.model_copy(update={"match_method": "parent_context_v2_c2"})
                for match in parent_matches
            ], False
    return [], False
