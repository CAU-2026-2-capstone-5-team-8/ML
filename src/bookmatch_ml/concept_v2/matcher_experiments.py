"""Versioned deterministic matcher experiments evaluated against frozen gold."""

import re
from collections import defaultdict

from pydantic import BaseModel, ConfigDict, Field

from bookmatch_ml.book.text import normalize_text
from bookmatch_ml.concept_v2.graph import LoadedConceptGraph
from bookmatch_ml.concept_v2.profile import (
    LoadedConceptMatchingConfig,
    _mapping_inputs,
    validate_toc_mapping_rules,
)
from bookmatch_ml.config import LoadedFeatureConfig


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class ExperimentalConceptMatch(_Strict):
    """One span-backed concept match emitted by an experiment variant."""

    concept_id: str
    matching_alias: str
    match_method: str
    span_start: int = Field(ge=0)
    span_end: int = Field(gt=0)


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
    """Suppress only shorter cross-concept spans contained in a longer match."""

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
    matches, ambiguous = _span_matches(
        text,
        aliases,
        exclusions,
        "normalized_alias_span_v2_a",
    )
    if ambiguous:
        return [], True
    return _one_match_per_concept(_suppress_nested_overlaps(matches)), False
