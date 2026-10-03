"""Inspectable lexicon-based concept and prerequisite extraction."""

from collections import defaultdict

from bookmatch_ml.book.text import contains_any_phrase, count_alias_mentions, sentences
from bookmatch_ml.config import LoadedFeatureConfig, TopicLexicon
from bookmatch_ml.data.evidence import PROSE_DOCUMENT_TYPES
from bookmatch_ml.schemas import (
    BookEvidence,
    ConceptEvidenceRef,
    ConceptProfile,
    CoveredConcept,
    PrerequisiteConcept,
)


def _topic_ids(evidence: BookEvidence, topics: dict[str, TopicLexicon]) -> list[str]:
    configured = set(topics)
    metadata_topics = configured.intersection(evidence.metadata.topics)
    return sorted(metadata_topics or configured)


def _concept_evidence(
    evidence: BookEvidence,
    aliases: list[str],
) -> list[ConceptEvidenceRef]:
    references: list[ConceptEvidenceRef] = []
    for entry in evidence.toc:
        mentions = count_alias_mentions(entry.title, aliases)
        if mentions:
            references.append(
                ConceptEvidenceRef(
                    evidence_type="toc",
                    evidence_id=entry.toc_entry_id,
                    mention_count=mentions,
                )
            )
    for document in evidence.documents:
        mentions = count_alias_mentions(document.text, aliases)
        if mentions:
            references.append(
                ConceptEvidenceRef(
                    evidence_type=document.document_type,
                    evidence_id=document.document_id,
                    mention_count=mentions,
                )
            )
    return sorted(
        references,
        key=lambda item: (item.evidence_type, item.evidence_id),
    )


def _covered_concepts(
    evidence: BookEvidence,
    loaded_config: LoadedFeatureConfig,
) -> tuple[list[CoveredConcept], dict[str, float]]:
    config = loaded_config.config.concept
    concepts: list[CoveredConcept] = []
    topic_strength: dict[str, float] = {}

    for topic_id in _topic_ids(evidence, config.topics):
        topic_concepts: list[CoveredConcept] = []
        for concept, aliases in sorted(config.topics[topic_id].root.items()):
            references = _concept_evidence(evidence, aliases)
            weighted_mentions = sum(
                config.evidence_weights[reference.evidence_type] * reference.mention_count
                for reference in references
            )
            weight = min(1.0, weighted_mentions / config.weight_saturation)
            if weight >= config.minimum_weight:
                topic_concepts.append(
                    CoveredConcept(
                        concept=concept,
                        weight=weight,
                        evidence_types=sorted(
                            {reference.evidence_type for reference in references}
                        ),
                        evidence=references,
                    )
                )
        topic_strength[topic_id] = sum(concept.weight for concept in topic_concepts) / len(
            config.topics[topic_id].root
        )
        concepts.extend(topic_concepts)

    concepts.sort(key=lambda item: (-item.weight, item.concept))
    metadata_topics = set(evidence.metadata.topics).intersection(config.topics)
    if metadata_topics:
        distribution = {
            topic_id: 1.0 / len(metadata_topics) for topic_id in sorted(metadata_topics)
        }
    else:
        total_strength = sum(topic_strength.values())
        distribution = (
            {
                topic_id: strength / total_strength
                for topic_id, strength in sorted(topic_strength.items())
                if strength > 0
            }
            if total_strength
            else {}
        )
    return concepts, distribution


def _prerequisite_concepts(
    evidence: BookEvidence,
    loaded_config: LoadedFeatureConfig,
) -> list[PrerequisiteConcept]:
    config = loaded_config.config.prerequisite
    results: list[PrerequisiteConcept] = []

    for topic_id in _topic_ids(evidence, config.topics):
        for concept, aliases in sorted(config.topics[topic_id].root.items()):
            explicit_counts: dict[tuple[str, str], int] = defaultdict(int)
            early_counts: dict[tuple[str, str], int] = defaultdict(int)
            for document in evidence.documents:
                for sentence in sentences(document.text):
                    if contains_any_phrase(sentence, config.cue_phrases):
                        mentions = count_alias_mentions(sentence, aliases)
                        if mentions:
                            explicit_counts[(document.document_type, document.document_id)] += (
                                mentions
                            )
                if document.document_type in PROSE_DOCUMENT_TYPES:
                    early_text = document.text[: config.early_prose_characters]
                    mentions = count_alias_mentions(early_text, aliases)
                    if mentions:
                        early_counts[(document.document_type, document.document_id)] += mentions

            weighted_mentions = (
                sum(explicit_counts.values()) * config.explicit_weight
                + sum(early_counts.values()) * config.early_proxy_weight
            )
            weight = min(1.0, weighted_mentions / config.weight_saturation)
            if weight < config.minimum_weight:
                continue
            if explicit_counts and early_counts:
                method = "explicit_and_early_prose_proxy"
            elif explicit_counts:
                method = "explicit"
            else:
                method = "early_prose_proxy"

            combined_counts = defaultdict(int)
            for key, count in explicit_counts.items():
                combined_counts[key] = max(combined_counts[key], count)
            for key, count in early_counts.items():
                combined_counts[key] = max(combined_counts[key], count)
            references = [
                ConceptEvidenceRef(
                    evidence_type=evidence_type,
                    evidence_id=evidence_id,
                    mention_count=count,
                )
                for (evidence_type, evidence_id), count in sorted(combined_counts.items())
            ]
            results.append(
                PrerequisiteConcept(
                    concept=concept,
                    weight=weight,
                    method=method,
                    evidence=references,
                )
            )

    return sorted(results, key=lambda item: (-item.weight, item.concept))


def build_concept_profile(
    evidence: BookEvidence,
    loaded_config: LoadedFeatureConfig,
) -> ConceptProfile:
    """Build a deterministic concept profile while retaining used evidence identities."""

    covered_concepts, topic_distribution = _covered_concepts(evidence, loaded_config)
    return ConceptProfile(
        book_id=evidence.book_id,
        covered_concepts=covered_concepts,
        prerequisite_concepts=_prerequisite_concepts(evidence, loaded_config),
        topic_distribution=topic_distribution,
        analyzed_toc_entry_ids=[entry.toc_entry_id for entry in evidence.toc],
        analyzed_document_ids=[document.document_id for document in evidence.documents],
        profile_version=loaded_config.config.concept_profile_version,
        config_version=loaded_config.config.config_version,
        config_hash=loaded_config.content_hash,
    )
