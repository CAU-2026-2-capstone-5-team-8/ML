"""Transparent prose-only text difficulty baseline."""

from statistics import median

from bookmatch_ml.book.text import (
    contains_any_phrase,
    count_alias_mentions,
    sentences,
    tokenize,
)
from bookmatch_ml.config import LoadedFeatureConfig, NormalizationRange, TopicLexicon
from bookmatch_ml.data.evidence import PROSE_DOCUMENT_TYPES
from bookmatch_ml.schemas import (
    BookEvidence,
    DifficultyProfile,
    DifficultyRawFeatures,
    DifficultyScores,
    Document,
    DocumentDifficulty,
    ExcludedProseDocument,
)


def _topic_ids(evidence: BookEvidence, topics: dict[str, TopicLexicon]) -> list[str]:
    configured = set(topics)
    metadata_topics = configured.intersection(evidence.metadata.topics)
    return sorted(metadata_topics or configured)


def _normalize(value: float, limits: NormalizationRange) -> float:
    return min(1.0, max(0.0, (value - limits.minimum) / (limits.maximum - limits.minimum)))


def _weighted_score(
    raw_values: dict[str, float],
    weights: dict[str, float],
    normalization: dict[str, NormalizationRange],
) -> float:
    return sum(
        _normalize(raw_values[name], normalization[name]) * weight
        for name, weight in weights.items()
    )


def _aliases_by_concept(
    evidence: BookEvidence,
    topics: dict[str, TopicLexicon],
) -> dict[str, list[str]]:
    aliases: dict[str, list[str]] = {}
    for topic_id in _topic_ids(evidence, topics):
        aliases.update(topics[topic_id].root)
    return aliases


def _analyze_document(
    document: Document,
    evidence: BookEvidence,
    loaded_config: LoadedFeatureConfig,
) -> DocumentDifficulty:
    config = loaded_config.config
    difficulty_config = config.difficulty
    tokens = tokenize(document.text)
    sentence_values = sentences(document.text)
    sentence_token_counts = [len(tokenize(sentence)) for sentence in sentence_values]

    concepts = _aliases_by_concept(evidence, config.concept.topics)
    prerequisites = _aliases_by_concept(evidence, config.prerequisite.topics)
    concept_counts = {
        concept: count_alias_mentions(document.text, aliases)
        for concept, aliases in concepts.items()
    }
    prerequisite_counts = {
        concept: count_alias_mentions(document.text, aliases)
        for concept, aliases in prerequisites.items()
    }
    concept_mention_count = sum(concept_counts.values())
    prerequisite_mention_count = sum(prerequisite_counts.values())
    token_count = len(tokens)
    sentence_count = len(sentence_values)
    raw = DifficultyRawFeatures(
        mean_token_length=sum(len(token) for token in tokens) / token_count,
        vocabulary_diversity=len(set(tokens)) / token_count,
        long_word_ratio=sum(
            len(token) >= difficulty_config.long_word_characters for token in tokens
        )
        / token_count,
        domain_term_ratio=concept_mention_count / token_count,
        mean_sentence_tokens=sum(sentence_token_counts) / sentence_count,
        median_sentence_tokens=float(median(sentence_token_counts)),
        long_sentence_ratio=sum(
            count >= difficulty_config.long_sentence_tokens for count in sentence_token_counts
        )
        / sentence_count,
        clause_markers_per_sentence=sum(document.text.count(marker) for marker in ",;:")
        / sentence_count,
        concept_mention_count=concept_mention_count,
        unique_concept_count=sum(count > 0 for count in concept_counts.values()),
        concept_mentions_per_1000_tokens=concept_mention_count * 1000 / token_count,
        unique_concepts_per_1000_tokens=sum(count > 0 for count in concept_counts.values())
        * 1000
        / token_count,
        prerequisite_mention_count=prerequisite_mention_count,
        prerequisite_mentions_per_1000_tokens=prerequisite_mention_count * 1000 / token_count,
        prerequisite_cue_sentence_ratio=sum(
            contains_any_phrase(sentence, config.prerequisite.cue_phrases)
            for sentence in sentence_values
        )
        / sentence_count,
    )
    raw_values = raw.model_dump()
    scores = DifficultyScores(
        lexical_difficulty=_weighted_score(
            raw_values,
            difficulty_config.lexical_weights,
            difficulty_config.normalization,
        ),
        syntactic_complexity=_weighted_score(
            raw_values,
            difficulty_config.syntactic_weights,
            difficulty_config.normalization,
        ),
        concept_density=_weighted_score(
            raw_values,
            difficulty_config.concept_density_weights,
            difficulty_config.normalization,
        ),
        prerequisite_demand=_weighted_score(
            raw_values,
            difficulty_config.prerequisite_demand_weights,
            difficulty_config.normalization,
        ),
    )
    return DocumentDifficulty(
        document_id=document.document_id,
        document_type=document.document_type,
        character_count=len(document.text),
        token_count=token_count,
        sentence_count=sentence_count,
        raw=raw,
        scores=scores,
    )


def build_difficulty_profile(
    evidence: BookEvidence,
    loaded_config: LoadedFeatureConfig,
) -> DifficultyProfile:
    """Calculate per-document prose features, then aggregate by analyzed token count."""

    minimum_tokens = loaded_config.config.difficulty.minimum_document_tokens
    analyzed: list[DocumentDifficulty] = []
    excluded: list[ExcludedProseDocument] = []
    for document in evidence.documents:
        if document.document_type not in PROSE_DOCUMENT_TYPES:
            continue
        token_count = len(tokenize(document.text))
        if token_count < minimum_tokens:
            excluded.append(
                ExcludedProseDocument(
                    document_id=document.document_id,
                    document_type=document.document_type,
                    character_count=len(document.text),
                    token_count=token_count,
                    reason="no_tokens" if token_count == 0 else "below_minimum_tokens",
                )
            )
            continue
        analyzed.append(_analyze_document(document, evidence, loaded_config))

    analyzed_token_count = sum(document.token_count for document in analyzed)

    def aggregate(name: str) -> float | None:
        if not analyzed_token_count:
            return None
        return (
            sum(getattr(document.scores, name) * document.token_count for document in analyzed)
            / analyzed_token_count
        )

    config = loaded_config.config
    return DifficultyProfile(
        book_id=evidence.book_id,
        lexical_difficulty=aggregate("lexical_difficulty"),
        syntactic_complexity=aggregate("syntactic_complexity"),
        concept_density=aggregate("concept_density"),
        prerequisite_demand=aggregate("prerequisite_demand"),
        analyzed_document_count=len(analyzed),
        analyzed_character_count=sum(document.character_count for document in analyzed),
        analyzed_token_count=analyzed_token_count,
        documents=analyzed,
        excluded_documents=excluded,
        aggregation_rule=config.difficulty.aggregation_rule,
        feature_version=config.difficulty_profile_version,
        config_version=config.config_version,
        config_hash=loaded_config.content_hash,
    )
