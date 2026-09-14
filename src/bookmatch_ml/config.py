"""Versioned configuration models and strict YAML loading."""

import hashlib
from pathlib import Path
from typing import Literal

import yaml
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    RootModel,
    ValidationError,
    field_validator,
    model_validator,
)
from yaml.constructor import ConstructorError
from yaml.nodes import MappingNode


class ConfigError(ValueError):
    """Feature configuration is missing or invalid."""


class _UniqueKeySafeLoader(yaml.SafeLoader):
    """Load safe YAML while rejecting duplicate mapping keys."""

    def construct_mapping(self, node: MappingNode, deep: bool = False) -> dict[object, object]:
        self.flatten_mapping(node)
        mapping: dict[object, object] = {}
        for key_node, value_node in node.value:
            key = self.construct_object(key_node, deep=deep)
            try:
                duplicate = key in mapping
            except TypeError as exc:
                raise ConstructorError(
                    "while constructing a mapping",
                    node.start_mark,
                    "found unhashable key",
                    key_node.start_mark,
                ) from exc
            if duplicate:
                raise ConstructorError(
                    "while constructing a mapping",
                    node.start_mark,
                    f"found duplicate mapping key {key!r}",
                    key_node.start_mark,
                )
            mapping[key] = self.construct_object(value_node, deep=deep)
        return mapping


def _load_unique_key_yaml(content: bytes) -> object:
    """Parse trusted configuration syntax without permitting silent key overwrites."""

    return yaml.load(content, Loader=_UniqueKeySafeLoader)


class ConfigModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)


class TopicLexicon(RootModel[dict[str, list[str]]]):
    @field_validator("root")
    @classmethod
    def concepts_and_aliases_must_not_be_empty(
        cls, value: dict[str, list[str]]
    ) -> dict[str, list[str]]:
        if not value or any(
            not concept.strip() or not aliases for concept, aliases in value.items()
        ):
            raise ValueError("topic lexicons require non-empty concepts and aliases")
        if any(not alias.strip() for aliases in value.values() for alias in aliases):
            raise ValueError("concept aliases must not be blank")
        return value


class ConceptConfig(ConfigModel):
    evidence_weights: dict[str, float]
    weight_saturation: float = Field(gt=0)
    minimum_weight: float = Field(ge=0, le=1)
    topics: dict[str, TopicLexicon]

    @model_validator(mode="after")
    def evidence_weights_must_cover_all_types(self) -> "ConceptConfig":
        expected = {
            "toc",
            "description",
            "publisher_summary",
            "preface",
            "introduction",
            "preview",
            "sample_chapter",
            "index",
            "other",
        }
        if set(self.evidence_weights) != expected:
            raise ValueError(f"evidence_weights keys must be exactly: {sorted(expected)}")
        if any(weight < 0 for weight in self.evidence_weights.values()):
            raise ValueError("concept evidence weights must be non-negative")
        return self


class PrerequisiteConfig(ConfigModel):
    cue_phrases: list[str]
    early_prose_characters: int = Field(ge=0)
    explicit_weight: float = Field(ge=0)
    early_proxy_weight: float = Field(ge=0)
    weight_saturation: float = Field(gt=0)
    minimum_weight: float = Field(ge=0, le=1)
    topics: dict[str, TopicLexicon]

    @field_validator("cue_phrases")
    @classmethod
    def cue_phrases_must_not_be_empty(cls, value: list[str]) -> list[str]:
        if not value or any(not phrase.strip() for phrase in value):
            raise ValueError("prerequisite cue phrases must not be empty or blank")
        return value


class NormalizationRange(ConfigModel):
    minimum: float
    maximum: float

    @model_validator(mode="after")
    def maximum_must_exceed_minimum(self) -> "NormalizationRange":
        if self.maximum <= self.minimum:
            raise ValueError("maximum must be greater than minimum")
        return self


class DifficultyConfig(ConfigModel):
    minimum_document_tokens: int = Field(ge=1)
    long_word_characters: int = Field(ge=1)
    long_sentence_tokens: int = Field(ge=1)
    normalization: dict[str, NormalizationRange]
    lexical_weights: dict[str, float]
    syntactic_weights: dict[str, float]
    concept_density_weights: dict[str, float]
    prerequisite_demand_weights: dict[str, float]
    aggregation_rule: str = Field(min_length=1)

    @model_validator(mode="after")
    def weights_must_be_normalized(self) -> "DifficultyConfig":
        groups = (
            (
                "lexical_weights",
                self.lexical_weights,
                {"mean_token_length", "long_word_ratio", "domain_term_ratio"},
            ),
            (
                "syntactic_weights",
                self.syntactic_weights,
                {
                    "mean_sentence_tokens",
                    "long_sentence_ratio",
                    "clause_markers_per_sentence",
                },
            ),
            (
                "concept_density_weights",
                self.concept_density_weights,
                {"concept_mentions_per_1000_tokens", "unique_concepts_per_1000_tokens"},
            ),
            (
                "prerequisite_demand_weights",
                self.prerequisite_demand_weights,
                {
                    "prerequisite_mentions_per_1000_tokens",
                    "prerequisite_cue_sentence_ratio",
                },
            ),
        )
        required_normalizers: set[str] = set()
        for name, group, expected in groups:
            if set(group) != expected:
                raise ValueError(f"{name} keys must be exactly: {sorted(expected)}")
            if abs(sum(group.values()) - 1.0) > 1e-9:
                raise ValueError(f"{name} must sum to 1.0")
            if any(weight < 0 for weight in group.values()):
                raise ValueError(f"{name} must contain non-negative weights")
            required_normalizers.update(expected)
        if set(self.normalization) != required_normalizers:
            raise ValueError(f"normalization keys must be exactly: {sorted(required_normalizers)}")
        return self


class FeatureConfig(ConfigModel):
    config_version: str = Field(min_length=1)
    concept_profile_version: str = Field(min_length=1)
    difficulty_profile_version: str = Field(min_length=1)
    book_profile_version: str = Field(min_length=1)
    concept: ConceptConfig
    prerequisite: PrerequisiteConfig
    difficulty: DifficultyConfig


class LoadedFeatureConfig(ConfigModel):
    config: FeatureConfig
    content_hash: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")


class ReaderConfig(ConfigModel):
    config_version: str = Field(min_length=1)
    profile_version: str = Field(min_length=1)
    difficulty_weights: dict[str, float]
    required_question_types: list[str]
    minimum_responses_per_type: int = Field(ge=1)

    @model_validator(mode="after")
    def scoring_contract_must_be_complete(self) -> "ReaderConfig":
        expected_types = {"vocabulary", "background_knowledge", "comprehension"}
        if set(self.required_question_types) != expected_types:
            raise ValueError(f"required_question_types must be exactly: {sorted(expected_types)}")
        if len(self.required_question_types) != len(expected_types):
            raise ValueError("required_question_types must not contain duplicates")
        if not self.difficulty_weights:
            raise ValueError("difficulty_weights must not be empty")
        if any(
            not label.strip() or weight <= 0 for label, weight in self.difficulty_weights.items()
        ):
            raise ValueError("difficulty weights require non-blank labels and positive values")
        return self


class LoadedReaderConfig(ConfigModel):
    config: ReaderConfig
    content_hash: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")


class ExplanationConfig(ConfigModel):
    close_gap: float = Field(ge=0, le=1)
    moderate_gap: float = Field(ge=0, le=1)
    topic_match: float = Field(ge=0, le=1)
    maximum_covered_concepts: int = Field(ge=0)

    @model_validator(mode="after")
    def moderate_gap_must_not_be_smaller(self) -> "ExplanationConfig":
        if self.moderate_gap < self.close_gap:
            raise ValueError("moderate_gap must be greater than or equal to close_gap")
        return self


class RankingConfig(ConfigModel):
    config_version: str = Field(min_length=1)
    model_version: str = Field(min_length=1)
    fit_strategy: Literal["absolute_gap_v1"]
    filter_topic_candidates: bool
    component_weights: dict[str, float]
    knowledge_weights: dict[str, float]
    explanation: ExplanationConfig

    @model_validator(mode="after")
    def weights_must_be_complete_and_normalized(self) -> "RankingConfig":
        groups = (
            (
                "component_weights",
                self.component_weights,
                {"topic_fit", "vocabulary_fit", "knowledge_fit", "comprehension_fit"},
            ),
            (
                "knowledge_weights",
                self.knowledge_weights,
                {
                    "concept_density_fit",
                    "prerequisite_demand_fit",
                    "prerequisite_concept_fit",
                },
            ),
        )
        for name, weights, expected in groups:
            if set(weights) != expected:
                raise ValueError(f"{name} keys must be exactly: {sorted(expected)}")
            if any(weight < 0 for weight in weights.values()):
                raise ValueError(f"{name} must contain non-negative weights")
            if abs(sum(weights.values()) - 1.0) > 1e-9:
                raise ValueError(f"{name} must sum to 1.0")
        return self


class LoadedRankingConfig(ConfigModel):
    config: RankingConfig
    content_hash: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")


class DifficultyEvaluationConfig(ConfigModel):
    component_weights: dict[str, float]
    minimum_comparable_books: int = Field(ge=2)
    small_sample_warning_threshold: int = Field(ge=2)

    @model_validator(mode="after")
    def weights_must_be_complete_and_normalized(self) -> "DifficultyEvaluationConfig":
        expected = {
            "lexical_difficulty",
            "syntactic_complexity",
            "concept_density",
            "prerequisite_demand",
        }
        if set(self.component_weights) != expected:
            raise ValueError(
                f"difficulty component_weights keys must be exactly: {sorted(expected)}"
            )
        if any(weight < 0 for weight in self.component_weights.values()):
            raise ValueError("difficulty component_weights must contain non-negative weights")
        if abs(sum(self.component_weights.values()) - 1.0) > 1e-9:
            raise ValueError("difficulty component_weights must sum to 1.0")
        if self.small_sample_warning_threshold < self.minimum_comparable_books:
            raise ValueError(
                "small_sample_warning_threshold must be at least minimum_comparable_books"
            )
        return self


class RecommendationEvaluationConfig(ConfigModel):
    top_k: int = Field(ge=1)
    topic_only_version: str = Field(min_length=1)


class EvaluationConfig(ConfigModel):
    config_version: str = Field(min_length=1)
    evaluation_version: str = Field(min_length=1)
    difficulty: DifficultyEvaluationConfig
    recommendation: RecommendationEvaluationConfig


class LoadedEvaluationConfig(ConfigModel):
    config: EvaluationConfig
    content_hash: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")


def load_feature_config(path: Path) -> LoadedFeatureConfig:
    """Load strict YAML and retain a hash so results identify exact configuration bytes."""

    try:
        content = path.read_bytes()
    except OSError as exc:
        raise ConfigError(f"cannot read feature config: {path}: {exc}") from exc
    try:
        payload = _load_unique_key_yaml(content)
        config = FeatureConfig.model_validate(payload)
    except (yaml.YAMLError, ValidationError, ValueError) as exc:
        raise ConfigError(f"invalid feature config: {path}: {exc}") from exc
    return LoadedFeatureConfig(
        config=config,
        content_hash=f"sha256:{hashlib.sha256(content).hexdigest()}",
    )


def load_reader_config(path: Path) -> LoadedReaderConfig:
    """Load strict reader-scoring YAML and retain its exact content hash."""

    try:
        content = path.read_bytes()
    except OSError as exc:
        raise ConfigError(f"cannot read reader config: {path}: {exc}") from exc
    try:
        payload = _load_unique_key_yaml(content)
        config = ReaderConfig.model_validate(payload)
    except (yaml.YAMLError, ValidationError, ValueError) as exc:
        raise ConfigError(f"invalid reader config: {path}: {exc}") from exc
    return LoadedReaderConfig(
        config=config,
        content_hash=f"sha256:{hashlib.sha256(content).hexdigest()}",
    )


def load_ranking_config(path: Path) -> LoadedRankingConfig:
    """Load strict ranking YAML and retain its exact content hash."""

    try:
        content = path.read_bytes()
    except OSError as exc:
        raise ConfigError(f"cannot read ranking config: {path}: {exc}") from exc
    try:
        payload = _load_unique_key_yaml(content)
        config = RankingConfig.model_validate(payload)
    except (yaml.YAMLError, ValidationError, ValueError) as exc:
        raise ConfigError(f"invalid ranking config: {path}: {exc}") from exc
    return LoadedRankingConfig(
        config=config,
        content_hash=f"sha256:{hashlib.sha256(content).hexdigest()}",
    )


def load_evaluation_config(path: Path) -> LoadedEvaluationConfig:
    """Load strict evaluation YAML and retain its exact content hash."""

    try:
        content = path.read_bytes()
    except OSError as exc:
        raise ConfigError(f"cannot read evaluation config: {path}: {exc}") from exc
    try:
        payload = _load_unique_key_yaml(content)
        config = EvaluationConfig.model_validate(payload)
    except (yaml.YAMLError, ValidationError, ValueError) as exc:
        raise ConfigError(f"invalid evaluation config: {path}: {exc}") from exc
    return LoadedEvaluationConfig(
        config=config,
        content_hash=f"sha256:{hashlib.sha256(content).hexdigest()}",
    )
