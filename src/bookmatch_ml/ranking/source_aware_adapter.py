"""Compatibility adapter from source-aware concept presence to ranking inputs."""

from collections import Counter
from pathlib import Path
from typing import Literal

from pydantic import Field, ValidationError

from bookmatch_ml.concept_v2.book_evidence_mapping import (
    BookEvidenceConceptMappingReport,
)
from bookmatch_ml.schemas import (
    BookProfile,
    MatchingBookProfile,
    MatchingConcept,
    StrictModel,
)

ADAPTER_VERSION = "source-aware-matching-profile-adapter-v1"
BINARY_PRESENCE_WEIGHT = 1.0


class SourceAwareAdapterError(ValueError):
    """Source-aware mappings and legacy profiles cannot be joined safely."""


class DifficultyAvailability(StrictModel):
    """Availability counts copied from existing difficulty profiles."""

    available: int = Field(ge=0)
    missing: int = Field(ge=0)


class SourceAwareMatchingAdapterReport(StrictModel):
    """Reproducibility and coverage metadata for one adapter artifact."""

    adapter_version: Literal["source-aware-matching-profile-adapter-v1"]
    concept_mapping_report_version: str
    concept_mapping_hash: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    matcher_version: str
    matching_config_version: str
    matching_config_hash: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    source_book_profiles_hash: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    candidate_artifact_hash: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    source_feature_versions: list[str]
    source_config_versions: list[str]
    source_config_hashes: list[str]
    mapping_book_count: int = Field(ge=0)
    source_book_profile_count: int = Field(ge=0)
    joined_book_count: int = Field(ge=0)
    missing_in_mapping: list[str]
    missing_in_book_profiles: list[str]
    duplicate_mapping_book_ids: list[str]
    duplicate_book_profile_ids: list[str]
    candidates_with_covered_concepts: int = Field(ge=0)
    candidates_without_covered_concepts: int = Field(ge=0)
    candidates_with_prerequisite_concepts: int = Field(ge=0)
    topic_candidate_counts: dict[str, int]
    difficulty_availability: dict[str, DifficultyAvailability]
    binary_covered_concept_weight: Literal[1.0]
    binary_weight_semantics: Literal["concept_presence_compatibility_marker"]
    raw_mapping_occurrence_count: int = Field(ge=0)
    raw_occurrence_count_used_for_ranking: Literal[False]


def load_concept_mapping_report(path: Path) -> BookEvidenceConceptMappingReport:
    """Load one strict source-aware concept mapping report."""

    try:
        return BookEvidenceConceptMappingReport.model_validate_json(path.read_bytes())
    except (OSError, ValidationError, ValueError) as exc:
        raise SourceAwareAdapterError(f"invalid concept mapping report {path}: {exc}") from exc


def _duplicates(values: list[str]) -> list[str]:
    return sorted(value for value, count in Counter(values).items() if count > 1)


def build_source_aware_matching_book_profiles(
    mapping: BookEvidenceConceptMappingReport,
    profiles: list[BookProfile],
) -> list[MatchingBookProfile]:
    """Join exact book IDs and replace only covered concepts with binary presence."""

    mapping_ids = [book.book_id for book in mapping.books]
    profile_ids = [book.book_id for book in profiles]
    duplicate_mapping = _duplicates(mapping_ids)
    duplicate_profiles = _duplicates(profile_ids)
    if duplicate_mapping or duplicate_profiles:
        raise SourceAwareAdapterError(
            "duplicate adapter book IDs; "
            f"mapping={duplicate_mapping}, book_profiles={duplicate_profiles}"
        )
    missing_in_mapping = sorted(set(profile_ids) - set(mapping_ids))
    missing_in_profiles = sorted(set(mapping_ids) - set(profile_ids))
    if missing_in_mapping or missing_in_profiles:
        raise SourceAwareAdapterError(
            "adapter book IDs differ; "
            f"missing_in_mapping={missing_in_mapping}, "
            f"missing_in_book_profiles={missing_in_profiles}"
        )

    mapping_by_id = {book.book_id: book for book in mapping.books}
    candidates: list[MatchingBookProfile] = []
    for profile in sorted(profiles, key=lambda item: item.book_id):
        source_mapping = mapping_by_id[profile.book_id]
        profile_topics = set(profile.concept_profile.topic_distribution)
        mapping_topics = set(source_mapping.topics)
        if mapping_topics != profile_topics:
            raise SourceAwareAdapterError(
                f"mapping topics for {profile.book_id} differ from its topic distribution; "
                f"mapping={sorted(mapping_topics)}, book_profile={sorted(profile_topics)}"
            )
        unknown_topics = sorted(
            {concept.topic_id for concept in source_mapping.concepts} - profile_topics
        )
        if unknown_topics:
            raise SourceAwareAdapterError(
                f"mapped concepts for {profile.book_id} are outside its topic distribution: "
                f"{unknown_topics}"
            )
        covered_ids = sorted({concept.concept_id for concept in source_mapping.concepts})
        difficulty = profile.difficulty_profile
        candidates.append(
            MatchingBookProfile(
                book_id=profile.book_id,
                topic_distribution=profile.concept_profile.topic_distribution,
                covered_concepts=[
                    MatchingConcept(concept=concept_id, weight=BINARY_PRESENCE_WEIGHT)
                    for concept_id in covered_ids
                ],
                prerequisite_concepts=[
                    MatchingConcept(concept=item.concept, weight=item.weight)
                    for item in profile.concept_profile.prerequisite_concepts
                ],
                lexical_difficulty=difficulty.lexical_difficulty,
                syntactic_complexity=difficulty.syntactic_complexity,
                concept_density=difficulty.concept_density,
                prerequisite_demand=difficulty.prerequisite_demand,
                feature_version=profile.feature_version,
                config_version=profile.config_version,
                config_hash=profile.config_hash,
            )
        )
    return candidates


def build_source_aware_adapter_report(
    mapping: BookEvidenceConceptMappingReport,
    profiles: list[BookProfile],
    candidates: list[MatchingBookProfile],
    concept_mapping_hash: str,
    source_book_profiles_hash: str,
    candidate_artifact_hash: str,
) -> SourceAwareMatchingAdapterReport:
    """Summarize a validated one-to-one join without adding ranking semantics."""

    mapping_ids = [book.book_id for book in mapping.books]
    profile_ids = [book.book_id for book in profiles]
    candidate_ids = [book.book_id for book in candidates]
    if candidate_ids != sorted(profile_ids) or len(candidate_ids) != len(set(candidate_ids)):
        raise SourceAwareAdapterError("candidate artifact does not match sorted source profiles")
    fields = (
        "lexical_difficulty",
        "syntactic_complexity",
        "concept_density",
        "prerequisite_demand",
    )
    topic_counts = Counter(
        topic
        for candidate in candidates
        for topic, weight in candidate.topic_distribution.items()
        if weight > 0
    )
    return SourceAwareMatchingAdapterReport(
        adapter_version=ADAPTER_VERSION,
        concept_mapping_report_version=mapping.report_version,
        concept_mapping_hash=concept_mapping_hash,
        matcher_version=mapping.matcher_version,
        matching_config_version=mapping.matching_config_version,
        matching_config_hash=mapping.matching_config_hash,
        source_book_profiles_hash=source_book_profiles_hash,
        candidate_artifact_hash=candidate_artifact_hash,
        source_feature_versions=sorted({profile.feature_version for profile in profiles}),
        source_config_versions=sorted({profile.config_version for profile in profiles}),
        source_config_hashes=sorted({profile.config_hash for profile in profiles}),
        mapping_book_count=len(mapping_ids),
        source_book_profile_count=len(profile_ids),
        joined_book_count=len(candidate_ids),
        missing_in_mapping=sorted(set(profile_ids) - set(mapping_ids)),
        missing_in_book_profiles=sorted(set(mapping_ids) - set(profile_ids)),
        duplicate_mapping_book_ids=_duplicates(mapping_ids),
        duplicate_book_profile_ids=_duplicates(profile_ids),
        candidates_with_covered_concepts=sum(bool(book.covered_concepts) for book in candidates),
        candidates_without_covered_concepts=sum(not book.covered_concepts for book in candidates),
        candidates_with_prerequisite_concepts=sum(
            bool(book.prerequisite_concepts) for book in candidates
        ),
        topic_candidate_counts=dict(sorted(topic_counts.items())),
        difficulty_availability={
            field: DifficultyAvailability(
                available=sum(getattr(book, field) is not None for book in candidates),
                missing=sum(getattr(book, field) is None for book in candidates),
            )
            for field in fields
        },
        binary_covered_concept_weight=BINARY_PRESENCE_WEIGHT,
        binary_weight_semantics="concept_presence_compatibility_marker",
        raw_mapping_occurrence_count=mapping.raw_match_occurrence_count,
        raw_occurrence_count_used_for_ranking=False,
    )
