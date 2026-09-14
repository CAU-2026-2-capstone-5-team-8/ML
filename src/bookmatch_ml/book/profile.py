"""Book-level aggregation of concept and difficulty profiles."""

from bookmatch_ml.book.concepts import build_concept_profile
from bookmatch_ml.book.difficulty import build_difficulty_profile
from bookmatch_ml.config import LoadedFeatureConfig
from bookmatch_ml.schemas import BookEvidence, BookProfile


def build_book_profile(
    evidence: BookEvidence,
    loaded_config: LoadedFeatureConfig,
) -> BookProfile:
    config = loaded_config.config
    return BookProfile(
        book_id=evidence.book_id,
        concept_profile=build_concept_profile(evidence, loaded_config),
        difficulty_profile=build_difficulty_profile(evidence, loaded_config),
        evidence_coverage=evidence.coverage,
        feature_version=config.book_profile_version,
        config_version=config.config_version,
        config_hash=loaded_config.content_hash,
    )


def build_book_profiles(
    evidence: list[BookEvidence],
    loaded_config: LoadedFeatureConfig,
) -> list[BookProfile]:
    return [
        build_book_profile(item, loaded_config)
        for item in sorted(evidence, key=lambda item: item.book_id)
    ]
