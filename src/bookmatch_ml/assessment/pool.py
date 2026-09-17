"""Topic concept aggregation from provider-independent book profiles."""

from collections import defaultdict

from bookmatch_ml.assessment.schemas import (
    AssessmentEvidenceRef,
    ConceptBookSupport,
    TopicConcept,
    TopicConceptPool,
)
from bookmatch_ml.book.profile import build_book_profiles
from bookmatch_ml.config import LoadedAssessmentConfig, LoadedFeatureConfig
from bookmatch_ml.data.evidence import PROSE_DOCUMENT_TYPES, assemble_book_evidence
from bookmatch_ml.schemas import BookProfile, CanonicalDataset


class AssessmentBlueprintError(ValueError):
    """Canonical evidence or profile input cannot support a valid blueprint."""


def validate_profile_handoff(
    dataset: CanonicalDataset,
    profiles: list[BookProfile],
    feature_config: LoadedFeatureConfig,
) -> None:
    """Check profile identity, provenance, and feature rules against canonical input."""

    canonical_ids = {book.book_id for book in dataset.books}
    profile_ids = {book.book_id for book in profiles}
    if canonical_ids != profile_ids:
        raise AssessmentBlueprintError(
            "book profiles do not match canonical dataset: "
            f"missing {sorted(canonical_ids - profile_ids)}, "
            f"extra {sorted(profile_ids - canonical_ids)}"
        )
    toc = {entry.toc_entry_id: entry for entry in dataset.toc}
    documents = {document.document_id: document for document in dataset.documents}
    for profile in profiles:
        if profile.config_hash != feature_config.content_hash:
            raise AssessmentBlueprintError(
                f"book profile feature config hash mismatch: {profile.book_id}"
            )
        if profile.config_version != feature_config.config.config_version:
            raise AssessmentBlueprintError(
                f"book profile feature config version mismatch: {profile.book_id}"
            )
        for concept in [
            *profile.concept_profile.covered_concepts,
            *profile.concept_profile.prerequisite_concepts,
        ]:
            for reference in concept.evidence:
                if reference.evidence_type == "toc":
                    entry = toc.get(reference.evidence_id)
                    valid = entry is not None and entry.book_id == profile.book_id
                else:
                    document = documents.get(reference.evidence_id)
                    valid = (
                        document is not None
                        and document.book_id == profile.book_id
                        and document.document_type == reference.evidence_type
                    )
                if not valid:
                    raise AssessmentBlueprintError(
                        f"invalid concept evidence reference {reference.evidence_id} "
                        f"for book {profile.book_id}"
                    )
        for analyzed in profile.difficulty_profile.documents:
            document = documents.get(analyzed.document_id)
            if (
                document is None
                or document.book_id != profile.book_id
                or document.document_type != analyzed.document_type
                or document.document_type not in PROSE_DOCUMENT_TYPES
            ):
                raise AssessmentBlueprintError(
                    f"invalid analyzed prose document {analyzed.document_id} "
                    f"for book {profile.book_id}"
                )
    expected = build_book_profiles(assemble_book_evidence(dataset), feature_config)
    if sorted(profiles, key=lambda item: item.book_id) != sorted(
        expected, key=lambda item: item.book_id
    ):
        raise AssessmentBlueprintError(
            "book profiles are stale or inconsistent with current canonical data and feature config"
        )


def build_topic_concept_pool(
    topic_id: str,
    profiles: list[BookProfile],
    feature_config: LoadedFeatureConfig,
    assessment_config: LoadedAssessmentConfig,
) -> TopicConceptPool:
    """Merge identical names within each role without inferring concept difficulty."""

    features = feature_config.config
    if topic_id not in features.concept.topics or topic_id not in features.prerequisite.topics:
        raise AssessmentBlueprintError(f"unsupported assessment topic: {topic_id}")
    topic_books = sorted(
        (
            profile
            for profile in profiles
            if profile.concept_profile.topic_distribution.get(topic_id, 0.0) > 0
        ),
        key=lambda item: item.book_id,
    )
    if not topic_books:
        raise AssessmentBlueprintError(f"no book profiles for topic: {topic_id}")
    names_by_role = {
        "covered": set(features.concept.topics[topic_id].root),
        "prerequisite": set(features.prerequisite.topics[topic_id].root),
    }
    other_topic_names = {
        "covered": {
            name
            for other_topic, lexicon in features.concept.topics.items()
            if other_topic != topic_id
            for name in lexicon.root
        },
        "prerequisite": {
            name
            for other_topic, lexicon in features.prerequisite.topics.items()
            if other_topic != topic_id
            for name in lexicon.root
        },
    }
    grouped: dict[tuple[str, str], list[ConceptBookSupport]] = defaultdict(list)
    for profile in topic_books:
        for role, concepts in (
            ("covered", profile.concept_profile.covered_concepts),
            ("prerequisite", profile.concept_profile.prerequisite_concepts),
        ):
            seen: set[str] = set()
            for concept in concepts:
                if concept.concept not in names_by_role[role]:
                    if concept.concept in other_topic_names[role]:
                        continue
                    raise AssessmentBlueprintError(
                        f"concept {concept.concept!r} is outside {topic_id} {role} lexicon"
                    )
                if concept.concept in seen:
                    raise AssessmentBlueprintError(
                        f"duplicate {role} concept {concept.concept!r} in {profile.book_id}"
                    )
                seen.add(concept.concept)
                grouped[(concept.concept, role)].append(
                    ConceptBookSupport(
                        book_id=profile.book_id,
                        weight=concept.weight,
                        prerequisite_method=(concept.method if role == "prerequisite" else None),
                        evidence=[
                            AssessmentEvidenceRef(
                                concept_id=concept.concept,
                                book_id=profile.book_id,
                                evidence_type=reference.evidence_type,
                                evidence_id=reference.evidence_id,
                                mention_count=reference.mention_count,
                            )
                            for reference in concept.evidence
                        ],
                    )
                )
    priority = assessment_config.config.priority
    priority_total = priority.mean_book_weight + priority.book_coverage_rate
    concepts: list[TopicConcept] = []
    for (concept_id, role), supports in grouped.items():
        supports.sort(key=lambda item: item.book_id)
        coverage_rate = len(supports) / len(topic_books)
        mean_weight = sum(item.weight for item in supports) / len(supports)
        concepts.append(
            TopicConcept(
                concept_id=concept_id,
                topic_id=topic_id,
                role=role,
                book_coverage_count=len(supports),
                book_coverage_rate=coverage_rate,
                mean_book_weight=mean_weight,
                assessment_priority=(
                    priority.mean_book_weight * mean_weight
                    + priority.book_coverage_rate * coverage_rate
                )
                / priority_total,
                evidence_types=sorted(
                    {reference.evidence_type for item in supports for reference in item.evidence}
                ),
                prerequisite_methods=sorted(
                    {
                        item.prerequisite_method
                        for item in supports
                        if item.prerequisite_method is not None
                    }
                ),
                book_support=supports,
            )
        )
    concepts.sort(
        key=lambda item: (
            0 if item.role == "covered" else 1,
            -item.assessment_priority,
            item.concept_id,
        )
    )
    return TopicConceptPool(
        topic_id=topic_id,
        topic_book_count=len(topic_books),
        covered_concept_count=sum(item.role == "covered" for item in concepts),
        prerequisite_concept_count=sum(item.role == "prerequisite" for item in concepts),
        concepts=concepts,
        pool_version=assessment_config.config.pool_version,
        feature_config_version=features.config_version,
        feature_config_hash=feature_config.content_hash,
        assessment_config_version=assessment_config.config.config_version,
        assessment_config_hash=assessment_config.content_hash,
    )
